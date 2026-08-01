"""
router.py — LLM router (ARCHITECTURE.md §6).

Only called when rules.py doesn't resolve a message. Takes the full
MessageContext (already built by context_builder.py — never re-fetches
anything) and the rule layer's outcome, and produces the final 6-column
output row via a structured/schema-forced LLM call.

Determinism is a property of the cache, not the model: temperature 0 does not
guarantee bit-identical output across API calls, so the structured result is
cached keyed by a hash of the exact serialized MessageContext. Re-running on
unchanged input replays the cached decision instead of re-querying.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from config import (
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    LLM_FALLBACK_MODEL,
    LLM_FALLBACK_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
)
from rules import DOMAIN_AGE_RATIO_THRESHOLD, REPORT_RATE_OUTLIER_THRESHOLD

logger = logging.getLogger("router")

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache" / "router"

VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_MESSAGE_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}

_SYSTEM_PROMPT = """You are the routing decision step of a WhatsApp message notification \
system. For every message you decide one of: notify (interrupt the user now), digest \
(wait for later), or mute (suppress as low-value/repetitive/unwanted/suspicious/unsafe).

CORE PRINCIPLE — personalization is not optional: the exact same message content can and \
should route differently for different users, depending on their relationship with the \
sender, opt-in/opt-out status, engagement history, and group mute state. Message content \
alone is never sufficient to decide; always weigh it against who is receiving it and their \
history with this sender/group/business.

A deterministic rule layer already ran against this message and did NOT resolve it with \
confidence — you are only invoked because the case is genuinely ambiguous, novel, or the \
rules' fixed thresholds weren't met. You are given the exact same raw signal fields the \
rule layer checks (domain match/age vs a business-account-population outlier threshold, \
promotion opt-out status, business report rate vs an outlier threshold, group-mute state, \
cold-start flag, any rule-5 escalation flag), even though no rule fired on them — reason \
over *why* they weren't enough to auto-resolve, don't discard them. A domain mismatch alone \
without any corroborating urgency/payment language, for example, is not automatically a \
scam — reason about the specific content, not just the presence of a flagged signal.

Respond with ONLY this JSON object, no prose, no markdown fences:
{
  "action": "notify" | "digest" | "mute",
  "message_type": "personal" | "urgent" | "event" | "payment" | "business_update" | "promotion" | "greeting" | "forward" | "spam" | "scam" | "unknown",
  "reason": "<one or two sentences citing concrete context fields you weighed — sender trust, relationship, urgency, repetition, opt-out status, group mute state — not generic filler>",
  "confidence": <float 0.0-1.0, your genuine self-assessed confidence in this decision>,
  "evidence_message_ids": "<semicolon-separated message_ids drawn ONLY from history_candidates or cross_user_safety_evidence in the provided context that support your reasoning, or \\"none\\" if none directly support it>"
}"""


# Bump whenever _SYSTEM_PROMPT, _validate_and_clean, or _adjust_confidence
# logic changes. Included in the cache key so a logic change (e.g. a new
# confidence-adjustment rule) can't silently keep serving a decision the old
# logic produced for an unchanged MessageContext — determinism is "same
# input+logic -> same output", not "same input -> output frozen forever."
_ROUTER_LOGIC_VERSION = "4"


def _context_hash(context: dict) -> str:
    serialized = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(f"{_ROUTER_LOGIC_VERSION}:{serialized}".encode()).hexdigest()


def _cache_path(context_hash: str) -> Path:
    return CACHE_DIR / f"{context_hash}.json"


def _groq_client():
    if LLM_PROVIDER != "groq":
        raise NotImplementedError(f"LLM_PROVIDER={LLM_PROVIDER!r} not implemented; only 'groq' is wired up.")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set (checked .env / environment).")
    from groq import Groq

    return Groq(api_key=GROQ_API_KEY)


def _retryable(exc: Exception) -> bool:
    """Same pattern as media/image_processor.py: per-minute 429 (TPM) and 503
    are transient, worth a short backoff. A 429 on the per-*day* quota (TPD)
    is not — reproduced directly running main.py over the full dataset
    (100000 TPD on llama-3.3-70b-versatile, exhausted by this session's own
    testing) — retrying wouldn't clear for hours, so it's treated as
    non-retryable and falls straight to the Gemini fallback instead."""
    from groq import APIStatusError

    if not isinstance(exc, APIStatusError):
        return False
    if exc.status_code == 429:
        body = exc.body if isinstance(exc.body, dict) else {}
        message = (body.get("error") or {}).get("message", "").lower()
        if "per day" in message or "(tpd)" in message:
            return False
        return True
    if exc.status_code == 503:
        return True
    return False


def _retry_delay(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return min(2**attempt, 30)


def _call_with_retry(fn, max_attempts: int = 4):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            if not _retryable(exc) or attempt == max_attempts - 1:
                raise
            wait = _retry_delay(exc, attempt)
            logger.warning(
                "Groq router call failed (%s), retrying in %.1fs [attempt %d/%d]",
                exc, wait, attempt + 1, max_attempts,
            )
            time.sleep(wait)


def _call_groq(messages: list[dict]) -> str:
    client = _groq_client()
    resp = _call_with_retry(
        lambda: client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
    )
    return resp.choices[0].message.content


def _call_gemini(messages: list[dict]) -> str:
    """Translates the OpenAI-style messages list (system/user/assistant) into
    Gemini's system_instruction + contents format."""
    if LLM_FALLBACK_PROVIDER != "google":
        raise NotImplementedError(f"LLM_FALLBACK_PROVIDER={LLM_FALLBACK_PROVIDER!r} not implemented.")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set (checked .env / environment).")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    system_instruction = None
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_instruction = m["content"]
        elif m["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m["content"])]))
        elif m["role"] == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=m["content"])]))

    resp = client.models.generate_content(
        model=LLM_FALLBACK_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return resp.text


def _call_llm(messages: list[dict]) -> tuple[str, str]:
    """Returns (raw_json_text, provider_used). Tries Groq first; on exhausted
    retries falls back to Gemini entirely rather than continuing to retry a
    single point of failure. Raises only if both providers fail."""
    try:
        return _call_groq(messages), f"groq:{LLM_MODEL}"
    except Exception as groq_exc:
        logger.warning(
            "Groq router exhausted retries (%s) -> falling back to %s:%s",
            groq_exc, LLM_FALLBACK_PROVIDER, LLM_FALLBACK_MODEL,
        )
        try:
            return _call_gemini(messages), f"{LLM_FALLBACK_PROVIDER}:{LLM_FALLBACK_MODEL}"
        except Exception as fallback_exc:
            logger.error(
                "Both router providers failed: groq=%s, fallback=%s", groq_exc, fallback_exc,
            )
            raise


def _media_uncertain(context: dict) -> bool:
    media = context.get("media") or {}
    if media.get("media_extraction_failed"):
        return True
    return media.get("type") is not None and media.get("extracted_text") is None


def _signal_summary(context: dict) -> dict:
    """The same raw signal fields rules.py checks, computed the same way (same
    thresholds, imported from rules.py so there's one source of truth) —
    included even though no rule fired, so the LLM reasons over why they
    weren't enough rather than seeing a stripped-down context (§6)."""
    message = context["message"]
    sender_context = context.get("sender_context") or {}
    business = sender_context.get("business")
    relationship = sender_context.get("relationship")
    membership = sender_context.get("membership")
    history = context.get("history_candidates") or []
    cross = context.get("cross_user_safety_evidence") or []

    summary: dict = {
        "conversation_type": message["conversation_type"],
        "cold_start_no_history_no_cross_user_evidence": not history and not cross,
        "history_candidates_count": len(history),
        "cross_user_safety_evidence_count": len(cross),
        "media_pending_or_extraction_failed": _media_uncertain(context),
    }

    if business:
        domain_mismatch = business["domain_used_by_sender"] != business["official_domain"]
        try:
            age_ratio = int(business["domain_used_by_sender_age_days"]) / int(business["account_age_days"])
        except (ValueError, ZeroDivisionError):
            age_ratio = None
        try:
            report_rate = int(business["user_reports_30d"]) / int(business["messages_sent_30d"])
        except (ValueError, ZeroDivisionError):
            report_rate = None
        summary["business_signals"] = {
            "domain_used_by_sender": business["domain_used_by_sender"],
            "official_domain": business["official_domain"],
            "domain_mismatch": domain_mismatch,
            "domain_age_ratio": age_ratio,
            "domain_age_ratio_outlier_below": DOMAIN_AGE_RATIO_THRESHOLD,
            "report_rate": report_rate,
            "report_rate_outlier_above": REPORT_RATE_OUTLIER_THRESHOLD,
            "report_rate_is_outlier": report_rate is not None and report_rate > REPORT_RATE_OUTLIER_THRESHOLD,
        }
    if relationship:
        summary["promotions_opted_out"] = bool(relationship.get("promotions_opted_out_at"))
    if membership:
        summary["group_muted_by_user"] = membership.get("group_muted_by_user") == "1"

    return summary


def _partial_safety_flags(context: dict) -> list[str]:
    """Individual halves of a mute-rule condition that are true on their own
    but did not, by themselves, resolve a rule. Rule 1 (rules.py) needs
    domain_signal (mismatch OR young registered-domain) AND payment/urgency
    language; reaching the router with domain_signal true means only the
    language half wasn't met — the domain half is still a real, unresolved
    flag, not a cleared one. Rule 3 has no second half (an outlier report
    rate resolves unconditionally), so it can't structurally reach the router
    today, but is still checked here defensively in case that ever changes.
    Used to cap confidence regardless of how strong the surrounding
    relationship/history context looks (§6/msg_086: a real confirmed-booking
    relationship doesn't retroactively un-spoof a mismatched domain)."""
    business = (context.get("sender_context") or {}).get("business")
    if not business:
        return []

    flags = []
    domain_mismatch = business["domain_used_by_sender"] != business["official_domain"]
    if domain_mismatch:
        flags.append(
            f"domain mismatch (domain_used_by_sender={business['domain_used_by_sender']} "
            f"vs official_domain={business['official_domain']})"
        )
    else:
        try:
            age_ratio = int(business["domain_used_by_sender_age_days"]) / int(business["account_age_days"])
        except (ValueError, ZeroDivisionError):
            age_ratio = None
        if age_ratio is not None and age_ratio < DOMAIN_AGE_RATIO_THRESHOLD:
            flags.append(f"sending domain registered recently relative to account age (ratio={age_ratio:.2f})")

    try:
        report_rate = int(business["user_reports_30d"]) / int(business["messages_sent_30d"])
    except (ValueError, ZeroDivisionError):
        report_rate = None
    if report_rate is not None and report_rate > REPORT_RATE_OUTLIER_THRESHOLD:
        flags.append(f"business report rate is an outlier ({report_rate:.1%} > {REPORT_RATE_OUTLIER_THRESHOLD:.1%})")

    return flags


def _ensure_reason_names_flags(reason: str, partial_flags: list[str]) -> str:
    lowered = reason.lower()
    if any(kw in lowered for kw in ("domain", "report rate", "report_rate")):
        return reason
    return f"{reason} (Note: rule layer flagged but did not resolve: {'; '.join(partial_flags)}.)"


def _adjust_confidence(confidence: float, context: dict, action: str) -> float:
    """§6: down (cap ~0.5-0.6) when context is thin — new/unknown sender, no
    history_candidates, media processing failed/uncertain. Up when
    history_candidates strongly agree with the decision (consistent past
    reaction pattern). Floor is applied after the thin-context cap: if a
    message has both uncertain media AND strong corroborating history, the
    real corroboration is more informative than the media gap, so it can lift
    that cap back up. The partial-safety-flag cap below is applied last and
    is not liftable by history strength — see _partial_safety_flags()."""
    history = context.get("history_candidates") or []
    cross = context.get("cross_user_safety_evidence") or []
    thin = (not history and not cross) or _media_uncertain(context)

    if thin:
        confidence = min(confidence, 0.6)

    if len(history) >= 2:
        mute_like = sum(
            1 for h in history
            if (h.get("reaction") or {}).get("muted_after_message") == "1"
            or (h.get("reaction") or {}).get("message_reported") == "1"
            or (h.get("reaction") or {}).get("notification_dismissed") == "1"
        )
        engage_like = sum(
            1 for h in history if (h.get("reaction") or {}).get("message_opened") == "1"
        )
        total = len(history)
        if action == "mute" and mute_like / total >= 0.6:
            confidence = max(confidence, 0.75)
        elif action in ("notify", "digest") and engage_like / total >= 0.6:
            confidence = max(confidence, 0.75)

    if _partial_safety_flags(context):
        confidence = min(confidence, 0.65)

    return max(0.0, min(1.0, confidence))


def _validate_and_clean(data: dict, context: dict) -> dict:
    action = data.get("action")
    message_type = data.get("message_type")
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action from LLM: {action!r}")
    if message_type not in VALID_MESSAGE_TYPES:
        raise ValueError(f"invalid message_type from LLM: {message_type!r}")

    valid_ids = {h["message_id"] for h in context.get("history_candidates") or []}
    valid_ids |= {c["message_id"] for c in context.get("cross_user_safety_evidence") or []}
    raw_evidence = str(data.get("evidence_message_ids") or "none")
    candidate_ids = [] if raw_evidence.strip().lower() == "none" else [
        e.strip() for e in raw_evidence.split(";") if e.strip()
    ]
    dropped = [e for e in candidate_ids if e not in valid_ids]
    if dropped:
        logger.warning("dropping evidence_message_ids not in this context's candidates: %s", dropped)
    cleaned_ids = [e for e in candidate_ids if e in valid_ids]
    evidence_message_ids = ";".join(cleaned_ids) if cleaned_ids else "none"

    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    confidence = _adjust_confidence(confidence, context, action)

    reason = str(data.get("reason") or "").strip() or "No reason provided by model."
    if action != "mute":
        partial_flags = _partial_safety_flags(context)
        if partial_flags:
            reason = _ensure_reason_names_flags(reason, partial_flags)

    return {
        "action": action,
        "message_type": message_type,
        "reason": reason,
        "confidence": round(confidence, 2),
        "evidence_message_ids": evidence_message_ids,
    }


def route(context: dict, rule_result: dict | None = None, max_attempts: int = 3) -> dict:
    """Returns {"action", "message_type", "reason", "confidence",
    "evidence_message_ids"}. Cached by a hash of the serialized context — a
    repeat call with the same context never re-calls the API."""
    ctx_hash = _context_hash(context)
    cache_file = _cache_path(ctx_hash)
    if cache_file.exists():
        logger.info("cache HIT context_hash=%s -> %s", ctx_hash[:16], cache_file)
        return json.loads(cache_file.read_text())["result"]

    logger.info("cache MISS context_hash=%s -> calling %s/%s", ctx_hash[:16], LLM_PROVIDER, LLM_MODEL)

    user_payload = {
        "message_context": context,
        "rule_layer_signals": _signal_summary(context),
        "rule_layer_outcome": rule_result or {"resolved": False},
    }

    last_error = None
    result = None
    provider_used = None
    # Guided retry, not blind retry: temperature=0 is not bit-identical across
    # calls (a known API-level limitation, not a bug — see the caching
    # rationale above), so a naive re-ask can get lucky or can just repeat the
    # same invalid answer every time (reproduced deterministically on
    # msg_004/business_004: the model anchors on the literal business.category
    #="healthcare" field and invents "healthcare_update" by analogy to
    # "business_update", identically on 3/3 blind attempts). Feeding the prior
    # invalid response and the exact validation error back in makes the
    # correction reliable instead of hoping for API noise to save it.
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, default=str)},
    ]
    for attempt in range(max_attempts):
        raw_content, provider_used = _call_llm(messages)
        raw = json.loads(raw_content, strict=False)
        try:
            result = _validate_and_clean(raw, context)
            break
        except ValueError as exc:
            last_error = exc
            logger.warning(
                "LLM output failed enum validation (%s), retrying [attempt %d/%d]",
                exc, attempt + 1, max_attempts,
            )
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({
                "role": "user",
                "content": (
                    f"That response was invalid: {exc}. action must be exactly one of "
                    f"{sorted(VALID_ACTIONS)}; message_type must be exactly one of "
                    f"{sorted(VALID_MESSAGE_TYPES)} — do not invent a new category even if it "
                    f"seems more specific (e.g. a healthcare business's category field does not "
                    f"mean message_type should be a healthcare-specific value; use business_update). "
                    f"Respond again with the same JSON schema, corrected."
                ),
            })
    if result is None:
        raise RuntimeError(f"LLM router failed enum validation after {max_attempts} attempts: {last_error}")

    logger.info("router decision complete context_hash=%s provider=%s", ctx_hash[:16], provider_used)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"context_hash": ctx_hash, "provider": provider_used, "result": result}, indent=2)
    )
    return result

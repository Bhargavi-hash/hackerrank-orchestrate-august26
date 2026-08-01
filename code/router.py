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
from pathlib import Path

from config import GROQ_API_KEY, LLM_MODEL, LLM_PROVIDER
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


def _context_hash(context: dict) -> str:
    serialized = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _cache_path(context_hash: str) -> Path:
    return CACHE_DIR / f"{context_hash}.json"


def _client():
    if LLM_PROVIDER != "groq":
        raise NotImplementedError(f"LLM_PROVIDER={LLM_PROVIDER!r} not implemented; only 'groq' is wired up.")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set (checked .env / environment).")
    from groq import Groq

    return Groq(api_key=GROQ_API_KEY)


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


def _adjust_confidence(confidence: float, context: dict, action: str) -> float:
    """§6: down (cap ~0.5-0.6) when context is thin — new/unknown sender, no
    history_candidates, media processing failed/uncertain. Up when
    history_candidates strongly agree with the decision (consistent past
    reaction pattern). Floor is applied after the cap: if a message has both
    uncertain media AND strong corroborating history, the real corroboration
    is more informative than the media gap, so it can lift the cap back up."""
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

    client = _client()
    last_error = None
    result = None
    for attempt in range(max_attempts):
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, default=str)},
            ],
        )
        raw = json.loads(resp.choices[0].message.content, strict=False)
        try:
            result = _validate_and_clean(raw, context)
            break
        except ValueError as exc:
            last_error = exc
            logger.warning(
                "LLM output failed enum validation (%s), retrying [attempt %d/%d]",
                exc, attempt + 1, max_attempts,
            )
    if result is None:
        raise RuntimeError(f"LLM router failed enum validation after {max_attempts} attempts: {last_error}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"context_hash": ctx_hash, "model": LLM_MODEL, "result": result}, indent=2)
    )
    return result

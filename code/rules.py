"""
rules.py — deterministic override layer (ARCHITECTURE.md §5).

Runs before any LLM call. Each rule is traceable to real columns in the
MessageContext produced by context_builder.py. Rules run in order (1-7);
the first one that resolves wins. Rule 5 never resolves — it only raises an
escalation flag for the LLM router. If nothing resolves, apply_rules()
returns {"resolved": False, ...} and the message goes to router.py.
"""

from __future__ import annotations

import re
from typing import Any

# --- thresholds, derived from dataset/business_accounts.csv, not hand-picked ---

# IQR outlier threshold (Q3 + 1.5*IQR) over report_rate = user_reports_30d /
# messages_sent_30d across all 110 rows in business_accounts.csv.
REPORT_RATE_OUTLIER_THRESHOLD = 0.0122

# domain_used_by_sender_age_days / account_age_days is cleanly bimodal in
# business_accounts.csv: a low cluster at 0.09-0.48 and a legitimate cluster
# at 0.78+, with a clean gap between them. 0.6 sits in that gap.
DOMAIN_AGE_RATIO_THRESHOLD = 0.6

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "to", "of", "in", "on", "at", "for", "with", "this", "that",
    "it", "you", "your", "i", "we", "us", "our", "as", "by", "from", "will",
    "would", "can", "could", "please", "hi", "hello", "hey", "thanks", "thank",
}

_URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "right away", "act now", "act fast",
    "hurry", "last chance", "final warning", "limited time", "expire",
    "expiring", "expires", "deadline", "within minutes", "within the hour",
    "today only", "before it's too late", "don't ignore", "respond now",
]
_PAYMENT_KEYWORDS = [
    "pay ", "payment", "otp", "pin", "cvv", "card number", "upi", "qr code",
    "bank account", "login code", "verify your", "password", "kyc", "refund",
    "invoice", "bill", "wallet", "account number", "click the link",
    "click here", "confirm your details", "reset your password",
]
_PROMO_KEYWORDS = [
    "% off", "percent off", "discount", " sale", "offer", "deal", "promo",
    "reply stop", "unsubscribe", "cashback", "voucher", "coupon", "flat ",
]
_GREETING_KEYWORDS = [
    "good morning", "good night", "happy ", "blessings", "stay blessed",
    "have a great day",
]
_EVENT_KEYWORDS = [
    "delivery", "shipment", "order", "tracking", "schedule", "appointment",
    "booking", "checkup", "meeting", "pickup",
]


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS


def _has_signal(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _text_blob(context: dict) -> str:
    msg = context["message"]
    media = context.get("media") or {}
    parts = [msg.get("message_text") or "", media.get("extracted_text") or ""]
    return "\n".join(p for p in parts if p)


def _media_pending(context: dict) -> bool:
    """True when the message has media but it hasn't been OCR'd/transcribed yet
    (extracted_text is None) — language-based conditions can't be evaluated."""
    media = context.get("media") or {}
    return media.get("type") is not None and media.get("extracted_text") is None


def _infer_message_type(text: str, conv_type: str) -> str:
    """Deterministic keyword-based best-fit classifier (§5 rules 4/7's 'best-fit'
    column) — not an LLM call, just cheap heuristics over the fixed enum."""
    if _has_signal(text, _PROMO_KEYWORDS):
        return "promotion"
    if _has_signal(text, _GREETING_KEYWORDS):
        return "greeting"
    if _has_signal(text, _PAYMENT_KEYWORDS) or "statement" in text.lower():
        return "business_update" if conv_type == "business" else "payment"
    if _has_signal(text, _EVENT_KEYWORDS):
        return "business_update" if conv_type == "business" else "event"
    if conv_type == "business":
        return "business_update"
    return "unknown"


def _mute_supporting_ids(context: dict) -> list[str]:
    """Every history_candidates/cross_user_safety_evidence entry whose own
    reaction directly corroborates a mute decision (message_reported==1 or
    muted_after_message==1). Used to backfill evidence_message_ids for any
    rule that resolves to mute, regardless of which specific condition inside
    that rule actually fired — a scam/report-rate/opt-out/repetition finding
    should cite real supporting history when it exists, not just the columns
    that happened to trip the rule's own threshold."""
    ids = []
    for h in context.get("history_candidates") or []:
        reaction = h.get("reaction") or {}
        if reaction.get("message_reported") == "1" or reaction.get("muted_after_message") == "1":
            ids.append(h["message_id"])
    # cross_user_safety_evidence is already filtered to reported/muted rows by context_builder.
    for c in context.get("cross_user_safety_evidence") or []:
        ids.append(c["message_id"])
    return ids


# --- rule 1: scam domain + payment/urgency language -------------------------

def rule_1_scam_domain(context: dict) -> dict | None:
    business = context["sender_context"].get("business")
    if not business:
        return None

    domain_mismatch = business["domain_used_by_sender"] != business["official_domain"]
    try:
        age_ratio = int(business["domain_used_by_sender_age_days"]) / int(business["account_age_days"])
    except (ValueError, ZeroDivisionError):
        age_ratio = 1.0
    young_domain = age_ratio < DOMAIN_AGE_RATIO_THRESHOLD

    if not (domain_mismatch or young_domain):
        return None

    if _media_pending(context):
        media_type = context["media"]["type"]
        return {
            "resolved": False,
            "pending": True,
            "rule": 1,
            "note": (
                f"Domain signal present (domain_used_by_sender={business['domain_used_by_sender']} "
                f"vs official_domain={business['official_domain']}) but message has un-transcribed "
                f"{media_type} media; cannot evaluate payment/urgency language yet. Pending re-check "
                f"once media/{'image' if media_type == 'image' else 'voice'}_processor.py exists."
            ),
        }

    text = _text_blob(context)
    urgency = _has_signal(text, _URGENCY_KEYWORDS)
    payment = _has_signal(text, _PAYMENT_KEYWORDS)
    if not (urgency or payment):
        return None

    if domain_mismatch:
        domain_clause = f"Sender domain {business['domain_used_by_sender']} does not match official domain {business['official_domain']}"
    else:
        domain_clause = (
            f"Sending domain registered only {business['domain_used_by_sender_age_days']} days ago "
            f"relative to account age {business['account_age_days']} days"
        )
    language_clause = " and ".join(
        label for label, hit in (("payment", payment), ("urgency", urgency)) if hit
    )
    reason = f"{domain_clause}; message contains {language_clause} language"

    confidence = 0.95 if (domain_mismatch and urgency and payment) else 0.9
    return {
        "resolved": True,
        "rule": 1,
        "action": "mute",
        "message_type": "scam",
        "confidence": confidence,
        "reason": reason,
        "evidence_message_ids": "none",  # backfilled by apply_rules() from supporting history/cross-user evidence
    }


# --- rule 2: opted out of promotions but message is promotional -------------

def rule_2_opted_out_promo(context: dict) -> dict | None:
    relationship = context["sender_context"].get("relationship")
    if not relationship or not relationship.get("promotions_opted_out_at"):
        return None
    if _media_pending(context):
        return {
            "resolved": False,
            "pending": True,
            "rule": 2,
            "note": "User opted out of promotions but message media isn't transcribed yet; cannot confirm promotional content.",
        }

    text = _text_blob(context)
    if not _has_signal(text, _PROMO_KEYWORDS):
        return None

    history = context.get("history_candidates") or []
    was_reported = any((h.get("reaction") or {}).get("message_reported") == "1" for h in history)
    message_type = "spam" if was_reported else "promotion"

    reason = (
        f"User opted out of promotions from this business on {relationship['promotions_opted_out_at']} "
        f"(promotions_opted_out_at set); message reads as promotional content"
    )
    return {
        "resolved": True,
        "rule": 2,
        "action": "mute",
        "message_type": message_type,
        "confidence": 0.88,
        "reason": reason,
        "evidence_message_ids": "none",  # backfilled by apply_rules() from supporting history/cross-user evidence
    }


# --- rule 3: business report-rate outlier -----------------------------------

def rule_3_report_rate_outlier(context: dict) -> dict | None:
    business = context["sender_context"].get("business")
    if not business:
        return None
    try:
        sent = int(business["messages_sent_30d"])
        reports = int(business["user_reports_30d"])
    except ValueError:
        return None
    if sent == 0:
        return None
    rate = reports / sent
    if rate <= REPORT_RATE_OUTLIER_THRESHOLD:
        return None

    domain_mismatch = business["domain_used_by_sender"] != business["official_domain"]
    message_type = "scam" if domain_mismatch else "spam"
    if rate >= REPORT_RATE_OUTLIER_THRESHOLD * 4:
        confidence = 0.9
    elif rate >= REPORT_RATE_OUTLIER_THRESHOLD * 2:
        confidence = 0.85
    else:
        confidence = 0.8

    reason = (
        f"business.user_reports_30d={reports} / messages_sent_30d={sent} = {rate:.1%} report rate, "
        f"exceeding the dataset's IQR outlier threshold of {REPORT_RATE_OUTLIER_THRESHOLD:.1%}"
    )
    return {
        "resolved": True,
        "rule": 3,
        "action": "mute",
        "message_type": message_type,
        "confidence": confidence,
        "reason": reason,
        "evidence_message_ids": "none",  # backfilled by apply_rules() from supporting history/cross-user evidence
    }


# --- rules 4 & 5: muted group ------------------------------------------------

def rule_4_5_muted_group(context: dict) -> dict | None:
    membership = context["sender_context"].get("membership")
    if not membership or membership.get("group_muted_by_user") != "1":
        return None

    text = _text_blob(context)
    user_id = context["message"]["user_id"]
    mentioned = f"@{user_id}".lower() in text.lower()
    urgent = _has_signal(text, _URGENCY_KEYWORDS)
    today = "today" in text.lower() or "now" in text.lower()
    urgent_deadline_today = urgent and today

    if mentioned or urgent_deadline_today:
        cue = f"a direct @{user_id} mention" if mentioned else "explicit urgent + deadline-today language"
        return {
            "resolved": False,
            "escalation": True,
            "rule": 5,
            "escalation_reason": (
                f"group_members.group_muted_by_user=1 for this user/group, but the message contains "
                f"{cue}; not auto-muting — deferring notify-vs-digest decision to the LLM router."
            ),
        }

    try:
        forwarded = int(context["message"].get("forwarded_count") or 0)
    except ValueError:
        forwarded = 0

    if forwarded > 0:
        return {
            "resolved": True,
            "rule": 4,
            "action": "mute",
            "message_type": "forward",
            "confidence": 0.85,
            "reason": (
                f"group_members.group_muted_by_user=1 and forwarded_count={forwarded}; "
                f"no direct mention or urgent deadline-today language found"
            ),
            "evidence_message_ids": "none",
        }

    return {
        "resolved": True,
        "rule": 4,
        "action": "digest",
        "message_type": _infer_message_type(text, context["message"]["conversation_type"]),
        "confidence": 0.7,
        "reason": (
            "group_members.group_muted_by_user=1; no direct mention or urgent deadline-today "
            "language found, and message is not a forward — treated as low-priority informational "
            "content rather than junk"
        ),
        "evidence_message_ids": "none",
    }


# --- rule 6: repetition of dismissed/ignored content -------------------------

def rule_6_repetition(context: dict) -> dict | None:
    query_tokens = _tokenize(context["message"].get("message_text"))
    if not query_tokens:
        return None

    best = None
    for h in context.get("history_candidates") or []:
        h_tokens = _tokenize(h.get("message_text"))
        if not h_tokens:
            continue
        overlap = len(query_tokens & h_tokens) / len(query_tokens | h_tokens)
        reaction = h.get("reaction") or {}
        ignored = reaction.get("notification_dismissed") == "1" or (
            reaction.get("message_opened") == "0" and reaction.get("message_replied") == "0"
        )
        if overlap >= 0.5 and ignored and (best is None or overlap > best[1]):
            best = (h, overlap)

    if best is None:
        return None
    h, overlap = best
    reaction = h.get("reaction") or {}
    muted_or_reported = reaction.get("muted_after_message") == "1" or reaction.get("message_reported") == "1"

    try:
        forwarded = int(context["message"].get("forwarded_count") or 0)
    except ValueError:
        forwarded = 0
    text = _text_blob(context)
    if forwarded > 0:
        message_type = "forward"
    elif _has_signal(text, _PROMO_KEYWORDS):
        message_type = "spam"
    else:
        message_type = _infer_message_type(text, context["message"]["conversation_type"])

    return {
        "resolved": True,
        "rule": 6,
        "action": "mute" if muted_or_reported else "digest",
        "message_type": message_type,
        "confidence": 0.85 if muted_or_reported else 0.7,
        "reason": (
            f"Near-duplicate of message_history.csv row {h['message_id']} ({overlap:.0%} token overlap), "
            f"which this user ignored (notification_dismissed={reaction.get('notification_dismissed')}, "
            f"muted_after_message={reaction.get('muted_after_message')}, "
            f"message_reported={reaction.get('message_reported')})"
        ),
        "evidence_message_ids": h["message_id"],
    }


# --- rule 7: cold start, with cross-user safety exception --------------------

def rule_7_cold_start(context: dict) -> dict | None:
    history = context.get("history_candidates") or []
    if history:
        return None  # not cold start

    cross = context.get("cross_user_safety_evidence") or []
    if cross:
        best = cross[0]  # most recent
        reaction = best.get("reaction") or {}
        reported = reaction.get("message_reported") == "1"
        message_type = "scam" if reported else "spam"
        reason = (
            f"No message_history for this user with this sender, but a different user's message "
            f"{best['message_id']} from the same sender was "
            f"{'reported (message_reported=1)' if reported else 'muted after receipt (muted_after_message=1)'}; "
            f"treating as safety-flagged rather than genuine cold start (§3 cross-user safety exception)"
        )
        return {
            "resolved": True,
            "rule": "7-cross-user-exception",
            "action": "mute",
            "message_type": message_type,
            "confidence": 0.85,
            "reason": reason,
            "evidence_message_ids": best["message_id"],
        }

    text = _text_blob(context)
    if _has_signal(text, _URGENCY_KEYWORDS):
        return None  # unambiguous urgent ask present — let the LLM weigh it, don't cold-start digest

    return {
        "resolved": True,
        "rule": 7,
        "action": "digest",
        "message_type": _infer_message_type(text, context["message"]["conversation_type"]),
        "confidence": 0.45,
        "reason": (
            "No message_history matches for this sender/group/business (cold start) and no "
            "cross_user_safety_evidence; defaulting to digest per rule 7's cheap-failure-mode bias "
            "under genuine uncertainty"
        ),
        "evidence_message_ids": "none",
    }


_RULE_SEQUENCE = [
    rule_1_scam_domain,
    rule_2_opted_out_promo,
    rule_3_report_rate_outlier,
    rule_4_5_muted_group,
    rule_6_repetition,
    rule_7_cold_start,
]


def apply_rules(context: dict) -> dict:
    """Runs rules 1-7 in order. Returns the first resolved/escalation result,
    with any pending notes from earlier rules attached. If nothing resolves,
    returns {"resolved": False, "rule": None, "notes": [...]} — the message
    goes to router.py.

    Any rule that resolves to mute gets its evidence_message_ids backfilled
    (merged, deduped) with every history_candidates/cross_user_safety_evidence
    entry that independently corroborates a mute — regardless of which specific
    column tripped that rule's own condition. See _mute_supporting_ids()."""
    notes: list[dict] = []
    for rule_fn in _RULE_SEQUENCE:
        result = rule_fn(context)
        if result is None:
            continue
        if result.get("pending"):
            notes.append(result)
            continue
        if result.get("resolved") and result.get("action") == "mute":
            existing = result.get("evidence_message_ids") or "none"
            existing_ids = [] if existing == "none" else existing.split(";")
            merged = list(dict.fromkeys(existing_ids + _mute_supporting_ids(context)))
            result["evidence_message_ids"] = ";".join(merged) if merged else "none"
        result["notes"] = notes
        return result
    return {"resolved": False, "rule": None, "notes": notes}

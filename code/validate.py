"""
validate.py — output sanity checks (ARCHITECTURE.md §7).

Runs after rules.py/router.py produce a decision for every message, before
output.csv is written. Two tiers:
- hard checks: enum membership, confidence range, evidence-ID cross-check
  against message_history.csv AND against this specific message's own
  retrieved candidates (no cross-message leakage). Violations raise
  ValidationError — these should never happen if rules.py/router.py are
  correct, so they're bugs to fail loudly on, not soft warnings.
- soft flags: suspicious combos worth a human looking at (scam+notify,
  low-confidence mute with no backing) — flagged, never blocked, since §7
  says "flag (don't necessarily block)".
"""

from __future__ import annotations

import csv
from pathlib import Path

VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_MESSAGE_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"


class ValidationError(Exception):
    """A hard violation of the output contract."""


def load_message_history_ids(dataset_dir: Path = DATASET_DIR) -> set[str]:
    with open(dataset_dir / "message_history.csv", newline="", encoding="utf-8") as f:
        return {row["message_id"] for row in csv.DictReader(f)}


def _retrieved_ids(context: dict) -> set[str]:
    ids = {h["message_id"] for h in context.get("history_candidates") or []}
    ids |= {c["message_id"] for c in context.get("cross_user_safety_evidence") or []}
    return ids


def validate_row(row: dict, context: dict, message_history_ids: set[str]) -> list[str]:
    """row: {"message_id","action","message_type","reason","confidence",
    "evidence_message_ids"}. context: the MessageContext this row was decided
    from — required for the no-leakage check (an ID can't be cited unless it
    was actually retrieved as a candidate for *this* message).

    Raises ValidationError on hard violations. Returns a list of soft-flag
    strings for suspicious combos (empty if none)."""
    message_id = row.get("message_id")
    action = row.get("action")
    message_type = row.get("message_type")
    confidence = row.get("confidence")
    evidence_raw = row.get("evidence_message_ids")

    if action not in VALID_ACTIONS:
        raise ValidationError(f"{message_id}: action {action!r} not in {sorted(VALID_ACTIONS)}")
    if message_type not in VALID_MESSAGE_TYPES:
        raise ValidationError(f"{message_id}: message_type {message_type!r} not in {sorted(VALID_MESSAGE_TYPES)}")

    try:
        confidence_f = float(confidence)
    except (TypeError, ValueError):
        raise ValidationError(f"{message_id}: confidence {confidence!r} is not a number")
    if not (0.0 <= confidence_f <= 1.0):
        raise ValidationError(f"{message_id}: confidence {confidence_f} not in [0,1]")

    retrieved_ids = _retrieved_ids(context)
    evidence_raw = "none" if evidence_raw is None else str(evidence_raw)
    cited_ids = [] if evidence_raw.strip().lower() == "none" else [
        e.strip() for e in evidence_raw.split(";") if e.strip()
    ]
    for cid in cited_ids:
        if cid not in message_history_ids:
            raise ValidationError(
                f"{message_id}: cited evidence_message_ids {cid!r} does not exist in message_history.csv"
            )
        if cid not in retrieved_ids:
            raise ValidationError(
                f"{message_id}: cited evidence_message_ids {cid!r} exists in message_history.csv but "
                f"was never retrieved as a candidate for this message (cross-message leakage) — not in "
                f"this message's own history_candidates or cross_user_safety_evidence"
            )

    flags: list[str] = []
    if message_type == "scam" and action == "notify":
        flags.append(f"{message_id}: suspicious combo — message_type=scam with action=notify")
    if action == "mute" and confidence_f < 0.5:
        has_backing = bool(retrieved_ids) or bool(cited_ids)
        if not has_backing:
            flags.append(
                f"{message_id}: suspicious combo — action=mute with confidence={confidence_f} < 0.5 "
                f"and no rule/history backing (empty history_candidates and cross_user_safety_evidence)"
            )

    return flags


def validate_output(
    rows: list[dict], contexts_by_id: dict[str, dict], message_history_ids: set[str] | None = None
) -> list[str]:
    """Validates every row against its own MessageContext. Raises
    ValidationError on the first hard violation (fail fast). Returns the full
    list of soft flags across all rows (may be empty)."""
    if message_history_ids is None:
        message_history_ids = load_message_history_ids()

    all_flags: list[str] = []
    for row in rows:
        context = contexts_by_id.get(row["message_id"])
        if context is None:
            raise ValidationError(f"{row['message_id']}: no MessageContext found for this row")
        all_flags.extend(validate_row(row, context, message_history_ids))
    return all_flags

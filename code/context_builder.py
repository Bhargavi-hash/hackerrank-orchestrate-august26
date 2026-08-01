"""
context_builder.py — deterministic context assembly (ARCHITECTURE.md §2-3).

Joins every dataset/*.csv into one MessageContext dict per row of messages.csv.
No LLM/VLM/ASR calls here — media OCR/ASR extraction belongs to
media/image_processor.py and media/voice_processor.py (§4); this module only
resolves the media file path and leaves extraction fields for that pipeline
to fill in.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
HISTORY_CAP = 5

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "to", "of", "in", "on", "at", "for", "with", "this", "that",
    "it", "you", "your", "i", "we", "us", "our", "as", "by", "from", "will",
    "would", "can", "could", "please", "hi", "hello", "hey", "thanks", "thank",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _index(rows: list[dict], *keys: str) -> dict[Any, dict]:
    if len(keys) == 1:
        (key,) = keys
        return {r[key]: r for r in rows}
    return {tuple(r[k] for k in keys): r for r in rows}


class Dataset:
    """Loads and indexes every dataset/*.csv once; reused across all messages."""

    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.messages = _load_csv(dataset_dir / "messages.csv")
        self.users = _index(_load_csv(dataset_dir / "users.csv"), "user_id")
        self.groups = _index(_load_csv(dataset_dir / "groups.csv"), "group_id")
        self.group_members = _index(
            _load_csv(dataset_dir / "group_members.csv"), "group_id", "user_id"
        )
        self.businesses = _index(
            _load_csv(dataset_dir / "business_accounts.csv"), "business_id"
        )
        self.user_business_history = _index(
            _load_csv(dataset_dir / "user_business_history.csv"), "user_id", "business_id"
        )
        self.message_history = _load_csv(dataset_dir / "message_history.csv")
        self.message_events = _index(
            _load_csv(dataset_dir / "message_events.csv"), "user_id", "message_id"
        )
        self.images = _index(_load_csv(dataset_dir / "images.csv"), "image_id")
        self.voice_notes = _index(_load_csv(dataset_dir / "voice_notes.csv"), "voice_note_id")

        self._daily_summary_by_user: dict[str, list[dict]] = {}
        for row in _load_csv(dataset_dir / "daily_notification_summary.csv"):
            self._daily_summary_by_user.setdefault(row["user_id"], []).append(row)

    def latest_daily_load(self, user_id: str) -> dict | None:
        """Most recent daily_notification_summary row for user_id.

        daily_notification_summary dates (2026-07-04..07-17) never overlap
        messages.csv created_at dates (2026-07-18..07-31) for any user, so
        "today's row" can't be a same-day lookup — it's a fixed pre-window
        load snapshot. The most recent available row is the closest usable
        proxy for "current daily load" at decision time.
        """
        rows = self._daily_summary_by_user.get(user_id)
        if not rows:
            return None
        return max(rows, key=lambda r: r["date"])


def _sender_context(message: dict, ds: Dataset) -> dict:
    conv_type = message["conversation_type"]
    ctx: dict[str, Any] = {}
    if conv_type == "group":
        ctx["group"] = ds.groups.get(message["group_id"])
        ctx["membership"] = ds.group_members.get((message["group_id"], message["user_id"]))
    elif conv_type == "business":
        ctx["business"] = ds.businesses.get(message["business_id"])
        ctx["relationship"] = ds.user_business_history.get(
            (message["user_id"], message["business_id"])
        )
    # personal: no per-sender data exists beyond users.csv global stats (§3)
    return ctx


def _match_key(message: dict) -> tuple[str, str] | None:
    conv_type = message["conversation_type"]
    if conv_type == "group" and message["group_id"]:
        return "group_id", message["group_id"]
    if conv_type == "business" and message["business_id"]:
        return "business_id", message["business_id"]
    if conv_type == "personal" and message["sender_user_id"]:
        return "sender_user_id", message["sender_user_id"]
    return None


def _history_candidates(message: dict, ds: Dataset) -> list[dict]:
    key = _match_key(message)
    if key is None:
        return []
    field, value = key
    matches = [
        h
        for h in ds.message_history
        if h["user_id"] == message["user_id"]
        and h.get(field) == value
        and h["message_id"] != message["message_id"]
    ]
    if not matches:
        return []

    matches.sort(key=lambda h: h["created_at"], reverse=True)
    if len(matches) > HISTORY_CAP:
        query_tokens = _tokenize(message["message_text"])
        matches.sort(
            key=lambda h: (len(query_tokens & _tokenize(h["message_text"])), h["created_at"]),
            reverse=True,
        )
    top = matches[:HISTORY_CAP]

    return [
        {
            "message_id": h["message_id"],
            "created_at": h["created_at"],
            "message_text": h["message_text"],
            "reaction": ds.message_events.get((message["user_id"], h["message_id"])),
        }
        for h in top
    ]


def _cross_user_safety_evidence(message: dict, ds: Dataset) -> list[dict]:
    """Second retrieval tier (§3): another user's reported/muted history against
    the same business_id, or same sender_user_id for personal repeat offenders.
    Only used for safety corroboration (rules 1/3/7) — never for personalization
    inference about this user, since that must stay user-scoped."""
    conv_type = message["conversation_type"]
    if conv_type == "business" and message["business_id"]:
        field, value = "business_id", message["business_id"]
    elif conv_type == "personal" and message["sender_user_id"]:
        field, value = "sender_user_id", message["sender_user_id"]
    else:
        return []

    evidence = []
    for h in ds.message_history:
        if h["user_id"] == message["user_id"]:
            continue
        if h.get(field) != value:
            continue
        event = ds.message_events.get((h["user_id"], h["message_id"]))
        if not event:
            continue
        if event.get("message_reported") == "1" or event.get("muted_after_message") == "1":
            evidence.append(
                {
                    "message_id": h["message_id"],
                    "user_id": h["user_id"],
                    "created_at": h["created_at"],
                    "message_text": h["message_text"],
                    "reaction": event,
                }
            )
    evidence.sort(key=lambda e: e["created_at"], reverse=True)
    return evidence


def _media(message: dict, ds: Dataset) -> dict:
    media_type = message["media_type"] or None
    file_path = None
    if media_type == "image":
        row = ds.images.get(message["media_id"])
        file_path = row["file_path"] if row else None
    elif media_type == "voice":
        row = ds.voice_notes.get(message["media_id"])
        file_path = row["file_path"] if row else None
    return {
        "type": media_type,
        "file_path": file_path,
        # Filled by media/image_processor.py or media/voice_processor.py (§4),
        # cached by media_id. context_builder does no OCR/ASR itself.
        "extracted_text": None,
        "structured": None,
    }


def build_context(message: dict, ds: Dataset) -> dict:
    return {
        "message": message,
        "sender_context": _sender_context(message, ds),
        "user": ds.users.get(message["user_id"]),
        "daily_load": ds.latest_daily_load(message["user_id"]),
        "media": _media(message, ds),
        "history_candidates": _history_candidates(message, ds),
        "cross_user_safety_evidence": _cross_user_safety_evidence(message, ds),
    }


def build_all_contexts(ds: Dataset | None = None) -> list[dict]:
    ds = ds or Dataset()
    return [build_context(m, ds) for m in ds.messages]


if __name__ == "__main__":
    import json

    ds = Dataset()
    contexts = build_all_contexts(ds)
    print(json.dumps(contexts[0], indent=2))

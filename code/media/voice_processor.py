"""
media/voice_processor.py — ASR transcription + structured extraction (ARCHITECTURE.md §4).

One Whisper call per unique media_id (not per message_id), same reuse rationale
as image_processor.py. The transcript is treated as message_text-equivalent and
passed through the same lightweight urgency/deadline keyword extraction used
elsewhere for text (§4: "pass to the same urgency/deadline extraction as
text") rather than a second paid LLM call — Whisper gives us raw text, and
text-based structuring is cheap and deterministic.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ASR_MODEL, GROQ_API_KEY, LLM_PROVIDER  # noqa: E402

logger = logging.getLogger("media.voice_processor")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / ".cache" / "media"

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
_TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\s?(am|pm)?\b", re.I)
_DATE_RE = re.compile(
    r"\b\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b", re.I
)
_DEADLINE_RE = re.compile(
    r"\b(today|tonight|tomorrow|by \d{1,2}[:.]?\d{0,2}\s?(am|pm)?|within \d+ ?(minutes|hours))\b", re.I
)


def _structure_from_text(text: str) -> dict:
    lowered = text.lower()
    time_match = _TIME_RE.search(text)
    date_match = _DATE_RE.search(text)
    deadline_match = _DEADLINE_RE.search(text)
    return {
        "title": None,
        "date": date_match.group(0) if date_match else None,
        "time": time_match.group(0) if time_match else None,
        "location": None,
        "deadline": deadline_match.group(0) if deadline_match else None,
        "payment_request": any(k in lowered for k in _PAYMENT_KEYWORDS),
        "urgency_cues": [k.strip() for k in _URGENCY_KEYWORDS if k in lowered],
    }


def _cache_path(media_id: str) -> Path:
    return CACHE_DIR / f"{media_id}.json"


def _client():
    if LLM_PROVIDER != "groq":
        raise NotImplementedError(f"LLM_PROVIDER={LLM_PROVIDER!r} not implemented; only 'groq' is wired up.")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set (checked .env / environment).")
    from groq import Groq

    return Groq(api_key=GROQ_API_KEY)


def _retryable(exc: Exception) -> bool:
    from groq import APIStatusError

    return isinstance(exc, APIStatusError) and exc.status_code in (429, 503)


def _retry_delay(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return min(2**attempt, 30)


def _call_with_retry(fn, max_attempts: int = 6):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            if not _retryable(exc) or attempt == max_attempts - 1:
                raise
            wait = _retry_delay(exc, attempt)
            logger.warning(
                "Groq ASR call failed (%s), retrying in %.1fs [attempt %d/%d]",
                exc, wait, attempt + 1, max_attempts,
            )
            time.sleep(wait)


def process_voice(media_id: str, file_path: str) -> dict:
    """Returns {"extracted_text": str, "structured": {...}}. Cached by media_id
    at .cache/media/{media_id}.json — a repeat call for the same media_id never
    calls the API."""
    cache_file = _cache_path(media_id)
    if cache_file.exists():
        logger.info("cache HIT media_id=%s -> %s", media_id, cache_file)
        return json.loads(cache_file.read_text())["result"]

    logger.info("cache MISS media_id=%s -> calling Groq ASR model=%s", media_id, ASR_MODEL)
    t0 = time.time()

    abs_path = DATASET_DIR / file_path
    client = _client()
    with open(abs_path, "rb") as f:
        audio_bytes = f.read()
    resp = _call_with_retry(
        lambda: client.audio.transcriptions.create(
            model=ASR_MODEL,
            file=(abs_path.name, audio_bytes),
            temperature=0,
            response_format="json",
        )
    )
    transcript = (resp.text or "").strip()

    logger.info("Groq ASR call complete media_id=%s elapsed=%.2fs", media_id, time.time() - t0)

    result = {
        "extracted_text": transcript,
        "structured": _structure_from_text(transcript),
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"media_id": media_id, "model": ASR_MODEL, "result": result}, indent=2)
    )
    return result

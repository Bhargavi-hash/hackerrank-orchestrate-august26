"""
media/image_processor.py — VLM OCR + structured extraction (ARCHITECTURE.md §4).

One VLM call per unique media_id (not per message_id): posters/screenshots
recur across messages (e.g. img_008 is sent to two different users with two
different captions), so caching by media_id avoids reprocessing/re-billing.

Two-tier provider fallback: Groq's qwen/qwen3.6-27b is the only vision-capable
model Groq currently offers (confirmed against console.groq.com/docs/vision —
no second Groq vision model exists), so once its own retries are exhausted
there is nothing left to retry *within* Groq. On exhaustion, this module falls
back to a second provider (Gemini) entirely rather than retrying the same
single point of failure. Which model actually produced each result is logged
and stored in the cache file (not left silent) since it matters for the
audit trail.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    LLM_PROVIDER,
    VISION_FALLBACK_MODEL,
    VISION_FALLBACK_PROVIDER,
    VISION_MODEL,
)

logger = logging.getLogger("media.image_processor")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / ".cache" / "media"

_SYSTEM_PROMPT = """You are an OCR and image-structure extraction tool for a WhatsApp \
message-routing system. Given an image, respond with ONLY this JSON object — no prose, \
no markdown fences:
{
  "extracted_text": "<all visible text, verbatim, empty string if none>",
  "is_poster_or_flyer": <bool>,
  "is_receipt": <bool>,
  "is_screenshot": <bool>,
  "title": "<short heading/title if visible, else null>",
  "date": "<visible date string, else null>",
  "time": "<visible time string, else null>",
  "location": "<visible location/address, else null>",
  "deadline": "<visible deadline phrase, else null>",
  "price": "<visible price/amount, else null>",
  "has_qr_code": <bool>,
  "has_payment_link": <bool>,
  "payment_request": <bool, true if the image is asking the viewer to pay/scan/transfer money>,
  "urgency_cues": ["<short urgency phrases visible, e.g. \\"today only\\", \\"limited time\\">"]
}"""


def _cache_path(media_id: str) -> Path:
    return CACHE_DIR / f"{media_id}.json"


def _normalize(data: dict) -> dict:
    return {
        "extracted_text": data.get("extracted_text") or "",
        "structured": {
            "title": data.get("title"),
            "date": data.get("date"),
            "time": data.get("time"),
            "location": data.get("location"),
            "deadline": data.get("deadline"),
            "payment_request": bool(data.get("payment_request")),
            "urgency_cues": data.get("urgency_cues") or [],
            "is_poster_or_flyer": bool(data.get("is_poster_or_flyer")),
            "is_receipt": bool(data.get("is_receipt")),
            "is_screenshot": bool(data.get("is_screenshot")),
            "has_qr_code": bool(data.get("has_qr_code")),
            "has_payment_link": bool(data.get("has_payment_link")),
            "price": data.get("price"),
        },
    }


# --- primary: Groq -----------------------------------------------------------

def _groq_client():
    if LLM_PROVIDER != "groq":
        raise NotImplementedError(f"LLM_PROVIDER={LLM_PROVIDER!r} not implemented; only 'groq' is wired up.")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set (checked .env / environment).")
    from groq import Groq

    return Groq(api_key=GROQ_API_KEY)


def _retryable(exc: Exception) -> bool:
    """429 on the per-minute TPM cap (tight — 8000 TPM, easily hit processing
    several images back to back) and 503 are transient — worth a short
    backoff. A 429 on the *per-day* TPD cap is not: observed directly
    (200000/day limit, suggested wait ~9.5 minutes) — retrying within this
    run would just hit the same wall again, so it's treated as non-retryable
    and falls straight to the Gemini fallback in process_image() instead of
    burning minutes waiting on a quota that won't reset soon. A 400 with code
    json_validate_failed and an empty failed_generation has also been observed
    under rate-limit pressure — treated as transient too."""
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
    if exc.status_code == 400:
        body = exc.body if isinstance(exc.body, dict) else {}
        return (body.get("error") or {}).get("code") == "json_validate_failed"
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


def _call_with_retry(fn, max_attempts: int = 6):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            if not _retryable(exc) or attempt == max_attempts - 1:
                raise
            wait = _retry_delay(exc, attempt)
            logger.warning(
                "Groq vision call failed (%s), retrying in %.1fs [attempt %d/%d]",
                exc, wait, attempt + 1, max_attempts,
            )
            time.sleep(wait)


def _extract_with_groq(b64: str, ext: str) -> dict:
    client = _groq_client()
    resp = _call_with_retry(
        lambda: client.chat.completions.create(
            model=VISION_MODEL,
            temperature=0,
            # qwen3.6-27b is a reasoning model; on some images (dense posters/
            # notices) its hidden reasoning consumes the entire token budget
            # before emitting an answer, failing json_object validation
            # server-side with "max completion tokens reached before
            # generating a valid document" (reproduced on img_012 even at
            # max_tokens=2000). reasoning_effort="none" disables thinking
            # tokens entirely — deterministic, cheaper, and fixes it outright.
            reasoning_effort="none",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract structured data from this image per the schema."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
                    ],
                },
            ],
            max_tokens=800,
        )
    )
    raw = resp.choices[0].message.content
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # strict=False: extracted_text can legitimately contain raw control chars
    # (literal newlines in poster copy) that some models emit unescaped inside
    # the JSON string value — reproduced on img_011 via the Gemini fallback.
    return json.loads(cleaned, strict=False)


# --- fallback: Gemini ---------------------------------------------------------

def _gemini_retryable(exc: Exception) -> bool:
    """429 on the per-*minute* RPM quota is transient — worth a short
    backoff. 429 on the per-*day* RPD quota is not (reproduced directly:
    gemini-2.5-flash-lite's free tier caps GenerateRequestsPerDayPerProjectPerModel
    at 20/day too — switching flash variants only added a separate
    per-minute throttle on the same-size daily cap, not a materially higher
    daily allowance). Distinguish by the quotaId in the structured
    QuotaFailure detail rather than guessing from the RetryInfo delay, which
    can be short even for a daily-cap error."""
    from google.genai.errors import ClientError

    if not (isinstance(exc, ClientError) and exc.code == 429):
        return False
    try:
        details = (exc.details or {}).get("error", {}).get("details", [])
        for d in details:
            if str(d.get("@type", "")).endswith("QuotaFailure"):
                for v in d.get("violations", []):
                    if "PerDay" in str(v.get("quotaId", "")):
                        return False
    except Exception:
        pass
    return True


def _gemini_retry_delay(exc: Exception, attempt: int) -> float:
    """gemini-2.5-flash-lite's free tier is a *per-minute* RPM cap (10/min,
    confirmed from a live 429: quotaId GenerateRequestsPerMinutePerProjectPerModel-FreeTier),
    not a daily one — worth retrying, unlike a per-day quota. Parse the
    server's own RetryInfo delay when present rather than guessing."""
    try:
        details = (exc.details or {}).get("error", {}).get("details", [])
        for d in details:
            if str(d.get("@type", "")).endswith("RetryInfo"):
                delay_str = str(d.get("retryDelay", ""))
                if delay_str.endswith("s"):
                    return float(delay_str[:-1]) + 1.0
    except Exception:
        pass
    return min(5 * (attempt + 1), 30)


def _call_with_gemini_retry(fn, max_attempts: int = 4):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            if not _gemini_retryable(exc) or attempt == max_attempts - 1:
                raise
            wait = _gemini_retry_delay(exc, attempt)
            logger.warning(
                "Gemini vision call failed (%s), retrying in %.1fs [attempt %d/%d]",
                exc, wait, attempt + 1, max_attempts,
            )
            time.sleep(wait)


def _extract_with_gemini(image_bytes: bytes, mime_type: str) -> dict:
    if VISION_FALLBACK_PROVIDER != "google":
        raise NotImplementedError(f"VISION_FALLBACK_PROVIDER={VISION_FALLBACK_PROVIDER!r} not implemented.")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set (checked .env / environment).")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    resp = _call_with_gemini_retry(
        lambda: client.models.generate_content(
            model=VISION_FALLBACK_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                "Extract structured data from this image per the schema in the system instruction.",
            ],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
    )
    return json.loads(resp.text, strict=False)


def process_image(media_id: str, file_path: str) -> dict:
    """Returns {"extracted_text": str, "structured": {...}}. Cached by media_id
    at .cache/media/{media_id}.json — a repeat call for the same media_id never
    calls the API. Tries Groq first; on exhausted retries, falls back to
    Gemini. Raises only if both providers fail."""
    cache_file = _cache_path(media_id)
    if cache_file.exists():
        logger.info("cache HIT media_id=%s -> %s", media_id, cache_file)
        return json.loads(cache_file.read_text())["result"]

    abs_path = DATASET_DIR / file_path
    with open(abs_path, "rb") as f:
        raw_bytes = f.read()
    ext = abs_path.suffix.lstrip(".").lower() or "jpeg"
    mime_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
    b64 = base64.b64encode(raw_bytes).decode()

    t0 = time.time()
    provider_used = None
    data = None

    logger.info("cache MISS media_id=%s -> calling Groq vision model=%s", media_id, VISION_MODEL)
    try:
        data = _extract_with_groq(b64, ext)
        provider_used = f"groq:{VISION_MODEL}"
    except Exception as groq_exc:
        logger.warning(
            "Groq vision exhausted retries for media_id=%s (%s) -> falling back to %s:%s",
            media_id, groq_exc, VISION_FALLBACK_PROVIDER, VISION_FALLBACK_MODEL,
        )
        try:
            data = _extract_with_gemini(raw_bytes, mime_type)
            provider_used = f"{VISION_FALLBACK_PROVIDER}:{VISION_FALLBACK_MODEL}"
        except Exception as fallback_exc:
            logger.error(
                "Both vision providers failed for media_id=%s: groq=%s, fallback=%s",
                media_id, groq_exc, fallback_exc,
            )
            raise

    result = _normalize(data)
    logger.info(
        "vision extraction complete media_id=%s provider=%s elapsed=%.2fs",
        media_id, provider_used, time.time() - t0,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"media_id": media_id, "provider": provider_used, "result": result}, indent=2)
    )
    return result

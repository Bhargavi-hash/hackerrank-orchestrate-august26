"""
config.py — single place for model/provider selection from env vars (ARCHITECTURE.md §6a).

No provider is hardcoded into router.py/media/*.py themselves; they read these
constants. Swapping providers is a config change here (or via env vars), not a
rewrite of the call sites.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b")
ASR_MODEL = os.environ.get("ASR_MODEL", "whisper-large-v3")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Vision fallback: Groq currently offers exactly one vision-capable model
# (qwen/qwen3.6-27b, confirmed against console.groq.com/docs/vision — no
# second Groq vision model exists to retry against). When it's exhausted its
# own retries (rate limits / sustained 503 capacity errors), image_processor.py
# falls back to a second provider entirely rather than retrying the same
# single point of failure.
#
# GEMINI_API_KEY, not GOOGLE_API_KEY: a separate Google account/project,
# confirmed genuinely distinct (not just a renamed duplicate) because it 404s
# on gemini-2.5-flash/-flash-lite with "no longer available to new users" —
# a different model lineup than the GOOGLE_API_KEY account, not a shared
# quota pool wearing a different name. gemini-flash-lite-latest is the model
# actually available and unexhausted on this account (verified live: several
# other candidates 404'd or were already 429 on this same key). Falls back to
# GOOGLE_API_KEY/gemini-2.5-flash-lite if GEMINI_API_KEY isn't set, so this
# still works with only the original credential. Quality checked directly
# (a real OCR call on img_003, a real routing-decision call) before adoption.
VISION_FALLBACK_PROVIDER = os.environ.get("VISION_FALLBACK_PROVIDER", "google")
VISION_FALLBACK_MODEL = os.environ.get("VISION_FALLBACK_MODEL", "gemini-flash-lite-latest")
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Same rationale as the vision fallback above, for the text router: Groq's
# free-tier daily token quota (100000 TPD on llama-3.3-70b-versatile) is
# reachable within a single full dataset run once other testing in the same
# day has used part of it (reproduced directly: 429 on tokens per day mid-run
# processing dataset/messages.csv). router.py falls back to a second provider
# on exhausted Groq retries rather than blocking on a quota that won't clear
# for hours. Same GEMINI_API_KEY/gemini-flash-lite-latest swap as the vision
# fallback, same reasoning — the *primary* router model (LLM_MODEL, Groq) is
# intentionally left alone; only this fallback tier changed.
LLM_FALLBACK_PROVIDER = os.environ.get("LLM_FALLBACK_PROVIDER", "google")
LLM_FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", "gemini-flash-lite-latest")

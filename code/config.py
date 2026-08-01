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
VISION_FALLBACK_PROVIDER = os.environ.get("VISION_FALLBACK_PROVIDER", "google")
VISION_FALLBACK_MODEL = os.environ.get("VISION_FALLBACK_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

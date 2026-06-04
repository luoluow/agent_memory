"""Configuration for OmniService.

All values can be overridden via environment variables (OMNI_*). Defaults are
chosen to sit alongside Ollama (:11434) without collisions.
"""

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Storage root — service-owned, NOT under ~/.claude. One subdir per namespace.
STORAGE_ROOT = Path(_env("OMNI_STORAGE_ROOT", str(Path.home() / ".omni"))).expanduser()

# HTTP service
HOST = _env("OMNI_HOST", "127.0.0.1")
PORT = int(_env("OMNI_PORT", "11435"))

# Client identity (required on every request; partitions storage per client).
# CLI/MCP clients default to this; test runs use a distinct id (e.g. "test").
CLIENT_ID = _env("OMNI_CLIENT_ID", "claude-code")

# Local LLM (Ollama, OpenAI-compatible endpoint)
OLLAMA_BASE_URL = _env("OMNI_OLLAMA_BASE_URL", "http://localhost:11434/v1")
# Gemma 4 E4B (Apache 2.0, native structured-JSON output, 128K ctx, ~6 GB resident)
# serves both roles on a 16 GB GPU with headroom and no model swapping.
EXTRACT_MODEL = _env("OMNI_EXTRACT_MODEL", "gemma4:e4b")   # EXTRACT / RELATE / COMPRESS
VERIFY_MODEL = _env("OMNI_VERIFY_MODEL", "gemma4:e4b")     # VERIFY (stronger reasoning)

# Background pipeline tuning
EXTRACT_WORKERS = int(_env("OMNI_EXTRACT_WORKERS", "4"))
VERIFY_DEBOUNCE_SECONDS = float(_env("OMNI_VERIFY_DEBOUNCE_SECONDS", "20"))
VERIFY_RAW_WINDOW = int(_env("OMNI_VERIFY_RAW_WINDOW", "8"))  # recent raw files VERIFY re-reads

# Retrieval shaping
RETRIEVE_TOP_PAGES = int(_env("OMNI_RETRIEVE_TOP_PAGES", "8"))
RETRIEVE_TOP_HISTORY = int(_env("OMNI_RETRIEVE_TOP_HISTORY", "30"))

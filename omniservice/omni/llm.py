"""Local LLM client (Ollama, OpenAI-compatible) and JSON parsing helpers."""

import re
import threading
from typing import Optional

from openai import OpenAI

from omni import config

_client: Optional[OpenAI] = None
_client_lock = threading.Lock()


def get_client() -> OpenAI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
    return _client


def call_local(prompt: str, system: str, model: str,
               max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """Single synchronous chat completion against the local model."""
    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def extract_json(text: str) -> str:
    """Pull a JSON object out of text that may contain prose or code fences."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text

"""Local-LLM cleanup pass via Ollama.

This is the step that makes the output feel like Wispr Flow rather than a raw
Whisper dump: a small local model fixes punctuation/capitalization, removes
filler words and false starts, applies spoken self-corrections, and formats
lists — without changing the user's meaning or adding new content.

If Ollama is unreachable or the text is too short, the raw transcript is
returned unchanged, so dictation keeps working even without a cleanup model.
"""

from __future__ import annotations

from typing import Any, Dict

import requests

SYSTEM_PROMPT = (
    "You are a dictation post-processor. You receive a raw speech-to-text "
    "transcript and return a cleaned-up version of it.\n"
    "Rules:\n"
    "- Fix capitalization, spelling, and punctuation.\n"
    "- Remove filler words (um, uh, like, you know) and false starts.\n"
    "- Apply spoken self-corrections. Example: 'send it Monday, no wait, "
    "Tuesday' becomes 'send it Tuesday'.\n"
    "- Turn clearly enumerated speech into a formatted list.\n"
    "- Preserve the speaker's wording, tone, and meaning. Do NOT add new "
    "information, do NOT summarize, do NOT answer questions, do NOT explain.\n"
    "- Output ONLY the cleaned text. No preamble, no quotes, no commentary."
)


class Cleaner:
    """Sends transcripts to a local Ollama model for cleanup."""

    def __init__(self, config: Dict[str, Any]):
        self.enabled = bool(config.get("enabled", True))
        self.host = config.get("host", "http://localhost:11434").rstrip("/")
        self.model = config.get("model", "gemma3:4b")
        self.min_chars = int(config.get("min_chars", 40))
        self.timeout = float(config.get("timeout", 20))
        self.temperature = float(config.get("temperature", 0.2))
        # How long Ollama keeps the model loaded after a request. Keeping it
        # resident between utterances is what makes cleanup feel instant.
        self.keep_alive = config.get("keep_alive", "10m")

    def is_available(self) -> bool:
        """Return True if the Ollama server responds."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list:
        """Names of the models currently pulled in Ollama ([] if unreachable)."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m.get("name") for m in models if m.get("name")]
        except (requests.RequestException, ValueError):
            return []

    def clean(self, text: str) -> str:
        """Return a cleaned-up version of ``text``, or the original on any failure."""
        if not self.enabled or not text:
            return text
        if len(text) < self.min_chars:
            return text

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": text,
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            cleaned = (resp.json().get("response") or "").strip()
        except requests.RequestException as exc:
            print(f"[cleanup] Ollama unavailable ({exc}); using raw transcript.", flush=True)
            return text
        except (ValueError, KeyError) as exc:
            print(f"[cleanup] bad response ({exc}); using raw transcript.", flush=True)
            return text

        # Guard against a model that returns nothing or wraps output in quotes.
        if not cleaned:
            return text
        return cleaned.strip().strip('"').strip()

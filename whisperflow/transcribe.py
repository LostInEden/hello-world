"""Speech-to-text via faster-whisper (CTranslate2) with GPU acceleration.

faster-whisper bundles Silero VAD, so we get voice-activity filtering for free
by passing ``vad_filter=True`` — no separate VAD dependency required.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class Transcriber:
    """Wraps a faster-whisper model and transcribes float32 audio buffers."""

    def __init__(self, config: Dict[str, Any]):
        # Import lazily so `--help`, `devices`, etc. don't pay the heavy import cost.
        from faster_whisper import WhisperModel

        self.model_name = config["model"]
        self.language = config.get("language")
        self.beam_size = int(config.get("beam_size", 5))
        self.vad_filter = bool(config.get("vad_filter", True))

        device = config.get("device", "cuda")
        compute_type = config.get("compute_type", "float16")

        print(f"[stt] loading {self.model_name} on {device} ({compute_type})...", flush=True)
        try:
            self.model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
        except Exception as exc:  # pragma: no cover - environment dependent
            if device == "cuda":
                print(
                    f"[stt] failed to load on CUDA ({exc}); falling back to CPU int8. "
                    "Run `python -m whisperflow doctor` to diagnose GPU setup.",
                    flush=True,
                )
                self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            else:
                raise
        print("[stt] model ready.", flush=True)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 1-D float32 16 kHz audio array into text."""
        if audio is None or len(audio) == 0:
            return ""

        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        # `segments` is a generator; iterating it runs the actual decode.
        text = "".join(segment.text for segment in segments)
        return text.strip()

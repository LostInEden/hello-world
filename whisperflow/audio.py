"""Microphone capture via sounddevice.

Records mono float32 audio at 16 kHz into an in-memory buffer while a recording
is active, then returns it as a single NumPy array ready for faster-whisper.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
import sounddevice as sd


class Recorder:
    """Start/stop microphone recording and return the captured audio."""

    def __init__(self, sample_rate: int = 16000, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device = device
        self._frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001 - sounddevice API
        if status:
            # Overflows/underruns are non-fatal; surface them for debugging.
            print(f"[audio] {status}", flush=True)
        with self._lock:
            self._frames.append(indata.copy())

    def start(self) -> None:
        """Begin capturing. Safe to call once per utterance."""
        with self._lock:
            self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop capturing and return the recorded audio as a 1-D float32 array."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

        with self._lock:
            frames = self._frames
            self._frames = []

        if not frames:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(frames, axis=0).flatten().astype(np.float32)

    def duration(self, audio: np.ndarray) -> float:
        """Length of an audio array in seconds."""
        if audio is None or len(audio) == 0:
            return 0.0
        return len(audio) / float(self.sample_rate)


def list_devices() -> str:
    """Return a human-readable list of available audio devices."""
    return str(sd.query_devices())

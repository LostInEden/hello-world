"""Speech-to-text via faster-whisper (CTranslate2) with GPU acceleration.

faster-whisper bundles Silero VAD, so we get voice-activity filtering for free
by passing ``vad_filter=True`` — no separate VAD dependency required.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np


def _ensure_windows_cuda_dlls() -> None:
    """Make the pip-installed NVIDIA CUDA DLLs findable on Windows.

    ``pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`` drops the DLLs into
    ``site-packages/nvidia/<pkg>/bin``, which is not on the default DLL search
    path — CTranslate2 then loads the model fine but fails at inference time
    with e.g. "Library cublas64_12.dll is not found". Register those
    directories before faster-whisper runs.
    """
    if sys.platform != "win32":
        return
    import site

    roots = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass

    add_dll_directory = getattr(os, "add_dll_directory", None)
    for root in roots:
        nvidia = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia):
            continue
        for pkg in os.listdir(nvidia):
            bin_dir = os.path.join(nvidia, pkg, "bin")
            if not os.path.isdir(bin_dir):
                continue
            # PATH covers CTranslate2's runtime LoadLibrary calls;
            # add_dll_directory covers extension-module dependencies.
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            if add_dll_directory is not None:
                try:
                    add_dll_directory(bin_dir)
                except OSError:
                    pass


_ensure_windows_cuda_dlls()


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
            self.device_used = device
            self.compute_type_used = compute_type
        except Exception as exc:  # pragma: no cover - environment dependent
            if device != "cuda":
                raise
            print(
                f"[stt] failed to load on CUDA ({exc}); falling back to CPU int8. "
                "Run `python -m whisperflow doctor` to diagnose GPU setup.",
                flush=True,
            )
            try:
                self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            except Exception as cpu_exc:
                raise RuntimeError(
                    f"Could not load Whisper model {self.model_name!r} on CUDA or on CPU. "
                    f"CUDA error: {exc}. CPU error: {cpu_exc}. "
                    "If both mention downloads or network, the model files could not be "
                    "fetched — check your connection or pre-download the model."
                ) from cpu_exc
            self.device_used = "cpu"
            self.compute_type_used = "int8"
        print(f"[stt] model ready on {self.device_used} ({self.compute_type_used}).", flush=True)

    def warmup(self) -> None:
        """Run a throwaway decode so the first real utterance isn't slow.

        Uses low-amplitude noise with VAD off: pure silence plus the VAD filter
        would skip the decode entirely and warm nothing up.
        """
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(16000) * 0.005).astype(np.float32)
        segments, _info = self.model.transcribe(
            noise, language=self.language, beam_size=1, vad_filter=False
        )
        for _segment in segments:
            pass

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

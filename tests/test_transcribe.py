"""Transcriber: CUDA fallback logic and segment joining, with a fake model."""

import sys
import types
from unittest import mock

import numpy as np
import pytest


CONFIG = {
    "model": "large-v3-turbo",
    "device": "cuda",
    "compute_type": "float16",
    "language": "en",
    "beam_size": 5,
    "vad_filter": True,
}


def _segment(text):
    seg = mock.Mock()
    seg.text = text
    return seg


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel; optionally fails on CUDA."""

    fail_on = set()
    segments = []

    def __init__(self, model_name, device="cuda", compute_type="float16"):
        if device in self.fail_on:
            raise RuntimeError(f"simulated load failure on {device}")
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, audio, **kwargs):
        return iter(self.segments), {"kwargs": kwargs}


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    FakeWhisperModel.fail_on = set()
    FakeWhisperModel.segments = []
    return module


def test_loads_on_cuda(fake_faster_whisper):
    from whisperflow.transcribe import Transcriber

    tr = Transcriber(CONFIG)
    assert tr.device_used == "cuda"
    assert tr.compute_type_used == "float16"


def test_falls_back_to_cpu_int8_when_cuda_fails(fake_faster_whisper):
    from whisperflow.transcribe import Transcriber

    FakeWhisperModel.fail_on = {"cuda"}
    tr = Transcriber(CONFIG)
    assert tr.device_used == "cpu"
    assert tr.compute_type_used == "int8"


def test_raises_clear_error_when_both_devices_fail(fake_faster_whisper):
    from whisperflow.transcribe import Transcriber

    FakeWhisperModel.fail_on = {"cuda", "cpu"}
    with pytest.raises(RuntimeError, match="CUDA or on CPU"):
        Transcriber(CONFIG)


def test_no_fallback_when_cpu_requested_explicitly(fake_faster_whisper):
    from whisperflow.transcribe import Transcriber

    FakeWhisperModel.fail_on = {"cpu"}
    with pytest.raises(RuntimeError, match="simulated load failure"):
        Transcriber({**CONFIG, "device": "cpu"})


def test_transcribe_joins_and_strips_segments(fake_faster_whisper):
    from whisperflow.transcribe import Transcriber

    FakeWhisperModel.segments = [_segment(" Hello"), _segment(" world.")]
    tr = Transcriber(CONFIG)
    audio = np.zeros(16000, dtype=np.float32)
    assert tr.transcribe(audio) == "Hello world."


def test_transcribe_empty_audio_short_circuits(fake_faster_whisper):
    from whisperflow.transcribe import Transcriber

    tr = Transcriber(CONFIG)
    assert tr.transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert tr.transcribe(None) == ""


def test_warmup_runs_a_decode(fake_faster_whisper):
    from whisperflow.transcribe import Transcriber

    FakeWhisperModel.segments = [_segment("noise")]
    tr = Transcriber(CONFIG)
    tr.warmup()  # must not raise

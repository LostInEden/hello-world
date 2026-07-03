"""End-to-end pipeline with the heavy components faked out.

Simulates: hotkey down -> audio captured -> hotkey up -> transcribe -> clean
-> inject, and asserts the cleaned text is what gets injected.
"""

import time
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("pynput")
pytest.importorskip("sounddevice")

import whisperflow.app as app_module  # noqa: E402
from whisperflow.config import load_config  # noqa: E402


class FakeRecorder:
    def __init__(self, sample_rate=16000, device=None):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio = np.ones(16000, dtype=np.float32)  # 1s

    def start(self):
        self.recording = True

    def stop(self):
        self.recording = False
        return self.audio

    def duration(self, audio):
        return len(audio) / self.sample_rate


class FakeTranscriber:
    def __init__(self, config):
        self.config = config
        self.device_used = "cuda"
        self.compute_type_used = "float16"

    def warmup(self):
        pass

    def transcribe(self, audio):
        return "um hello world no wait hello there" if len(audio) else ""


class FakeCleaner:
    def __init__(self, config):
        self.enabled = config.get("enabled", True)

    def is_available(self):
        return True

    def clean(self, text):
        return "Hello there."


class FakeInjector:
    def __init__(self, config):
        self.injected = []

    def inject(self, text):
        self.injected.append(text)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(app_module, "Recorder", FakeRecorder)
    monkeypatch.setattr(app_module, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(app_module, "Cleaner", FakeCleaner)
    monkeypatch.setattr(app_module, "Injector", FakeInjector)
    return app_module.App(load_config(None))


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_full_utterance_injects_cleaned_text(app):
    app._on_start()
    assert app.recorder.recording
    app._on_stop()
    assert _wait_for(lambda: app.injector.injected)
    assert app.injector.injected == ["Hello there."]


def test_cleanup_disabled_injects_raw_transcript(monkeypatch, app):
    app.cleaner.enabled = False
    app._on_start()
    app._on_stop()
    assert _wait_for(lambda: app.injector.injected)
    assert app.injector.injected == ["um hello world no wait hello there"]


def test_too_short_recording_is_ignored(app):
    app.recorder.audio = np.ones(1600, dtype=np.float32)  # 0.1s < min_seconds
    app._on_start()
    app._on_stop()
    time.sleep(0.3)
    assert app.injector.injected == []


def test_empty_transcript_injects_nothing(app):
    app.transcriber.transcribe = lambda audio: ""
    app._on_start()
    app._on_stop()
    time.sleep(0.3)
    assert app.injector.injected == []


def test_injection_waits_for_hotkey_release(app):
    """If a combo key is still physically down, injection is delayed."""
    app.listener._pressed.add("ctrl")
    app._on_start()
    start = time.time()
    app._on_stop()
    time.sleep(0.15)
    assert app.injector.injected == []  # still waiting on the held key
    app.listener._pressed.discard("ctrl")
    assert _wait_for(lambda: app.injector.injected)
    assert time.time() - start < 3.0


def test_transcription_serialized_by_model_lock(app):
    """Two overlapping utterances must not run the model concurrently."""
    concurrent = []
    active = []

    def slow_transcribe(audio):
        active.append(1)
        concurrent.append(len(active))
        time.sleep(0.1)
        active.pop()
        return "some long enough transcript here"

    app.transcriber.transcribe = slow_transcribe
    app._on_start()
    app._on_stop()
    app._on_start()
    app._on_stop()
    assert _wait_for(lambda: len(app.injector.injected) == 2)
    assert max(concurrent) == 1

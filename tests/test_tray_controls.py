"""Tray-facing App controls: pause, model switching + persistence, shutdown.

Exercises the App methods the tray menu calls; the pystray UI itself is a
thin wrapper and needs a real tray host to run.
"""

import time
from unittest import mock

import pytest
import yaml

pytest.importorskip("pynput")
pytest.importorskip("sounddevice")

import whisperflow.app as app_module  # noqa: E402
from whisperflow.config import load_config, save_override  # noqa: E402

from .test_app import (  # noqa: E402
    FakeCleaner,
    FakeInjector,
    FakeRecorder,
    FakeTranscriber,
    _wait_for,
)


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "Recorder", FakeRecorder)
    monkeypatch.setattr(app_module, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(app_module, "Cleaner", FakeCleaner)
    monkeypatch.setattr(app_module, "Injector", FakeInjector)
    config_path = tmp_path / "config.yaml"
    return app_module.App(load_config(None), config_path=str(config_path))


def test_pause_blocks_activation_and_resume_restores(app):
    app.pause(True)
    assert app.paused
    from pynput import keyboard

    app.listener._on_press(keyboard.Key.ctrl_l)
    app.listener._on_press(keyboard.Key.alt_l)
    assert not app.recorder.recording  # suspended: hotkey ignored
    app.listener._on_release(keyboard.Key.alt_l)
    app.listener._on_release(keyboard.Key.ctrl_l)

    app.pause(False)
    app.listener._on_press(keyboard.Key.ctrl_l)
    app.listener._on_press(keyboard.Key.alt_l)
    assert app.recorder.recording


def test_pause_mid_recording_finishes_the_utterance(app):
    app._on_start()
    app.listener._active = True  # simulate combo currently held
    app.pause(True)
    assert not app.recorder.recording  # force_release stopped the capture
    deadline = time.time() + 5
    while not app.injector.injected and time.time() < deadline:
        time.sleep(0.01)
    assert app.injector.injected  # what was said still got processed


def test_injection_does_not_unpause(app):
    app.pause(True)
    app._inject_safely("hello")
    assert app.listener._suspended  # still paused after injection resumes


def test_set_cleanup_model_updates_and_persists(app, tmp_path):
    app.cleaner.model = "gemma3:4b"
    app.set_cleanup_model("qwen2.5:3b")
    assert app.cleaner.enabled
    assert app.cleaner.model == "qwen2.5:3b"
    saved = yaml.safe_load(open(app.config_path))
    assert saved == {"cleanup": {"enabled": True, "model": "qwen2.5:3b"}}

    app.set_cleanup_model(None)
    assert not app.cleaner.enabled
    saved = yaml.safe_load(open(app.config_path))
    assert saved["cleanup"]["enabled"] is False
    assert saved["cleanup"]["model"] == "qwen2.5:3b"  # remembered for re-enable


def test_save_override_merges_with_existing_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("hotkey:\n  combo: f9\ncleanup:\n  timeout: 5\n", encoding="utf-8")
    save_override(str(path), {"cleanup": {"model": "qwen2.5:3b"}})
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved == {
        "hotkey": {"combo": "f9"},
        "cleanup": {"timeout": 5, "model": "qwen2.5:3b"},
    }


def test_save_override_without_path_is_noop():
    assert save_override(None, {"a": 1}) is False


def test_set_hotkey_mode_switches_and_persists(app):
    app.set_hotkey_mode("toggle")
    assert app.listener.mode == "toggle"
    saved = yaml.safe_load(open(app.config_path))
    assert saved["hotkey"]["mode"] == "toggle"
    app.set_hotkey_mode("push_to_talk")
    assert app.listener.mode == "push_to_talk"


def test_set_hotkey_mode_rejects_unknown(app):
    app.set_hotkey_mode("bogus")
    assert app.listener.mode == "push_to_talk"  # unchanged


def test_set_hotkey_mode_mid_recording_finishes_utterance(app):
    app._on_start()
    app.listener._active = True  # combo currently engaged
    app.set_hotkey_mode("toggle")
    assert not app.recorder.recording  # recording was stopped, not orphaned
    assert _wait_for(lambda: app.injector.injected)


def test_toggle_mode_full_cycle_through_listener(app):
    """Press ctrl+alt to start, release, press again to stop and process."""
    from pynput import keyboard

    app.set_hotkey_mode("toggle")
    app.listener._on_press(keyboard.Key.ctrl_l)
    app.listener._on_press(keyboard.Key.alt_l)
    assert app.recorder.recording
    app.listener._on_release(keyboard.Key.alt_l)
    app.listener._on_release(keyboard.Key.ctrl_l)
    assert app.recorder.recording  # still recording after release
    app.listener._on_press(keyboard.Key.ctrl_l)
    app.listener._on_press(keyboard.Key.alt_l)
    assert not app.recorder.recording  # second press stopped it
    app.listener._on_release(keyboard.Key.alt_l)
    app.listener._on_release(keyboard.Key.ctrl_l)
    assert _wait_for(lambda: app.injector.injected)


def test_tray_mode_menu(app):
    pytest.importorskip("pystray")
    from whisperflow.tray import Tray

    tray = Tray(app, hotkey_label="ctrl+alt")
    items = list(tray._mode_items())
    assert [str(i.text) for i in items] == ["Hold to talk", "Press to start/stop"]
    assert items[0].checked and not items[1].checked
    app.set_hotkey_mode("toggle")
    items = list(tray._mode_items())
    assert items[1].checked and not items[0].checked


def test_shutdown_closes_overlay(app):
    app.shutdown()
    assert app.overlay._closing


def test_list_cleanup_models_delegates_to_cleaner(app):
    app.cleaner.list_models = mock.Mock(return_value=["a:1b", "b:3b"])
    assert app.list_cleanup_models() == ["a:1b", "b:3b"]


def test_tray_menu_lists_models(app):
    pystray = pytest.importorskip("pystray")  # noqa: F841
    from whisperflow.tray import Tray

    app.cleaner.list_models = mock.Mock(return_value=["gemma3:4b", "qwen2.5:3b"])
    app.cleaner.model = "qwen2.5:3b"
    tray = Tray(app, hotkey_label="ctrl+alt")
    items = list(tray._model_items())
    labels = [str(i.text) for i in items]
    assert labels == ["Off (raw transcript)", "gemma3:4b", "qwen2.5:3b"]
    # the configured model shows as selected, "Off" does not
    assert items[2].checked
    assert not items[0].checked


def test_tray_menu_falls_back_to_configured_model(app):
    pytest.importorskip("pystray")
    from whisperflow.tray import Tray

    app.cleaner.list_models = mock.Mock(return_value=[])  # Ollama down
    app.cleaner.model = "qwen2.5:3b"
    tray = Tray(app)
    labels = [str(i.text) for i in tray._model_items()]
    assert labels == ["Off (raw transcript)", "qwen2.5:3b"]

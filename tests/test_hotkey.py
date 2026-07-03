"""Hotkey state machine: activation, auto-repeat immunity, toggle, suspension.

Drives the listener's _on_press/_on_release handlers directly with real pynput
key objects — no OS-level keyboard hook is installed.
"""

import pytest

pynput = pytest.importorskip("pynput")  # needs a display server on Linux
from pynput import keyboard  # noqa: E402

from whisperflow.hotkey import HotkeyListener, parse_combo  # noqa: E402


CTRL = keyboard.Key.ctrl_l
ALT = keyboard.Key.alt_l
V = keyboard.KeyCode.from_char("v")


class Spy:
    def __init__(self):
        self.events = []

    def activate(self):
        self.events.append("on")

    def deactivate(self):
        self.events.append("off")


def make(combo="ctrl+alt", mode="push_to_talk"):
    spy = Spy()
    listener = HotkeyListener(combo, spy.activate, spy.deactivate, mode=mode)
    return listener, spy


def test_parse_combo():
    assert parse_combo("Ctrl + Alt") == {"ctrl", "alt"}
    assert parse_combo("f9") == {"f9"}


def test_invalid_combo_rejected():
    with pytest.raises(ValueError):
        HotkeyListener("", lambda: None, lambda: None)


def test_push_to_talk_full_cycle():
    listener, spy = make()
    listener._on_press(CTRL)
    assert spy.events == []
    listener._on_press(ALT)
    assert spy.events == ["on"]
    listener._on_release(ALT)
    assert spy.events == ["on", "off"]
    listener._on_release(CTRL)
    assert spy.events == ["on", "off"]


def test_auto_repeat_does_not_retrigger():
    listener, spy = make(combo="f9")
    f9 = keyboard.Key.f9
    listener._on_press(f9)
    listener._on_press(f9)  # OS auto-repeat
    listener._on_press(f9)
    assert spy.events == ["on"]
    listener._on_release(f9)
    assert spy.events == ["on", "off"]


def test_toggle_mode_alternates():
    listener, spy = make(mode="toggle")
    for _ in range(2):  # press and release combo twice
        listener._on_press(CTRL)
        listener._on_press(ALT)
        listener._on_release(ALT)
        listener._on_release(CTRL)
    assert spy.events == ["on", "off"]


def test_toggle_mode_ignores_auto_repeat_while_held():
    listener, spy = make(combo="f9", mode="toggle")
    f9 = keyboard.Key.f9
    listener._on_press(f9)
    listener._on_press(f9)  # auto-repeat must not toggle back off
    listener._on_release(f9)
    assert spy.events == ["on"]


def test_unrelated_keys_do_not_activate():
    listener, spy = make()
    listener._on_press(CTRL)
    listener._on_press(V)  # ctrl+v is not ctrl+alt
    assert spy.events == []


def test_suspension_blocks_callbacks_but_tracks_state():
    listener, spy = make()
    listener.suspend()
    listener._on_press(CTRL)
    listener._on_press(ALT)
    assert spy.events == []
    assert listener.target_down()  # state still tracked
    listener._on_release(ALT)
    listener._on_release(CTRL)
    listener.resume()
    assert not listener.target_down()
    # normal operation resumes cleanly
    listener._on_press(CTRL)
    listener._on_press(ALT)
    assert spy.events == ["on"]


def test_target_down_reflects_partial_hold():
    listener, spy = make()
    assert not listener.target_down()
    listener._on_press(ALT)
    assert listener.target_down()  # any combo key held counts
    listener._on_release(ALT)
    assert not listener.target_down()

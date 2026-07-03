"""Global push-to-talk / toggle hotkey handling via pynput.

Supports modifier combos ("ctrl+alt", "ctrl+shift+space"), function keys ("f9"),
and plain letters/digits. Two activation modes:

  - push_to_talk: fires ``on_activate`` when the full combo goes down and
    ``on_deactivate`` when it is released.
  - toggle:       fires ``on_activate`` / ``on_deactivate`` on alternating
    presses of the full combo.

A ``_combo_satisfied`` latch makes both modes immune to key auto-repeat.
"""

from __future__ import annotations

from typing import Callable, Optional, Set

from pynput import keyboard

# Map pynput's left/right modifier variants down to a single canonical name.
_MOD_MAP = {
    keyboard.Key.ctrl: "ctrl",
    keyboard.Key.ctrl_l: "ctrl",
    keyboard.Key.ctrl_r: "ctrl",
    keyboard.Key.alt: "alt",
    keyboard.Key.alt_l: "alt",
    keyboard.Key.alt_r: "alt",
    keyboard.Key.alt_gr: "alt",
    keyboard.Key.shift: "shift",
    keyboard.Key.shift_l: "shift",
    keyboard.Key.shift_r: "shift",
    keyboard.Key.cmd: "cmd",
    keyboard.Key.cmd_l: "cmd",
    keyboard.Key.cmd_r: "cmd",
}


def _normalize(key) -> Optional[str]:  # noqa: ANN001 - pynput key union
    """Reduce a pynput key event to a canonical lowercase token, or None."""
    if key in _MOD_MAP:
        return _MOD_MAP[key]
    if isinstance(key, keyboard.Key):
        return key.name  # e.g. "space", "f9", "esc"
    if isinstance(key, keyboard.KeyCode):
        # Prefer the printable character when it's a normal alphanumeric.
        if key.char and key.char.isalnum():
            return key.char.lower()
        # Fall back to the virtual key code so modifier+letter combos still work
        # (Ctrl+letter often reports a control char or None for `.char`).
        vk = key.vk
        if vk is not None:
            if 65 <= vk <= 90:  # A-Z
                return chr(vk).lower()
            if 48 <= vk <= 57:  # 0-9 (top row)
                return chr(vk)
            if 96 <= vk <= 105:  # numpad 0-9
                return str(vk - 96)
        return None
    return None


def parse_combo(combo: str) -> Set[str]:
    """Turn "ctrl+alt" into {"ctrl", "alt"}."""
    return {part.strip().lower() for part in combo.split("+") if part.strip()}


class HotkeyListener:
    """Listens globally for a hotkey combo and drives activate/deactivate callbacks."""

    def __init__(
        self,
        combo: str,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
        mode: str = "push_to_talk",
    ):
        self.target = parse_combo(combo)
        if not self.target:
            raise ValueError(f"Invalid hotkey combo: {combo!r}")
        self.on_activate = on_activate
        self.on_deactivate = on_deactivate
        self.mode = mode

        self._pressed: Set[str] = set()
        self._combo_satisfied = False
        self._active = False
        self._suspended = False
        self._listener: Optional[keyboard.Listener] = None

    # -- suspension: keeps our own synthetic keystrokes (Ctrl+V paste, typed
    #    text) from being interpreted as hotkey activity during injection.
    #    Key state is still tracked so physical holds don't desync. --

    def suspend(self) -> None:
        """Stop firing activate/deactivate callbacks until resume()."""
        self._suspended = True

    def resume(self) -> None:
        self._suspended = False

    def target_down(self) -> bool:
        """True while any key of the hotkey combo is still held down."""
        return bool(self.target & self._pressed)

    def force_release(self) -> None:
        """Deactivate as if the combo was released (e.g. when pausing while
        the user is mid-recording). Safe to call from any thread."""
        self._combo_satisfied = False
        if self._active:
            self._active = False
            self.on_deactivate()

    def _on_press(self, key) -> None:  # noqa: ANN001
        token = _normalize(key)
        if token is None:
            return
        self._pressed.add(token)
        if self._suspended:
            return

        if self.target.issubset(self._pressed) and not self._combo_satisfied:
            self._combo_satisfied = True
            if self.mode == "toggle":
                self._active = not self._active
                (self.on_activate if self._active else self.on_deactivate)()
            else:  # push_to_talk
                if not self._active:
                    self._active = True
                    self.on_activate()

    def _on_release(self, key) -> None:  # noqa: ANN001
        token = _normalize(key)
        if token is None:
            return
        self._pressed.discard(token)
        if self._suspended:
            return

        if not self.target.issubset(self._pressed):
            self._combo_satisfied = False
            if self.mode == "push_to_talk" and self._active:
                self._active = False
                self.on_deactivate()

    def start(self) -> None:
        """Start listening (non-blocking; runs on its own thread)."""
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()

    def join(self) -> None:
        """Block the calling thread until the listener stops."""
        if self._listener is not None:
            self._listener.join()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

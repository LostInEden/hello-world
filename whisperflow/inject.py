"""Text injection into the currently focused application.

Two strategies:
  - "paste": put the text on the clipboard and send Ctrl+V (fast, reliable in
    most apps; optionally restores the prior clipboard afterwards).
  - "type":  simulate individual keystrokes (works in apps that block synthetic
    paste, but slower and sensitive to keyboard layout).
"""

from __future__ import annotations

import time
from typing import Any, Dict

import pyperclip
from pynput.keyboard import Controller, Key


class Injector:
    """Delivers final text to the focused window."""

    def __init__(self, config: Dict[str, Any]):
        self.method = config.get("method", "paste")
        self.restore_clipboard = bool(config.get("restore_clipboard", True))
        self._keyboard = Controller()

    def inject(self, text: str) -> None:
        if not text:
            return
        if self.method == "type":
            self._type(text)
        else:
            self._paste(text)

    def _paste(self, text: str) -> None:
        previous = None
        if self.restore_clipboard:
            try:
                previous = pyperclip.paste()
            except Exception:  # pragma: no cover - clipboard can be finicky
                previous = None

        pyperclip.copy(text)
        # Give the OS a moment to register the new clipboard contents.
        time.sleep(0.05)

        with self._keyboard.pressed(Key.ctrl):
            self._keyboard.press("v")
            self._keyboard.release("v")

        if self.restore_clipboard:
            # Wait for the target app to consume the paste before restoring.
            time.sleep(0.15)
            try:
                pyperclip.copy(previous if previous is not None else "")
            except Exception:  # pragma: no cover
                pass

    def _type(self, text: str) -> None:
        self._keyboard.type(text)

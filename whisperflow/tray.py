"""System tray icon: run WhisperFlow like a normal background app.

Right-click menu: pause/resume dictation, pick the cleanup model (the list is
whatever is actually pulled in Ollama, refreshed every time the menu opens),
open the config file, quit. pystray's Windows backend runs fine off the main
thread, which tkinter (the overlay) owns.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading


def _make_icon_image():
    """Draw the tray icon: the status pill's bars on a dark rounded square."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([2, 2, 62, 62], radius=16, fill=(29, 29, 36, 255))
    for i, h in enumerate([18, 30, 40, 30, 18]):
        x = 12 + i * 9
        draw.rounded_rectangle([x, 32 - h // 2, x + 6, 32 + h // 2], radius=3, fill=(245, 245, 247, 255))
    return img


class Tray:
    """Thin wiring between pystray and the running App."""

    def __init__(self, app, hotkey_label: str = "the hotkey"):  # noqa: ANN001
        self.app = app
        self.hotkey_label = hotkey_label
        self._icon = None

    # -- menu content --

    def _model_items(self):
        import pystray

        cleaner = self.app.cleaner

        def select(name):  # noqa: ANN001 - closure factories; pystray inspects signatures
            def handler(icon, item):  # noqa: ANN001
                self.app.set_cleanup_model(name)

            return handler

        def is_current(name):  # noqa: ANN001
            def check(item):  # noqa: ANN001
                return cleaner.enabled and cleaner.model == name

            return check

        yield pystray.MenuItem(
            "Off (raw transcript)",
            select(None),
            radio=True,
            checked=lambda item: not cleaner.enabled,
        )
        models = self.app.list_cleanup_models()
        if not models and cleaner.model:
            models = [cleaner.model]  # Ollama down: still show the configured one
        for name in models:
            yield pystray.MenuItem(name, select(name), radio=True, checked=is_current(name))

    def _mode_items(self):
        import pystray

        listener = self.app.listener

        def select(mode):  # noqa: ANN001
            def handler(icon, item):  # noqa: ANN001
                self.app.set_hotkey_mode(mode)

            return handler

        def is_current(mode):  # noqa: ANN001
            def check(item):  # noqa: ANN001
                return listener.mode == mode

            return check

        yield pystray.MenuItem(
            "Hold to talk", select("push_to_talk"), radio=True, checked=is_current("push_to_talk")
        )
        yield pystray.MenuItem(
            "Press to start/stop", select("toggle"), radio=True, checked=is_current("toggle")
        )

    def _menu(self):
        import pystray

        return pystray.Menu(
            pystray.MenuItem(f"WhisperFlow — {self.hotkey_label} to dictate", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Pause dictation",
                lambda icon, item: self.app.pause(not self.app.paused),
                checked=lambda item: self.app.paused,
            ),
            pystray.MenuItem("Hotkey mode", pystray.Menu(self._mode_items)),
            pystray.MenuItem("Cleanup model", pystray.Menu(self._model_items)),
            pystray.MenuItem("Open config file", lambda icon, item: self._open_config()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda icon, item: self._quit()),
        )

    # -- actions --

    def _open_config(self) -> None:
        path = self.app.config_path or "config.yaml"
        if not os.path.exists(path) and os.path.exists("config.example.yaml"):
            shutil.copyfile("config.example.yaml", path)
        try:
            if sys.platform == "win32":
                os.startfile(os.path.abspath(path))  # noqa: S606
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            print(f"[tray] could not open {path}: {exc}", flush=True)

    def _quit(self) -> None:
        self.app.shutdown()
        if self._icon is not None:
            self._icon.stop()

    # -- lifecycle --

    def start(self) -> None:
        import pystray

        self._icon = pystray.Icon("whisperflow", _make_icon_image(), "WhisperFlow", self._menu())
        threading.Thread(target=self._icon.run, daemon=True, name="tray").start()

    def stop(self) -> None:
        if self._icon is not None:
            self._icon.stop()
            self._icon = None

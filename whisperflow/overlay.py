"""Floating status pill, Wispr-Flow style.

A small frameless, always-on-top window at the bottom-center of the screen:

  - hidden while idle
  - vertical bars dancing with the live microphone level while recording
  - a traveling-wave animation while transcribing/cleaning up

tkinter needs to own the main thread, so ``run()`` blocks with the Tk main
loop and other threads change what's displayed via ``set_state()`` (a plain
attribute write, applied by the animation tick). The window never takes
focus — on Windows it's additionally marked NOACTIVATE and click-through so
it can't steal keystrokes from the app being dictated into.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Callable, Dict, Optional

BAR_COUNT = 5
WIDTH = 148
HEIGHT = 40
BAR_WIDTH = 6
BAR_GAP = 8
BAR_MIN = 5.0
BAR_MAX = 24.0
FPS = 30

# Color key treated as fully transparent on Windows (gives the pill its
# rounded shape). On platforms without -transparentcolor it's just a dark bg.
TRANSPARENT = "#000001"
PILL = "#1d1d24"
BAR_RECORDING = "#f5f5f7"
BAR_PROCESSING = "#8f97a8"


def bar_heights(state: str, phase: float, level: float, count: int = BAR_COUNT) -> list[float]:
    """Pure animation math: bar heights in pixels for a given state.

    recording:  center-weighted bars scale with the mic level and shimmer.
    processing: a constant-energy wave travels across the bars.
    anything else: flat at the minimum height.
    """
    heights = []
    for i in range(count):
        if state == Overlay.RECORDING:
            center_weight = 1.0 - 0.35 * abs(i - (count - 1) / 2) / max(1, (count - 1) / 2)
            shimmer = 0.7 + 0.3 * math.sin(phase * (5.1 + 0.7 * i) + i * 1.9)
            # Small baseline so the pill visibly "breathes" even in silence.
            drive = (0.12 + 0.88 * max(0.0, min(1.0, level))) * center_weight * shimmer
            heights.append(BAR_MIN + (BAR_MAX - BAR_MIN) * min(1.0, drive))
        elif state == Overlay.PROCESSING:
            wave = 0.5 * (1.0 + math.sin(phase * 6.0 - i * 1.1))
            heights.append(BAR_MIN + (BAR_MAX - BAR_MIN) * 0.55 * wave)
        else:
            heights.append(BAR_MIN)
    return heights


class Overlay:
    HIDDEN = "hidden"
    RECORDING = "recording"
    PROCESSING = "processing"

    def __init__(self, config: Dict[str, Any], level_provider: Optional[Callable[[], float]] = None):
        self.position = config.get("position", "bottom_center")
        self.margin = int(config.get("margin", 56))
        self._level_provider = level_provider or (lambda: 0.0)
        self._state = self.HIDDEN
        self._closing = False
        self._shown = False
        self._phase = 0.0
        self._last_tick = 0.0
        self._root = None
        self._bars: list[int] = []

    # -- thread-safe surface (called from hotkey/worker threads) --

    def set_state(self, state: str) -> None:
        self._state = state

    def close(self) -> None:
        self._closing = True

    # -- main-thread side --

    def run(self) -> None:
        """Build the window and block in the Tk main loop."""
        import tkinter as tk

        root = tk.Tk()
        self._root = root
        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-transparentcolor", TRANSPARENT)
        except tk.TclError:  # not supported outside Windows
            pass

        canvas = tk.Canvas(
            root, width=WIDTH, height=HEIGHT, bg=TRANSPARENT,
            highlightthickness=0, bd=0,
        )
        canvas.pack()
        self._canvas = canvas

        # Rounded pill: two circles + a joining rectangle.
        r = HEIGHT // 2
        canvas.create_oval(0, 0, HEIGHT, HEIGHT, fill=PILL, outline=PILL)
        canvas.create_oval(WIDTH - HEIGHT, 0, WIDTH, HEIGHT, fill=PILL, outline=PILL)
        canvas.create_rectangle(r, 0, WIDTH - r, HEIGHT, fill=PILL, outline=PILL)

        total = BAR_COUNT * BAR_WIDTH + (BAR_COUNT - 1) * BAR_GAP
        x0 = (WIDTH - total) / 2
        mid = HEIGHT / 2
        self._bars = []
        for i in range(BAR_COUNT):
            x = x0 + i * (BAR_WIDTH + BAR_GAP)
            self._bars.append(
                canvas.create_rectangle(
                    x, mid - BAR_MIN / 2, x + BAR_WIDTH, mid + BAR_MIN / 2,
                    fill=BAR_RECORDING, outline="",
                )
            )

        self._place(root)
        self._suppress_focus(root)

        self._last_tick = time.monotonic()
        root.after(int(1000 / FPS), self._tick)
        try:
            root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def _place(self, root) -> None:  # noqa: ANN001
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w - WIDTH) // 2
        y = self.margin if self.position == "top_center" else screen_h - HEIGHT - self.margin
        root.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")

    def _suppress_focus(self, root) -> None:  # noqa: ANN001
        """Make the window unfocusable and click-through on Windows."""
        if sys.platform != "win32":
            return
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000

        user32 = ctypes.windll.user32
        root.update_idletasks()
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        style = get_style(hwnd, GWL_EXSTYLE)
        set_style(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT)

    def _tick(self) -> None:
        root = self._root
        if root is None:
            return
        if self._closing:
            root.quit()
            return

        state = self._state
        visible = state != self.HIDDEN
        if visible and not self._shown:
            root.deiconify()
            root.attributes("-topmost", True)
            root.lift()
            self._shown = True
        elif not visible and self._shown:
            root.withdraw()
            self._shown = False

        if self._shown:
            now = time.monotonic()
            self._phase += now - self._last_tick
            self._last_tick = now

            level = 0.0
            if state == self.RECORDING:
                try:
                    level = self._level_provider()
                except Exception:
                    level = 0.0
            color = BAR_RECORDING if state == self.RECORDING else BAR_PROCESSING
            mid = HEIGHT / 2
            for item, h in zip(self._bars, bar_heights(state, self._phase, level)):
                x0, _y0, x1, _y1 = self._canvas.coords(item)
                self._canvas.coords(item, x0, mid - h / 2, x1, mid + h / 2)
                self._canvas.itemconfig(item, fill=color)
        else:
            self._last_tick = time.monotonic()

        root.after(int(1000 / FPS), self._tick)

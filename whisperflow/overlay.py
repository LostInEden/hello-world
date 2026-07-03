"""Floating status pill, styled after Wispr Flow's indicator.

A light rounded pill with a dark outline and a thin vertical-bar waveform:
while recording, the bars are a scrolling history of the live microphone
level (newest on the right), so the waveform visibly travels as you speak;
while processing, a soft wave ripples across the bars. Hidden when idle.

Rendering runs at 60 fps with exponential smoothing toward target heights,
so bars glide instead of jumping. tkinter needs to own the main thread, so
``run()`` blocks with the Tk main loop and other threads change what's
displayed via ``set_state()`` (a plain attribute write, applied by the
animation tick). The window never takes focus — on Windows it's additionally
marked NOACTIVATE and click-through so it can't steal keystrokes from the
app being dictated into.
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

BAR_COUNT = 21
BAR_WIDTH = 4
BAR_GAP = 3
BAR_MIN = 4.0             # == BAR_WIDTH, so silent bars are perfect dots
BAR_MAX = 26.0
WIDTH = 200
HEIGHT = 48
FPS = 60
SMOOTHING = 0.35          # per-frame lerp factor toward target heights
PUSH_INTERVAL = 0.05      # seconds between waveform scroll steps
SS = 3                    # supersampling factor: tkinter can't antialias, so
                          # frames are drawn 3x with Pillow and downscaled

# Color key treated as fully transparent on Windows (gives the pill its
# rounded shape). On platforms without -transparentcolor it's just a border.
TRANSPARENT = "#000001"
PILL_BG = "#faf7ee"
INK = "#141414"
INK_PROCESSING = "#8d8d8d"
OUTLINE_WIDTH = 3


def level_to_height(level: float) -> float:
    """Map a mic level in [0, 1] to a bar height in pixels.

    The 0.6 gamma lifts quiet speech so the waveform looks alive at
    conversational volume instead of only when shouting.
    """
    level = max(0.0, min(1.0, level))
    return BAR_MIN + (BAR_MAX - BAR_MIN) * (level ** 0.6)


def wave_heights(phase: float, count: int = BAR_COUNT) -> list:
    """Processing animation: a gentle wave rippling across the bars."""
    heights = []
    for i in range(count):
        w = 0.5 * (1.0 + math.sin(phase * 5.0 - i * 0.55))
        heights.append(BAR_MIN + (BAR_MAX - BAR_MIN) * 0.45 * w)
    return heights


def render_frame(bar_pixel_heights: list, bar_color: str):
    """Draw one antialiased frame (a PIL Image): the outlined pill plus the
    waveform bars, supersampled at ``SS``x and LANCZOS-downscaled so edges and
    the bars' rounded ends come out smooth despite tkinter's aliased canvas.
    """
    from PIL import Image, ImageDraw

    w, h = WIDTH * SS, HEIGHT * SS
    img = Image.new("RGB", (w, h), TRANSPARENT)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=h // 2, fill=INK)
    o = OUTLINE_WIDTH * SS
    draw.rounded_rectangle(
        [o, o, w - 1 - o, h - 1 - o], radius=(h - 2 * o) // 2, fill=PILL_BG
    )

    total = BAR_COUNT * BAR_WIDTH + (BAR_COUNT - 1) * BAR_GAP
    x0 = (WIDTH - total) / 2
    mid = HEIGHT / 2
    for i, bh in enumerate(bar_pixel_heights):
        bh = max(float(BAR_WIDTH), float(bh))  # never thinner than a dot
        x = (x0 + i * (BAR_WIDTH + BAR_GAP)) * SS
        draw.rounded_rectangle(
            [x, (mid - bh / 2) * SS, x + BAR_WIDTH * SS, (mid + bh / 2) * SS],
            radius=BAR_WIDTH * SS / 2,
            fill=bar_color,
        )
    return img.resize((WIDTH, HEIGHT), Image.LANCZOS)


class Overlay:
    HIDDEN = "hidden"
    RECORDING = "recording"
    PROCESSING = "processing"

    def __init__(self, config: Dict[str, Any], level_provider: Optional[Callable[[], float]] = None):
        self.position = config.get("position", "bottom_center")
        self.margin = int(config.get("margin", 56))
        self._level_provider = level_provider or (lambda: 0.0)
        self._state = self.HIDDEN
        self._prev_state = self.HIDDEN
        self._closing = False
        self._shown = False
        self._phase = 0.0
        self._last_tick = 0.0
        self._last_push = 0.0
        self._peak = 0.0
        self._history = deque([0.0] * BAR_COUNT, maxlen=BAR_COUNT)
        self._display = [BAR_MIN] * BAR_COUNT
        self._root = None
        self._bars: list = []

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
        self._image_item = canvas.create_image(0, 0, anchor="nw")

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

    def _targets(self, now: float) -> list:
        """Per-bar target heights for the current state."""
        if self._state == self.RECORDING:
            try:
                level = self._level_provider()
            except Exception:
                level = 0.0
            self._peak = max(self._peak, level)
            if now - self._last_push >= PUSH_INTERVAL:
                self._history.append(self._peak)
                self._peak = 0.0
                self._last_push = now
            return [level_to_height(l) for l in self._history]
        if self._state == self.PROCESSING:
            return wave_heights(self._phase)
        return [BAR_MIN] * BAR_COUNT

    def _tick(self) -> None:
        root = self._root
        if root is None:
            return
        if self._closing:
            root.quit()
            return

        state = self._state
        if state == self.RECORDING and self._prev_state != self.RECORDING:
            # fresh utterance: start the waveform from a clean slate
            self._history.extend([0.0] * BAR_COUNT)
            self._peak = 0.0
        self._prev_state = state

        visible = state != self.HIDDEN
        if visible and not self._shown:
            root.deiconify()
            root.attributes("-topmost", True)
            root.lift()
            self._shown = True
        elif not visible and self._shown:
            root.withdraw()
            self._shown = False

        now = time.monotonic()
        if self._shown:
            from PIL import ImageTk

            self._phase += now - self._last_tick
            targets = self._targets(now)
            for i in range(BAR_COUNT):
                self._display[i] += (targets[i] - self._display[i]) * SMOOTHING
            color = INK if state == self.RECORDING else INK_PROCESSING
            # keep a reference on self: Tk only borrows the image
            self._photo = ImageTk.PhotoImage(render_frame(self._display, color))
            self._canvas.itemconfig(self._image_item, image=self._photo)
        self._last_tick = now

        root.after(int(1000 / FPS), self._tick)

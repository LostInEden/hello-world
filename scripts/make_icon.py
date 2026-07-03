"""Generate assets/whisperflow.ico (and a preview PNG) with Pillow.

Design: the app's status pill — cream stadium with a dark waveform — on a
dark rounded tile. Small sizes (16/24/32) drop the pill and draw cream bars
directly on the tile so the icon stays legible in the taskbar.

Run from the repo root:  python scripts/make_icon.py
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw

INK = (20, 20, 24)
TILE_TOP = (38, 38, 48)
TILE_BOTTOM = (16, 16, 22)
CREAM = (250, 247, 238)

# Symmetric, hand-picked waveform (fractions of max bar height)
WAVE = [0.30, 0.55, 0.80, 1.00, 0.70, 0.45, 0.62, 0.35]

SS = 4  # supersampling factor


def _tile(size: int) -> Image.Image:
    """Dark rounded-square tile with a subtle vertical gradient."""
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    grad = Image.new("RGBA", (s, s))
    for y in range(s):
        t = y / max(1, s - 1)
        row = tuple(round(a + (b - a) * t) for a, b in zip(TILE_TOP, TILE_BOTTOM))
        grad.paste((*row, 255), (0, y, s, y + 1))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], radius=round(s * 0.22), fill=255)
    img.paste(grad, mask=mask)
    return img


def _bars(draw: ImageDraw.ImageDraw, cx: float, cy: float, span: float,
          max_h: float, color: tuple, scale: float) -> None:
    """The waveform, centered at (cx, cy), pill-shaped bars."""
    n = len(WAVE)
    bw = span / (n * 1.75)  # bar width; gaps are 0.75*bw
    step = bw * 1.75
    x = cx - (step * (n - 1)) / 2
    for frac in WAVE:
        h = max(bw, max_h * frac)
        draw.rounded_rectangle(
            [x - bw / 2, cy - h / 2, x + bw / 2, cy + h / 2],
            radius=bw / 2, fill=color,
        )
        x += step


def render(size: int) -> Image.Image:
    img = _tile(size)
    draw = ImageDraw.Draw(img)
    s = size * SS

    if size >= 48:
        # cream pill with dark waveform (the app's on-screen indicator)
        pw, ph = s * 0.80, s * 0.36
        x0, y0 = (s - pw) / 2, (s - ph) / 2
        draw.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=ph / 2, fill=CREAM)
        _bars(draw, s / 2, s / 2, pw * 0.72, ph * 0.62, INK, SS)
    else:
        # tiny sizes: cream bars straight on the tile
        _bars(draw, s / 2, s / 2, s * 0.72, s * 0.56, CREAM, SS)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(out_dir, exist_ok=True)

    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = {size: render(size) for size in sizes}

    ico_path = os.path.abspath(os.path.join(out_dir, "whisperflow.ico"))
    images[256].save(ico_path, format="ICO", sizes=[(n, n) for n in sizes],
                     append_images=[images[n] for n in sizes if n != 256])
    png_path = os.path.abspath(os.path.join(out_dir, "whisperflow.png"))
    images[256].save(png_path)
    print(f"wrote {ico_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()

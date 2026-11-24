"""A schematic camera frame of the kitchen scene for the VLM.

The agent under test is a vision-language model, so every trial sends an image
with the task. Without the LIBERO simulator installed we draw a labelled
top-down sketch of the same scene (objects at the positions the lookup tool
describes), with small per-episode jitter. Real simulator frames can be passed
to `run_trial(image=...)` instead; the injection channel is text either way.
"""

from __future__ import annotations

import random

# name -> (x, y, w, h, fill, shape) on a 448x448 canvas; y grows toward the front of the table.
_LAYOUT = {
    "cabinet": (224, 70, 190, 80, (150, 105, 60), "rect"),
    "wine rack": (70, 80, 90, 60, (180, 140, 90), "rect"),
    "stove": (370, 230, 110, 110, (70, 70, 80), "rect"),
    "wine bottle": (70, 240, 34, 34, (20, 90, 40), "circle"),
    "bowl": (224, 230, 64, 64, (25, 25, 25), "circle"),
    "cream cheese": (300, 250, 44, 30, (90, 130, 210), "rect"),
    "plate": (224, 370, 90, 90, (245, 245, 245), "circle"),
}


def render_scene(seed: int = 0, size: int = 448):
    """Return a PIL image of the scene. `seed` jitters the movable objects."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    img = Image.new("RGB", (448, 448), (205, 185, 150))
    draw = ImageDraw.Draw(img)
    draw.rectangle((8, 8, 440, 440), outline=(120, 100, 70), width=4)
    for name, (x, y, w, h, fill, shape) in _LAYOUT.items():
        if name not in ("cabinet", "wine rack", "stove"):
            x, y = x + rng.randint(-14, 14), y + rng.randint(-14, 14)
        box = (x - w // 2, y - h // 2, x + w // 2, y + h // 2)
        if shape == "circle":
            draw.ellipse(box, fill=fill, outline=(0, 0, 0), width=2)
        else:
            draw.rectangle(box, fill=fill, outline=(0, 0, 0), width=2)
        if name == "stove":
            draw.ellipse((x - 30, y - 30, x + 30, y + 30), outline=(200, 60, 40), width=4)
        if name == "cabinet":
            for k in (-1, 0, 1):
                draw.line((x - 80, y + 22 * k, x + 80, y + 22 * k), fill=(90, 60, 30), width=2)
        tw = draw.textlength(name)
        draw.text((x - tw / 2, box[3] + 3), name, fill=(0, 0, 0))
    return img if size == 448 else img.resize((size, size))

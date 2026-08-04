"""Render names into small grayscale images for the cross-modal toy task."""

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


@lru_cache(maxsize=2048)
def render_name(text, height=32, width=96):
    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 20)
    except OSError:
        try: font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except OSError: font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    x, y = max(1, (width - (box[2] - box[0])) // 2), max(1, (height - (box[3] - box[1])) // 2)
    draw.text((x, y), text, fill=255, font=font)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def render_batch(texts, height=32, width=96, device="cpu"):
    return torch.stack([render_name(text, height, width) for text in texts]).to(device)

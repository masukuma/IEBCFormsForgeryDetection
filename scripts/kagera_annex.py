"""Annexure image: the top of the sheet in the Kagera portal original and in both annexed pages.

usage: kagera_annex.py original.png page130.png page131.png mark out.png  (renders at 150 dpi)
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

o, a, b, mark, out = sys.argv[1:6]
W, PAD = 1000, 16
SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def bar(lines, size, bold=False):
    font = ImageFont.truetype(BOLD if bold else SANS, size)
    im = Image.new("RGB", (W, len(lines) * int(size * 1.35) + PAD), "white")
    d = ImageDraw.Draw(im)
    for k, line in enumerate(lines):
        d.text((PAD, PAD // 2 + k * int(size * 1.35)), line, fill="black", font=font)
    return im


def crop(path, box):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    c = im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    c = c.resize((W - 2 * PAD, int(c.size[1] * (W - 2 * PAD) / c.size[0])), Image.LANCZOS)
    row = Image.new("RGB", (W, c.size[1]), "white")
    row.paste(c, (PAD, 0))
    return row


parts = [
    bar([f"Annexure {mark}", "Kagera Primary School polling station 1 of 2", "The top of the sheet in the portal original and in the two annexed pages"], 24, True),
    bar(["Portal original"], 20, True), crop(o, (0, 0, 1, 0.13)),
    bar(['PDF page 130 (annexure page 129), captioned "IEBC PORTAL"'], 20, True), crop(a, (0.03, 0.03, 0.97, 0.16)),
    bar(['PDF page 131 (annexure page 130), captioned "IEBC (INITIAL) PORTAL"'], 20, True), crop(b, (0.03, 0.03, 0.97, 0.16)),
]
img = Image.new("RGB", (W, sum(p.size[1] for p in parts) + PAD), "white")
y = 0
for p in parts:
    img.paste(p, (0, y))
    y += p.size[1]
img.save(out, optimize=True)
print(out, img.size)

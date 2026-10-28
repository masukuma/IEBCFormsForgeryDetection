"""One annexure image per polling station whose annexed pages differ from the original: each
figure that differs, original on the left and annexed page on the right, in the order the
affidavit lists them.

usage: annex_images.py run_dir readings.json out_dir
writes <out_dir>/<case id>.png and index.json {case id: [mark, title, file]}
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from formguard.affidavit import _field_order, _num, _plain_title, annexure_marks, exhibit_facts

run, out = Path(sys.argv[1]), Path(sys.argv[3])
readings = json.load(open(sys.argv[2]))
results = json.load(open(run / "results.json"))
out.mkdir(parents=True, exist_ok=True)
W, PAD = 1000, 16
SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def text_bar(lines, size=22, bold=False, h=None):
    font = ImageFont.truetype(BOLD if bold else SANS, size)
    h = h or (len(lines) * int(size * 1.35) + PAD)
    im = Image.new("RGB", (W, h), "white")
    d = ImageDraw.Draw(im)
    for k, line in enumerate(lines):
        d.text((PAD, PAD // 2 + k * int(size * 1.35)), line, fill="black", font=font)
    return np.array(im)


def pair(tile_path):
    """The pipeline's tile without its label strip, with plain labels over each half."""
    t = cv2.cvtColor(cv2.imread(str(tile_path)), cv2.COLOR_BGR2RGB)[22:]
    half = (t.shape[1] - 8) // 2
    s = min(2.0, (W - 2 * PAD) / t.shape[1], 360 / t.shape[0])
    t = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    labels = Image.new("RGB", (W, 30), "white")
    d = ImageDraw.Draw(labels)
    font = ImageFont.truetype(SANS, 18)
    d.text((PAD, 6), "Original", fill="black", font=font)
    d.text((PAD + int((half + 8) * s), 6), "Annexed page", fill="black", font=font)
    row = np.full((t.shape[0], W, 3), 255, np.uint8)
    row[:, PAD : PAD + t.shape[1]] = t
    return np.vstack([np.array(labels), row])


marks = annexure_marks(results, "[XX]", readings)
index = {}
for c in results["cases"]:
    if c.get("id") not in marks:
        continue
    title = _plain_title(c["title"])
    station, _, place = title.partition(", ")
    parts = [text_bar([f"Annexure {marks[c['id']]}", station, place], size=24, bold=True)]
    for rep in c["exhibit_results"]:
        f = exhibit_facts(c, rep, readings.get(c["id"], {}))
        changed = sorted((d for d in f["differences_in_results"] if not (d["original_reads"] and d["original_reads"] == d["questioned_reads"])), key=_field_order)
        if not changed:
            continue
        cap = f', captioned "{f["caption"]}"' if f["caption"] else ""
        parts.append(text_bar([f"{f['label']}{cap}"], size=21, bold=True))
        for d in changed:
            parts.append(pair(run / rep["evidence_dir"] / d["evidence"]))
            what = (d["field"][0].upper() + d["field"][1:]) if d.get("field") else "Figure"
            reads = f"Original {d['original_reads']}, annexed page {d['questioned_reads']}. " if d["original_reads"] else ""
            parts.append(text_bar([f"{what}. {reads}Similarity {_num(d['similarity'])}."], size=19))
    img = np.vstack(parts + [np.full((PAD, W, 3), 255, np.uint8)])
    name = f"{c['id']}.png"
    Image.fromarray(img).save(out / name, optimize=True)
    index[c["id"]] = [marks[c["id"]], title, name]
json.dump(index, open(out / "index.json", "w"), indent=1)
print(len(index), "annexure images")

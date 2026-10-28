# formguard: detecting altered election forms

Compares submitted copies of IEBC Form 34B (from `Dropbox/Work/Kenya/2022/WhistleBlower`) with the
originals (`Dropbox/Work/Kenya/2022/Forms/34B`). It reports, per document, whether figures in the
results table were changed, which cells, what they read before and after, and side-by-side evidence.

How it works and why: [docs/DESIGN.md](docs/DESIGN.md). In short:

1. **Match pages** with SIFT + RANSAC, so reordered, rotated, duplicated, cropped or re-photographed pages still line up.
2. **Register** with a homography plus heavily smoothed optical flow (paper curl, lens distortion).
3. **Compare glyph by glyph**: flatten both pages, equalise sharpness, remove table ruling, and score
   every digit/mark with contrast-invariant correlation. Unchanged glyphs score about 1.0 across very
   different copies; edited figures score below 0.5.
4. **Explain**: merge flagged glyphs into table cells, OCR only those cells (`26 → 226`), and pair up values swapped between cells.

## Install

```bash
pip install -e .            # numpy, opencv-python-headless, pymupdf
pip install -e '.[dropbox]' # optional: Dropbox sync
sudo apt-get install tesseract-ocr   # optional: read the changed numbers
```

## Use

```bash
# 1. Mirror the folders (incremental: only new or changed files are downloaded)
export DROPBOX_TOKEN=...
python -m formguard sync --remote /Work/Kenya/2022/Forms/34B    --local data/originals
python -m formguard sync --remote /Work/Kenya/2022/WhistleBlower --local data/submissions

# 2. Scan everything (pairs submissions with originals by constituency name)
python -m formguard scan --originals data/originals --submissions data/submissions --out out --workers 8
#   -> out/index.html      summary, worst first, with evidence crops
#   -> out/summary.json    machine-readable
#   -> out/<constituency>/report.json, p<N>_overview.jpg, p<N>_change<K>.png

# Or a single pair
python -m formguard compare data/originals/Bungoma/Kimilili.pdf data/submissions/Kimilili.pdf --out out/kimilili
```

Verdicts: `identical`, `clean`, `altered` (figures in the table changed), `review` (changes outside
the table only), `inconclusive` (page could not be matched, or too degraded to judge),
`no_original`, `error`.

Tuning: `--flag-below` (glyph similarity threshold, default 0.5), `--long-side` (working resolution,
default 2600 px), `--no-ocr`.

## Tests

```bash
pip install -e '.[test]' && pytest
```

The tests build synthetic ruled forms on a textured background, then degrade them (rotation,
scaling, blur, contrast, JPEG) with and without edits. They check that clean copies come back
clean, and that replaced, shortened, prefixed and swapped figures are found with no false alarms
elsewhere. Page matching across reordering, rotation and duplication is covered too.

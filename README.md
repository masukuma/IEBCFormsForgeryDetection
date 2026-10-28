# formguard: detecting altered election forms

Compares submitted copies of IEBC Form 34B (from `Dropbox/Work/Kenya/2022/WhistleBlower`) with the
originals (`Dropbox/Work/Kenya/2022/Forms/34B`). It reports, per document, whether figures in the
results table were changed, which cells, what they read before and after, and side-by-side evidence.

How it works and why: [docs/DESIGN.md](docs/DESIGN.md). In short:

1. **Match pages** with SIFT + RANSAC, so reordered, rotated, duplicated, cropped or re-photographed pages still line up.
2. **Register** with a homography plus a smooth block-matching correction (paper curl, lens
   distortion), comparing only the paper of both pages.
3. **Compare glyph by glyph**: flatten both pages, equalise sharpness tile by tile, remove table
   ruling, and score every digit/mark with contrast-invariant correlation; numbers with a suspect
   glyph are re-scored in digit-sized windows. Unchanged glyphs score about 1.0 across very
   different copies; edited figures score below 0.5. Given other forms of the same type
   (`--template`), faded print on photographed copies is told apart from the entries.
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

## Case work: annexures to affidavits

For forms annexed to an affidavit (photographs of printouts, in any order, with captions), the
harness finds each page's original itself, compares every page, tests whether a page was made from
the same capture as the original, and drafts an affidavit from the measurements:

```bash
# 1. Which original does each annexed page show? (feature retrieval among all candidates)
python -m formguard discover "affidavit.pdf#pages=46-130" --candidates ref/34A --out manifest.json
#    then fill in each case's title, claim and the page captions; the draft also lists a
#    "template": a dozen candidates that are not originals of any case, used to recognise print

# 2. Compare every exhibit with its original; validate the originals' digital signatures
python -m formguard cases manifest.json --out runs/

# 3. Read the flagged figures with Ollama Cloud vision models (needs OLLAMA_API_KEY)
python -m formguard readings runs/results.json --out readings.json [--confirm] [--only figures.json]

# 4. Draft the affidavit (DOCX + Markdown, evidence images annexed)
python -m formguard affidavit runs/results.json --readings readings.json --deponent deponent.json --out affidavit
```

Figures scoring between 0.5 and 0.65 are possible changes: they get evidence images and readings,
and count as differing only when they read differently. Readings are saved after each case; failed
attempts (a dropped connection, malformed output) are retried on the next run. `--only` takes
`{case id: [evidence names]}` and reads just those figures, e.g. the ones an examiner has already
read as numbers, so a model can serve as an independent second reader without spending calls on
stamps and signatures. A model's answer is accepted as the JSON asked for, or as one
"panel: numbers" line per panel, since some models ignore the requested format.

Ollama Cloud models are used cheapest first. Readings start on `gemma4:31b` (the smallest vision
model) and drafting on `gpt-oss:20b`. A larger model is tried only when the smaller one's output
fails validation: an illegible or malformed reading, or a draft containing any number not in the
measured facts. `--confirm` adds an independent second reading from the next model up. The ladders
are set with `FORMGUARD_VISION_MODELS` and `FORMGUARD_TEXT_MODELS`. Models never decide what
differs; they only read figures off evidence images and word paragraphs around measured facts.

Tuning: `--flag-below` (similarity threshold, default 0.5), `--long-side` (working resolution,
default 2600 px), `--no-ocr`, `--template DIR` (other forms of the originals' type; recommended for
photographed copies, where printed labels fade).

## Tests

```bash
pip install -e '.[test]' && pytest
```

The tests build synthetic ruled forms on a textured background, then degrade them (rotation,
scaling, blur, contrast, JPEG) with and without edits. They check that clean copies come back
clean, and that replaced, shortened, prefixed and swapped figures are found with no false alarms
elsewhere. Page matching across reordering, rotation and duplication is covered too.

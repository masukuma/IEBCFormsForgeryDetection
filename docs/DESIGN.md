# Design: detecting alterations in submitted Form 34B copies

## 1. The problem, as the data presents it

| Dropbox folder | What it holds |
|---|---|
| `Work/Kenya/2022/Forms/34B/<County>/<Constituency>.pdf` | **Originals.** IEBC Form 34B (constituency collation of presidential results), scanned in August 2022 on several office scanners (RICOH, Canon MRC-compressed, Konica Minolta). 41 constituencies in 7 counties (Baringo, Bomet, Bungoma, Busia, Elgeyo-Marakwet, Embu, Garissa) plus Diaspora. |
| `Work/Kenya/2022/Forms/34A/1_34_0CC0000000000.zip` | **Originals.** Form 34A (polling-station results) for every county: one zip per county, ~10 GB in total. File names encode the station: `1_34_A_<county 3><constituency 3><ward 4><station 3><stream 2>_…PDF`. |
| `Work/Kenya/2022/WhistleBlower/<Constituency>.pdf` | **Submissions.** 247 Form 34B documents, all created 17 Jan 2023, each page re-rasterised to one JPEG. Some come from the original scans (identical geometry), others are re-photographed or rescaled; some have extra, missing, duplicated or rotated pages. |

29 submissions have an original 34B to compare against. In the sample checked by hand, the alterations are
edits to the **figures** in the results table: digits prefixed to inflate a count (`26 → 226`), values
**swapped** between candidates or polling stations (`189 ↔ 75`), whole candidate columns exchanged,
and digits pasted onto a patch (`9` on a grey box).

What the system has to cope with:

* Copies are never pixel-identical to the original: different scanner, resolution (150–600 dpi),
  JPEG quality, contrast, page rotation, perspective and paper curl.
* The forms are printed on a security background (guilloche frame, microtext), which dominates any
  naive pixel difference.
* The figures that matter are small (a digit is ~25 px tall at working resolution), and an edit can
  change one stroke (`3 → 8`).
* There are hundreds of documents, so the pipeline must run unattended, in parallel, and say
  "can't tell" rather than guess when a copy is too degraded.

## 2. Pipeline

```
Dropbox ──sync (content-hash, incremental)──▶ local mirror
   │
   ├─ pair submissions with originals by constituency name  (pairing.py)
   │      └─ byte-identical? → IDENTICAL, stop
   │
   ├─ render every page at a common working size              (render.py)
   ├─ match pages: SIFT + RANSAC homography, each submitted
   │  page against every original page                        (register.py)
   ├─ register: homography warp + heavily smoothed optical flow
   ├─ compare glyphs: normalise, match sharpness, drop ruling,
   │  ZNCC per glyph                                          (compare.py)
   ├─ interpret: merge glyphs into per-cell regions, OCR the
   │  flagged cells only, detect swapped values               (pipeline.py, ocr.py)
   └─ report: JSON per document + HTML summary with
      side-by-side evidence crops                             (report.py)
```

### 2.1 Page matching and registration
Each submitted page is matched independently against every page of the original with SIFT
keypoints (rotation- and scale-invariant) and a RANSAC homography. Matching per page instead of
assuming the same page order handles reordering, duplicates (Tongaren's submission contains
every page twice), 90° rotations (Lagdera, Teso South) and pages with no original
(Manyatta pp. 5–6). A page needs at least 150 RANSAC inliers. A wrong pairing or unrelated page then
reads as "no matching page", never as a false alteration.

The homography removes scale, rotation, crop and perspective. What is left (paper curl, lens
distortion, up to ~25 px on phone photos) is absorbed by dense optical flow (DIS) **smoothed with
a σ = 25 px Gaussian**. The smoothing matters: it lets the flow follow gentle paper deformation but
not a single edited digit, which is a sharp, local change.

### 2.2 Glyph comparison (the core)
Pixel differences fail on these forms, and so do thresholded ink differences: strict enough to
see `3 → 8` means flagging every faint stroke of a low-resolution copy; tolerant enough to ignore
blur means a different number in the same cell slips through. So pages are compared glyph by glyph:

1. **Flatten.** Divide by a local background estimate (morphological closing). Paper tone,
   shading and scanner contrast go away; 1.0 = paper, 0.0 = ink.
2. **Match sharpness.** Blur whichever page is sharper until it best matches the other (σ chosen
   on the table area), so a 150 dpi copy is not "missing" detail that a 300 dpi original has.
3. **Remove ruling, find glyphs.** Threshold ink on *both* pages, remove table lines
   (morphological opening with long line kernels), and take connected components of the union.
   A digit present on only one side is still a glyph.
4. **Score each glyph** with zero-mean normalised cross-correlation (ZNCC) between the two pages,
   maximised over a ±2 px shift. ZNCC is invariant to brightness and contrast. Before scoring,
   everything lighter than faint ink is clipped to paper. Without that, the shared security
   background behind two *different* numbers makes them look alike (a swapped `317`/`683` scored
   0.54; it scores well below 0.5 with clipping).
5. **Group** glyphs into numbers, cut along the table ruling so a group never spans two cells.
   Any number containing a low-scoring glyph becomes a region covering the whole number, classified
   as *substitution*, *addition* or *erasure* by the ink on each side. A figure that was re-typed with
   the same value (Mbalambala: `189` moved from right-aligned to centred) is still caught, because its
   glyphs moved. OCR then marks it as "value unchanged".
6. **Locate the results table** from line intersections. Changes inside it are alterations;
   changes elsewhere (headers, stamps, signatures) are reported for review.

### 2.3 Verdicts

| Verdict | Meaning |
|---|---|
| `identical` | byte-identical to the original |
| `clean` | every page matched and no glyph fell below the similarity threshold (0.5) |
| `altered` | at least one figure in the results table differs from the original |
| `review` | differences only outside the table (header, stamp, signature area) |
| `inconclusive` | a page could not be matched to the original, or the copy is too degraded to judge (median glyph similarity < 0.85) |
| `no_original` | no original available to compare against (see §4) |

### 2.4 Interpreting what changed
Only the flagged cells are sent to OCR (tesseract, digits only, trying the three plausible
orientations because many tables are printed sideways). That keeps OCR cost proportional to the
number of edits, not to the size of the form. Detection never depends on OCR; it only annotates
the evidence (`original 26 → submission 226`). With readings available, the report also pairs up
**swaps**: cell A shows B's value and B shows A's. That pattern means votes were moved between
candidates or stations rather than mis-transcribed.

### 2.5 Efficiency
* **Incremental ingest:** the Dropbox mirror skips files whose local Dropbox content hash already
  matches, so only new or changed submissions are downloaded.
* **Short-circuits:** byte-identical files are settled by hash, and submissions with no original are not rendered.
* **Bounded work per page:** keypoints are computed at 1200 px. Glyph comparison runs at 2600 px
  (~220 dpi on A3) using one `matchTemplate` call per glyph, roughly 1 s for a page of 2–4 k glyphs.
  OCR runs only on flagged cells.
* **Parallel by document** (`--workers`), with no shared state. A batch scales linearly with cores
  and a failing file is reported as `error` without stopping the batch.

## 3. Results on the available pairs
_(filled in from the scan; see README)_

## 4. Limits and next steps

* **Most submissions have no 34B original.** Only 29 of 247 have one. For the rest, the
  reference has to come from the **Form 34A originals**, because each 34B row is a polling station's
  34A result. Proposed tier: index the 34A scans by station code (already in the file names),
  read the station code and figures from each 34B row (table OCR: the table geometry is already
  extracted here), and reconcile them against the matching 34As. Reading the handwritten 34A
  figures is the hard part. Reconcile first against IEBC's published 34A tallies if a
  machine-readable copy is available, and route mismatches to human review with both images side
  by side.
* **Arithmetic consistency** (row sums equal total valid votes; votes ≤ registered voters) is a
  cheap no-reference check once rows are OCR'd. It catches inflation but not swaps, which preserve
  the sums.
* **Single-document forensics** (copy-move, double JPEG compression) were evaluated and rejected
  as a primary signal: the printed digits are pixel-identical by design, and the background
  microtext is regular enough that clean scans give the same "clone" scores as tampered ones.
* Thresholds (similarity 0.5, quality 0.85) are calibrated on these 29 pairs. Re-check them if a very
  different source of copies appears (e.g. fax). They are CLI flags.

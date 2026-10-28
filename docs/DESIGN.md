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
   ├─ register: homography warp + smooth block-matching correction,
   │  paper masks for both pages
   ├─ printed form (optional): median of other forms of the
   │  same type, to tell print from entries                    (template.py)
   ├─ compare glyphs: normalise, match sharpness per tile, drop
   │  ruling, ZNCC per glyph; re-score suspect numbers in
   │  digit-sized windows                                      (compare.py)
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
distortion) is corrected by **block matching**: 192 px blocks of the page are located in the
registered copy by correlating edge maps, unreliable blocks are dropped, outliers removed by a
median over neighbouring blocks, and the field is smoothed over about half a block spacing and
extended to the page edges. The correction is kept only if it improves the pages' overall edge
agreement. It follows the curl of a sheet, typically 1.5–5 px, but cannot move one figure: a block
is several times the size of a digit. (Dense optical flow was tried first and drifted by hundreds of
pixels on photographs of printouts.) Each glyph is then re-aligned by up to ±6 px using an 80 px
window of its surroundings, again too large for an edited figure to pull the alignment towards itself.

Only the **paper** is compared. On photographs, and on the KIEMS kit's own images of the forms,
the desk or scanner background shows around the sheet; areas much darker than the paper
(< 50 % of its brightness) that reach the image edge are masked on both pages.

### 2.2 Glyph comparison (the core)
Pixel differences fail on these forms, and so do thresholded ink differences: strict enough to
see `3 → 8` means flagging every faint stroke of a low-resolution copy; tolerant enough to ignore
blur means a different number in the same cell slips through. So pages are compared glyph by glyph:

1. **Flatten.** Divide by a local background estimate (morphological closing). Paper tone,
   shading and scanner contrast go away; 1.0 = paper, 0.0 = ink.
2. **Match sharpness.** Blur whichever page is sharper until it best matches the other, so a
   150 dpi copy is not "missing" detail that a 300 dpi original has. The level is chosen per
   192 px tile (median over neighbouring tiles), because a phone photograph is not equally sharp
   across the sheet.
3. **Remove ruling, find glyphs.** Threshold ink on *both* pages, remove table lines
   (morphological opening with long line kernels), and take connected components of the union.
   A digit present on only one side is still a glyph.
4. **Score each glyph** with zero-mean normalised cross-correlation (ZNCC) between the two pages,
   maximised over a ±2 px shift. ZNCC is invariant to brightness and contrast. Before scoring,
   everything lighter than faint ink is clipped to paper. Without that, the shared security
   background behind two *different* numbers makes them look alike (a swapped `317`/`683` scored
   0.54; it scores well below 0.5 with clipping).
5. **Group** glyphs into numbers, cut along the table ruling so a group never spans two cells.
   A number containing a low-scoring glyph is **re-scored in digit-sized windows** (about a digit's
   height, slid a quarter-window at a time across the number). How a number's strokes split into
   glyphs varies between copies: in `063 → 033` the `0` and `6` touch and form one glyph, which
   scores 0.64 against `03`, while a window over the middle digit scores below 0. The lowest
   window gives the region's similarity. Regions are classified as *substitution*, *addition* or
   *erasure* by the ink on each side, and as *on an entry* of the original or not. A figure that was
   re-typed with the same value (Mbalambala: `189` moved from right-aligned to centred) is still
   caught, because its glyphs moved. OCR then marks it as "value unchanged".
6. **Locate the results table** from its long horizontal rules, found at any slant with a Hough
   transform (line intersections as a fallback). Changes to entries inside it are alterations;
   changes elsewhere (headers, stamps, signatures) are reported for review.

### 2.2.1 Photographs of printouts
Annexures to affidavits are photographs of printed copies, often printed again into the court
bundle. Three things on such copies look like changes but are not, and are handled explicitly:

* **Faded print.** The grey printed labels of the form fade into the paper while pen entries stay
  dark. With a pool of other forms of the same type (`--template`, or `"template"` in a manifest),
  the **printed form** is the pixel-wise median of those forms registered onto the original: what
  most of them share is print, their entries differ and drop out. A reference glyph that matches the
  printed form (ZNCC ≥ 0.6; handwritten entries score below 0.4 against it) and has gained no ink
  is scored but never reported.
* **Faint but present.** Where a mark seems to be missing from the copy, the copy's unclipped
  darkness is correlated with the original's stroke shape. Print or writing that is merely lighter
  follows the strokes; erased or blank paper, and the security background, do not.
* **Marks added to the bundle.** Exhibit stamps, page numbers and captions are added ink. Large
  added marks, and bold type far thicker than a pen line, cover what is under them: those areas
  are reported as *obscured*, not altered. Added marks on blank paper are reported as *additions
  on blank paper*, separately from changes to entries.

Long straight strokes (cross-out lines through unused table space, broken rules) are treated like
ruling. Calibration on six photographed pages of three polling stations (each original against
a portal printout and an agent copy): every changed figure scored between −0.11 and 0.44, and every
area of a clean copy 0.54 or more, against a threshold of 0.5.

### 2.3 Verdicts

| Verdict | Meaning |
|---|---|
| `identical` | byte-identical to the original |
| `clean` | every page matched and nothing fell below the similarity threshold (0.5) |
| `altered` | at least one entry in the results table differs from the original |
| `review` | differences only outside the table (header, stamp, signature area), marks added on blank paper, or areas obscured by added marks |
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

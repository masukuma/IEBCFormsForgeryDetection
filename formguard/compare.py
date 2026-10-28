"""Glyph-level change detection between a reference page and a registered submission page.

The forms are printed on a security background (guilloche frame, microtext) and the copies
we receive differ in scanner, resolution, JPEG quality and contrast. A raw pixel diff is
dominated by those differences, and even a thresholded "ink" diff either drowns in noise or,
made tolerant enough, stops seeing a "3" turned into an "8". So we compare *glyphs*:

1. Both pages are flattened (divided by a local background estimate) and the sharper one is
   blurred down to the other's level, so contrast, paper tone and resolution stop mattering.
2. Ink is thresholded, table ruling is removed, and what remains is split into glyphs
   (connected blobs: digits, letters, marks) from *either* page, so added and removed marks
   both show up.
3. Each glyph box is scored by zero-mean normalised cross-correlation (ZNCC) between the two
   pages, maximised over a +/-2 px shift. ZNCC ignores brightness and contrast; an unchanged
   glyph scores ~0.95-1.0 even across very different copies, a changed digit scores < 0.5.
4. Glyphs are grouped into numbers; a number with a low-scoring glyph is re-scored in
   digit-sized windows, so the result does not depend on how its strokes were split into glyphs.
   Low-scoring numbers become regions, reported with the kind of change, whether they fall
   inside the results table and whether they sit on an entry of the original.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

INK = 0.55  # normalised intensity below which a pixel is ink, on a crisp copy (see ink_threshold)
INK_MAX = 0.7  # never above this: lighter marks are the printed security background
PAPER = 0.75  # ... above which it is treated as blank paper when scoring glyphs
BLUR_LEVELS = (0.0, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
SHIFT = 2  # residual misalignment absorbed when scoring a glyph, px
LOCAL = 6  # largest local correction of the registration, px (see local_shift)
CONTEXT = 40  # surroundings used to measure it, px
BLUR_TILE = 192  # px; sharpness is matched per tile of this size (see blur_map)
MIN_GLYPH_INK = 25  # px; smaller blobs are specks/noise
MAX_GLYPH = 150  # px; bigger blobs are stamps, signatures, logos
MAX_MARK = 700  # px; large marks up to this size are scored separately (not flagged)
BOLD = 6  # px; ink this far inside a stroke (12 px wide) is bold type, not a pen line
STROKE_LEN = 70  # px; straight strokes this long are ruling or cross-outs, not figures (see _is_stroke)
PRINT_SIM = 0.6  # a reference glyph matching the printed form this well is print (template.py);
# handwritten entries score below 0.4 against it, faded print on a photograph 0.6-1.0


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    similarity: float  # lowest glyph ZNCC in the region (1 = identical)
    ink_ref: int  # ink pixels in the reference
    ink_sub: int  # ink pixels in the submission
    kind: str  # substitution | addition | erasure | obscured (under a large mark added to the copy)
    in_table: bool
    glyphs: int
    on_figure: bool = True  # part of an entry on the original (not blank paper, not its print)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageComparison:
    regions: list[Region]
    quality: float  # median glyph similarity: how comparable the two copies are at all
    glyphs: int
    blur_sigma: float  # + blurred the reference, - blurred the submission
    table_roi: tuple[int, int, int, int] | None
    similarities: np.ndarray = field(default_factory=lambda: np.zeros(0))  # every scored glyph
    large_marks: list[dict] = field(default_factory=list)  # stamps, signatures: position + similarity
    printed: int = 0  # glyphs set aside as the printed form (see template.py)

    def flagged(self, threshold: float, table_only: bool = True) -> list[Region]:
        return [r for r in self.regions if r.similarity < threshold and (r.in_table or not table_only)]


def normalize(img: np.ndarray) -> np.ndarray:
    """Divide out the local paper background: 1.0 = paper, 0.0 = black ink."""
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 5)
    return np.clip(img.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0), 0.0, 1.0)


def stretch(n: np.ndarray) -> np.ndarray:
    """Stretch a normalised page so its darkest strokes reach black and its paper stays white.

    Photographs of printouts can be pale: pen strokes that only reach 0.8 of the paper's
    brightness. Levels are measured on the central part of the page, away from desk or shadow
    at the edges of a photograph.
    """
    h, w = n.shape
    centre = n[h // 10 : h - h // 10, w // 10 : w - w // 10]
    dark, paper = np.percentile(centre, [0.5, 50])
    if paper - dark < 0.05:
        return n
    return np.clip((n - dark) / (paper - dark), 0.0, 1.0).astype(np.float32)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img


def match_sharpness(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Blur whichever image is sharper until it best matches the other.

    Estimated on the central part of the page, where the table is.
    """
    h, w = a.shape
    cy, cx, r = h // 2, w // 2, min(h, w) // 4
    ca = a[cy - r : cy + r, cx - r : cx + r].astype(np.float32)
    cb = b[cy - r : cy + r, cx - r : cx + r].astype(np.float32)
    best = (np.inf, 0.0)
    for s in BLUR_LEVELS:
        best = min(best, (float(np.abs(_blur(ca, s) - cb).mean()), s))
        if s:
            best = min(best, (float(np.abs(ca - _blur(cb, s)).mean()), -s))
    s = best[1]
    if s > 0:
        return _blur(a, s), b, s
    if s < 0:
        return a, _blur(b, -s), s
    return a, b, 0.0


def blur_map(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None, tile: int = BLUR_TILE) -> np.ndarray:
    """Signed blur (as in match_sharpness) that best matches the two pages, tile by tile.

    A phone photograph is not equally sharp everywhere: focus falls off towards the edges and
    with the curl of the sheet, so one blur level for the whole page leaves parts of it sharper
    on one side, and their print scores low for no reason. Each tile holding enough strokes gets
    its own level; the rest take the page's level. A median over neighbouring tiles keeps one
    edited figure from setting its tile's level.
    """
    H, W = a.shape
    rows, cols = -(-H // tile), -(-W // tile)
    grid = np.full((rows, cols), np.nan, np.float32)
    for i in range(rows):
        for j in range(cols):
            ys, xs = slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile)
            ca, cb = a[ys, xs], b[ys, xs]
            if (valid is not None and valid[ys, xs].mean() < 0.5) or (ca < INK).mean() < 0.01:
                continue
            best = (np.inf, 0.0)
            for s in BLUR_LEVELS:
                best = min(best, (float(np.abs(_blur(ca, s) - cb).mean()), s))
                if s:
                    best = min(best, (float(np.abs(ca - _blur(cb, s)).mean()), -s))
            grid[i, j] = best[1]
    if np.isnan(grid).all():
        return np.zeros((rows, cols), np.float32)
    fill = float(np.nanmedian(grid))
    out = np.empty_like(grid)
    for i in range(rows):
        for j in range(cols):
            near = grid[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2]
            out[i, j] = float(np.nanmedian(near)) if not np.isnan(near).all() else fill
    return out


def ink_threshold(n: np.ndarray) -> float:
    """Ink threshold adapted to how dark the strokes on this copy are.

    Copies whose strokes reach solid black (0.5th percentile <= 0.3) use the standard threshold.
    On a blurred photograph of a printout, thin pen strokes only reach ~0.45, so the threshold
    moves to halfway between them and the paper, so the handwriting is not lost. Capped below
    the printed microtext.
    """
    dark, paper = np.percentile(n, [0.5, 50])
    if dark <= 0.3:
        return INK
    return float(np.clip((dark + paper) / 2, INK, INK_MAX))


def ruling(ink: np.ndarray, length: int = 41) -> np.ndarray:
    """Horizontal and vertical line segments (table ruling, underlines)."""
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    return h | v


def _hough_rules(ink: np.ndarray) -> np.ndarray | None:
    """Long, near-horizontal rules as line segments (x1, y1, x2, y2), whatever their slant.

    Photographed forms are rotated a few degrees and their rules are broken by the microtext
    background; a Hough transform finds them where a straight morphological kernel cannot.
    """
    w = ink.shape[1]
    segs = cv2.HoughLinesP(ink * 255, 1, np.pi / 720, threshold=max(w // 12, 60), minLineLength=w // 8, maxLineGap=15)
    if segs is None:
        return None
    segs = segs.reshape(-1, 4)  # (N, 1, 4) in OpenCV 4, (N, 4) in OpenCV 5
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    keep = np.abs(dy) <= 0.18 * np.abs(dx)  # within ~10 degrees of horizontal
    keep &= np.array([_thin(ink, sg) for sg in segs], bool)
    return segs[keep] if keep.any() else None


def _thin(ink: np.ndarray, seg: np.ndarray, gap: int = 12) -> bool:
    """A rule is a thin, solid line with paper on both sides. A line of bold text is broken and
    has ink above and below it; the edge of a dark band or of the sheet has ink on one side."""
    h, w = ink.shape
    x1, y1, x2, y2 = (int(v) for v in seg)
    t = np.linspace(0.0, 1.0, max(2, max(abs(x2 - x1), abs(y2 - y1)) // 3))
    xs = np.clip(np.round(x1 + (x2 - x1) * t).astype(int), 0, w - 1)
    ys = np.round(y1 + (y2 - y1) * t).astype(int)
    on = cv2.dilate(ink, np.ones((5, 5), np.uint8))[np.clip(ys, 0, h - 1), xs].mean()
    above = ink[np.clip(ys - gap, 0, h - 1), xs].mean()
    below = ink[np.clip(ys + gap, 0, h - 1), xs].mean()
    return on >= 0.6 and above < 0.35 and below < 0.35


def long_rules(ink: np.ndarray) -> np.ndarray:
    """Mask of horizontal rules at least an eighth of the page wide, even if broken or slanted.

    Too long to be confused with a handwritten number, so safe to use for cutting rows apart.
    """
    mask = np.zeros_like(ink)
    segs = _hough_rules(ink)
    if segs is not None:
        for x1, y1, x2, y2 in segs:
            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, 3)
    return mask & cv2.dilate(ink, np.ones((3, 3), np.uint8))


def sideways(img: np.ndarray) -> bool:
    """Whether a page's table is printed sideways: its long rules run down the page, not across."""
    ink = (stretch(normalize(img)) < 0.5).astype(np.uint8)
    across, down = _hough_rules(ink), _hough_rules(np.ascontiguousarray(ink.T))
    n_across, n_down = (0 if r is None else len(r) for r in (across, down))
    return n_down > 1.5 * max(n_across, 1)


def table_roi(ink: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) of the ruled body of the form.

    Primarily the extent of its long horizontal rules (found at any slant), which covers both
    grid tables (Form 34B) and forms ruled mostly with horizontal lines (Form 34A), so captions,
    page numbers and other annotations in the margins stay outside. Grid intersections are the
    fallback when too few long rules are found.
    """
    h, w = ink.shape
    m = int(0.01 * max(h, w))
    segs = _hough_rules(ink)
    if segs is not None and len(segs) >= 8:
        x0 = int(np.percentile(np.minimum(segs[:, 0], segs[:, 2]), 2))
        x1 = int(np.percentile(np.maximum(segs[:, 0], segs[:, 2]), 98))
        y0, y1 = np.percentile((segs[:, 1] + segs[:, 3]) / 2, [1, 99])
        return x0 - m, int(y0) - m, x1 + m, int(y1) + m
    # Short kernels so slightly skewed scans still yield line segments.
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(w // 30, 15)), np.uint8))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(h // 30, 15), 1), np.uint8))
    k = np.ones((5, 5), np.uint8)
    n, _, _, cent = cv2.connectedComponentsWithStats(cv2.dilate(hl, k) & cv2.dilate(vl, k))
    if n < 20:  # not a ruled form
        return None
    pts = cent[1:]
    # Percentiles drop the few intersections of the page frame with stray lines.
    x0, x1 = np.percentile(pts[:, 0], [1, 99])
    y0, y1 = np.percentile(pts[:, 1], [1, 99])
    return int(x0) - m, int(y0) - m, int(x1) + m, int(y1) + m


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Max ZNCC of the reference glyph (minus a SHIFT border) slid over the submission box."""
    tpl = a[SHIFT:-SHIFT, SHIFT:-SHIFT]
    if tpl.std() < 1e-3 and b.std() < 1e-3:
        return 1.0  # both blank
    if tpl.std() < 1e-3 or b.std() < 1e-3:
        return 0.0  # mark on one side only
    return float(cv2.matchTemplate(b, tpl, cv2.TM_CCOEFF_NORMED).max())


def _faint(strokes: np.ndarray, b: np.ndarray) -> float:
    """How closely the darkness of submission crop ``b`` follows the reference's stroke mask.

    Max ZNCC over a +/-SHIFT slide, on an unclipped crop: a mark that is still there but fainter
    (print fading on a photograph of a printout) follows the strokes; erased or blank paper,
    and the security background, do not.
    """
    tpl = strokes[SHIFT:-SHIFT, SHIFT:-SHIFT].astype(np.float32)
    if tpl.std() < 1e-3 or b.std() < 1e-3:
        return 0.0
    return float(cv2.matchTemplate((1.0 - b).astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED).max())


def _is_stroke(mask: np.ndarray) -> bool:
    """A single straight stroke longer than any digit: a rule, an underline or a cross-out line."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return False
    lo, hi = np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float64)))
    return np.sqrt(max(lo, 0.0)) < 2.5 and 4 * np.sqrt(hi) >= STROKE_LEN


def _starts(a: int, b: int, size: int) -> list[int]:
    """Starts of windows of ``size`` whose centres run from a to b, a quarter window apart.

    A digit's strokes may reach beyond the glyphs that scored low, so windows are centred on,
    not fitted inside, the number's extent.
    """
    n = max(1, int(np.ceil((b - a) / max(1, size // 4))) + 1)
    return [int(round(c)) - size // 2 for c in np.linspace(a, b, n)]


def local_shift(sa: np.ndarray, sb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int]:
    """Residual misalignment around a glyph, measured on its surroundings.

    A window of CONTEXT px around the glyph (neighbouring figures, printed rules and text) is
    located in the submission within +/-LOCAL px. The window is several times the glyph's area,
    so an edited figure inside it cannot pull the alignment towards itself; if the surroundings
    are blank or do not match, no shift is applied.
    """
    H, W = sa.shape
    ya, yb = max(LOCAL, y0 - CONTEXT), min(H - LOCAL, y1 + CONTEXT)
    xa, xb = max(LOCAL, x0 - CONTEXT), min(W - LOCAL, x1 + CONTEXT)
    tpl = sa[ya:yb, xa:xb]
    if tpl.size == 0 or tpl.std() < 0.02:
        return 0, 0
    r = cv2.matchTemplate(sb[ya - LOCAL : yb + LOCAL, xa - LOCAL : xb + LOCAL], tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (px, py) = cv2.minMaxLoc(r)
    return (px - LOCAL, py - LOCAL) if peak >= 0.6 else (0, 0)


def compare_pages(
    ref: np.ndarray,
    sub: np.ndarray,
    valid: np.ndarray | None = None,
    report_below: float = 0.8,
    template: np.ndarray | None = None,
) -> PageComparison:
    """Score every glyph on the page; keep regions whose similarity is below ``report_below``.

    ``template`` is the printed form aligned on ``ref`` (template.build_template). Reference
    glyphs that match it are print; unless the submission adds ink over them they are scored but
    never reported: on a photograph of a printout, faded print is a property of the copy, not
    an alteration.
    """
    ref_s, sub_s, sigma = match_sharpness(ref, sub)
    na, nb = stretch(normalize(ref_s)), stretch(normalize(sub_s))
    ink_a = (na < ink_threshold(na)).astype(np.uint8)
    ink_b = (nb < ink_threshold(nb)).astype(np.uint8)
    both = ink_a | ink_b

    lines = cv2.dilate(ruling(both) | long_rules(both), np.ones((3, 3), np.uint8))
    glyph = both & (1 - lines)
    inside = None
    if valid is not None:  # ignore the margin of the warped submission
        inside = cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8))
        glyph &= inside
    glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Score stroke shape only: clip everything lighter than faint ink to paper, so the shared
    # security background behind two different numbers does not make them look alike. A light
    # blur then averages out halftone dots and JPEG ringing.
    sa = cv2.GaussianBlur(np.minimum(na, PAPER), (0, 0), 1.0)
    sb = cv2.GaussianBlur(np.minimum(nb, PAPER), (0, 0), 1.0)
    # For the scores themselves, sharpness is matched locally (blur_map), on unblurred copies.
    fa, fb = stretch(normalize(ref)), stretch(normalize(sub))
    ca, cb = np.minimum(fa, PAPER), np.minimum(fb, PAPER)
    smap = blur_map(ca, cb, valid)
    P = 16  # crop padding, enough for the widest blur
    ca, cb = (cv2.copyMakeBorder(c, P, P, P, P, cv2.BORDER_REPLICATE) for c in (ca, cb))

    H, W = ref.shape

    def measure(x0: int, y0: int, x1: int, y1: int, dx: int = 0, dy: int = 0) -> tuple[float, int, int]:
        """(similarity, reference ink, submission ink) of one box; the submission side shifted."""
        if not (0 <= y0 + dy and y1 + dy <= H and 0 <= x0 + dx and x1 + dx <= W):
            dx = dy = 0
        ka, kb = int(ink_a[y0:y1, x0:x1].sum()), int(ink_b[y0 + dy : y1 + dy, x0 + dx : x1 + dx].sum())
        s = float(smap[min((y0 + y1) // 2 // BLUR_TILE, smap.shape[0] - 1), min((x0 + x1) // 2 // BLUR_TILE, smap.shape[1] - 1)])
        A = ca[y0 : y1 + 2 * P, x0 : x1 + 2 * P]
        B = cb[y0 + dy : y1 + dy + 2 * P, x0 + dx : x1 + dx + 2 * P]
        A, B = (_blur(A, s), B) if s > 0 else (A, _blur(B, -s))
        A, B = cv2.GaussianBlur(A, (0, 0), 1.0), cv2.GaussianBlur(B, (0, 0), 1.0)
        sim = _similarity(A[P:-P, P:-P], B[P:-P, P:-P])
        if kb < 0.25 * ka:  # gone from the submission, or only fainter?
            strokes = cv2.dilate(ink_a[y0:y1, x0:x1], np.ones((3, 3), np.uint8))
            sim = max(sim, _faint(strokes, cv2.GaussianBlur(fb[y0 + dy : y1 + dy, x0 + dx : x1 + dx], (0, 0), 1.5)))
        return sim, ka, kb
    st = cv2.GaussianBlur(np.minimum(template, PAPER), (0, 0), 1.0) if template is not None else None
    ref_ink = (fa < ink_threshold(fa)).astype(np.uint8)
    roi = table_roi(ref_ink & inside if inside is not None else ref_ink)  # not the edge of the sheet
    m = SHIFT + 3

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(glyph, np.ones((3, 3), np.uint8)))
    scored = []  # (x, y, w, h, sim, ink_a, ink_b, in_table, printed, entry)
    large = []
    for i in range(1, n):
        x, y, w, h, _ = (int(v) for v in stats[i])
        if max(w, h) < 12 or x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue  # specks, and marks cut off by the frame edge (only partly comparable)
        if w > MAX_GLYPH or h > MAX_GLYPH:
            if w <= MAX_MARK and h <= MAX_MARK:
                y0, y1, x0, x1 = max(0, y - m), min(H, y + h + m), max(0, x - m), min(W, x + w + m)
                large.append({"x": x, "y": y, "w": w, "h": h, "similarity": round(measure(x0, y0, x1, y1)[0], 3)})
            continue
        if max(w, h) >= STROKE_LEN and _is_stroke(labels[y : y + h, x : x + w] == i):
            continue  # treated like ruling
        y0, y1, x0, x1 = max(0, y - m), min(H, y + h + m), max(0, x - m), min(W, x + w + m)
        if max(int(ink_a[y0:y1, x0:x1].sum()), int(ink_b[y0:y1, x0:x1].sum())) < MIN_GLYPH_INK:
            continue
        sim, ka, kb = measure(x0, y0, x1, y1, *local_shift(sa, sb, x, y, x + w, y + h))
        in_table = roi is not None and roi[0] <= x + w / 2 <= roi[2] and roi[1] <= y + h / 2 <= roi[3]
        is_print = st is not None and ka >= MIN_GLYPH_INK and _similarity(sa[y0:y1, x0:x1], st[y0:y1, x0:x1]) >= PRINT_SIM
        printed = is_print and kb <= 1.5 * ka  # print with nothing written over it
        entry = ka >= MIN_GLYPH_INK and ka >= 0.25 * kb and not is_print  # something written on the original
        scored.append((x, y, w, h, sim, ka, kb, in_table, printed, entry))

    n_printed = sum(1 for s in scored if s[8])
    entries = [s for s in scored if not s[8]]
    # Quality and the provenance statistics describe the entries; all glyphs if too few remain.
    basis = entries if len(entries) >= 20 else scored
    sims = np.array([s[4] for s in basis]) if basis else np.array([1.0])
    quality = float(np.median(sims))

    # Group glyphs that sit next to each other into numbers, never across table ruling, so each
    # group stays within one cell. A number with a low-scoring glyph is then re-scored in
    # digit-sized windows slid across it: how its strokes were split into glyphs (by ruling, or
    # gaps in a faint pen line) varies between copies, a digit-sized window does not. A region is
    # a number with a low-scoring window; its box covers the whole number, to be read and shown.
    canvas = np.zeros((H, W), np.uint8)
    for x, y, w, h, *_ in entries:
        canvas[y : y + h, x : x + w] = 1
    canvas = cv2.dilate(canvas, np.ones((9, 9), np.uint8)) & (1 - lines)
    _, labels = cv2.connectedComponents(canvas)
    groups: dict[int, list] = {}
    for k, s in enumerate(entries):
        x, y, w, h = s[:4]
        lab = labels[y : y + h, x : x + w]
        lab = lab[lab > 0]  # the glyph's centre may sit on removed ruling
        groups.setdefault(int(np.bincount(lab).argmax()) if lab.size else -1 - k, []).append(s)
    heights = [s[3] for s in entries if s[9]]  # entries: digits, letters and their fragments
    wh = int(np.clip(1.1 * np.percentile(heights, 80), 32, 90)) if heights else 48  # ~ a whole digit
    ww = int(0.8 * wh)
    # Large marks added to the copy (exhibit stamps, page numbers, captions) hide what is under
    # them: that is reported as obscured, not altered.
    added = cv2.dilate(ink_b & (1 - cv2.dilate(ink_a, np.ones((5, 5), np.uint8))), np.ones((5, 5), np.uint8))
    k, alab, astats, _ = cv2.connectedComponentsWithStats(added)
    big = [i for i in range(1, k) if max(astats[i, 2], astats[i, 3]) > 2.5 * wh]
    cover = np.isin(alab, big)
    # Bold type only the copy has (a page number printed over the form) is broken up by what it
    # covers, so it is found by its stroke width instead: far thicker than a pen line.
    bold = (cv2.distanceTransform(ink_b, cv2.DIST_L2, 3) >= BOLD) & (cv2.distanceTransform(ink_a, cv2.DIST_L2, 3) < BOLD / 2)
    cover |= cv2.dilate(bold.astype(np.uint8), np.ones((4 * BOLD + 1, 4 * BOLD + 1), np.uint8)) > 0
    regions = []
    for group in groups.values():
        merged = any(s[3] > 1.6 * wh or s[2] > 3 * ww for s in group)  # figures that touch form one glyph
        if all(s[4] >= report_below for s in group) and not merged:
            continue  # a changed digit inside a merged glyph barely lowers its score: always window it
        x, y = min(s[0] for s in group), min(s[1] for s in group)
        w, h = max(s[0] + s[2] for s in group) - x, max(s[1] + s[3] for s in group) - y
        shift = local_shift(sa, sb, x, y, x + w, y + h)
        members = []  # low-scoring windows: (similarity, ink_a, ink_b)
        for wy in _starts(y, y + h, wh):
            for wx in _starts(x, x + w, ww):
                box = (max(0, wx), max(0, wy), min(W, wx + ww), min(H, wy + wh))
                if inside is not None and inside[box[1] : box[3], box[0] : box[2]].mean() < 0.99:
                    continue  # reaches beyond the comparable part of the page
                if int(glyph[box[1] + SHIFT : box[3] - SHIFT, box[0] + SHIFT : box[2] - SHIFT].sum()) < MIN_GLYPH_INK:
                    continue  # only ruling, or ink at the edge that a neighbouring window centres on
                sim, ka, kb = measure(*box, *shift)
                if max(ka, kb) >= MIN_GLYPH_INK and sim < report_below:
                    members.append((sim, ka, kb))
        if not members:
            continue
        worst = min(members)[0]
        ka, kb = sum(s[5] for s in group), sum(s[6] for s in group)  # the whole number: 002 -> 001 is a substitution
        if cover[y : y + h, x : x + w].mean() > 0.1:
            kind = "obscured"
        elif ka < 0.25 * kb:
            kind = "addition"
        elif kb < 0.25 * ka:
            kind = "erasure"
        else:
            kind = "substitution"
        regions.append(
            Region(
                x, y, w, h,
                similarity=round(worst, 3),
                ink_ref=ka, ink_sub=kb, kind=kind,
                in_table=any(s[7] for s in group),
                glyphs=len(group),
                on_figure=kind != "obscured" and any(s[9] for s in group),  # an entry, not blank paper or print
            )
        )
    regions.sort(key=lambda r: (not r.in_table, r.similarity))
    return PageComparison(regions, round(quality, 4), len(scored), sigma, roi, sims if basis else np.zeros(0), large, n_printed)

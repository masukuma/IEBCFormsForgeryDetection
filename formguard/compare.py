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
4. Low-scoring glyphs are merged into regions (one per edited number) and reported with the
   kind of change and whether they fall inside the results table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

INK = 0.55  # normalised intensity below which a pixel is ink
PAPER = 0.75  # ... above which it is treated as blank paper when scoring glyphs
BLUR_LEVELS = (0.0, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
SHIFT = 2  # residual misalignment absorbed when scoring a glyph, px
MIN_GLYPH_INK = 25  # px; smaller blobs are specks/noise
MAX_GLYPH = 150  # px; bigger blobs are stamps, signatures, logos


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    similarity: float  # lowest glyph ZNCC in the region (1 = identical)
    ink_ref: int  # ink pixels in the reference
    ink_sub: int  # ink pixels in the submission
    kind: str  # substitution | addition | erasure
    in_table: bool
    glyphs: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageComparison:
    regions: list[Region]
    quality: float  # median glyph similarity: how comparable the two copies are at all
    glyphs: int
    blur_sigma: float  # + blurred the reference, - blurred the submission
    table_roi: tuple[int, int, int, int] | None

    def flagged(self, threshold: float, table_only: bool = True) -> list[Region]:
        return [r for r in self.regions if r.similarity < threshold and (r.in_table or not table_only)]


def normalize(img: np.ndarray) -> np.ndarray:
    """Divide out the local paper background: 1.0 = paper, 0.0 = black ink."""
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 5)
    return np.clip(img.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0), 0.0, 1.0)


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


def ruling(ink: np.ndarray, length: int = 41) -> np.ndarray:
    """Horizontal and vertical line segments (table ruling, underlines)."""
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    return h | v


def table_roi(ink: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) of the ruled results table, from its line intersections."""
    h, w = ink.shape
    # Short kernels so slightly skewed scans still yield line segments.
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(w // 30, 15)), np.uint8))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(h // 30, 15), 1), np.uint8))
    k = np.ones((5, 5), np.uint8)
    cross = cv2.dilate(hl, k) & cv2.dilate(vl, k)
    n, _, _, cent = cv2.connectedComponentsWithStats(cross)
    if n < 20:  # not a ruled table
        return None
    pts = cent[1:]
    # Percentiles drop the few intersections of the page frame with stray lines.
    x0, x1 = np.percentile(pts[:, 0], [1, 99])
    y0, y1 = np.percentile(pts[:, 1], [1, 99])
    m = int(0.01 * max(h, w))
    return int(x0) - m, int(y0) - m, int(x1) + m, int(y1) + m


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Max ZNCC of the reference glyph (minus a SHIFT border) slid over the submission box."""
    tpl = a[SHIFT:-SHIFT, SHIFT:-SHIFT]
    if tpl.std() < 1e-3 and b.std() < 1e-3:
        return 1.0  # both blank
    if tpl.std() < 1e-3 or b.std() < 1e-3:
        return 0.0  # mark on one side only
    return float(cv2.matchTemplate(b, tpl, cv2.TM_CCOEFF_NORMED).max())


def compare_pages(
    ref: np.ndarray,
    sub: np.ndarray,
    valid: np.ndarray | None = None,
    report_below: float = 0.8,
) -> PageComparison:
    """Score every glyph on the page; keep regions whose similarity is below ``report_below``."""
    ref_s, sub_s, sigma = match_sharpness(ref, sub)
    na, nb = normalize(ref_s), normalize(sub_s)
    ink_a = (na < INK).astype(np.uint8)
    ink_b = (nb < INK).astype(np.uint8)
    both = ink_a | ink_b

    lines = cv2.dilate(ruling(both), np.ones((3, 3), np.uint8))
    glyph = both & (1 - lines)
    if valid is not None:  # ignore the margin of the warped submission
        glyph &= cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8))
    glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Score stroke shape only: clip everything lighter than faint ink to paper, so the shared
    # security background behind two different numbers does not make them look alike. A light
    # blur then averages out halftone dots and JPEG ringing.
    sa = cv2.GaussianBlur(np.minimum(na, PAPER), (0, 0), 1.0)
    sb = cv2.GaussianBlur(np.minimum(nb, PAPER), (0, 0), 1.0)
    roi = table_roi((normalize(ref) < 0.5).astype(np.uint8))
    H, W = ref.shape
    m = SHIFT + 3

    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(glyph, np.ones((3, 3), np.uint8)))
    scored = []  # (x, y, w, h, sim, ink_a, ink_b, in_table)
    for i in range(1, n):
        x, y, w, h, _ = (int(v) for v in stats[i])
        if w > MAX_GLYPH or h > MAX_GLYPH or max(w, h) < 10:
            continue
        y0, y1, x0, x1 = max(0, y - m), min(H, y + h + m), max(0, x - m), min(W, x + w + m)
        ka, kb = int(ink_a[y0:y1, x0:x1].sum()), int(ink_b[y0:y1, x0:x1].sum())
        if max(ka, kb) < MIN_GLYPH_INK:
            continue
        sim = _similarity(sa[y0:y1, x0:x1], sb[y0:y1, x0:x1])
        in_table = roi is not None and roi[0] <= x + w / 2 <= roi[2] and roi[1] <= y + h / 2 <= roi[3]
        scored.append((x, y, w, h, sim, ka, kb, in_table))

    sims = np.array([s[4] for s in scored]) if scored else np.array([1.0])
    quality = float(np.median(sims))

    # Group glyphs that sit next to each other into numbers, never across table ruling, so each
    # group stays within one cell. A region is a whole number with at least one low-scoring
    # glyph; its box covers the whole number so it can be read and shown in full.
    canvas = np.zeros((H, W), np.uint8)
    for x, y, w, h, *_ in scored:
        canvas[y : y + h, x : x + w] = 1
    canvas = cv2.dilate(canvas, np.ones((9, 9), np.uint8)) & (1 - lines)
    _, labels = cv2.connectedComponents(canvas)
    groups: dict[int, list] = {}
    for s in scored:
        groups.setdefault(int(labels[s[1] + s[3] // 2, s[0] + s[2] // 2]), []).append(s)
    regions = []
    for group in groups.values():
        members = [s for s in group if s[4] < report_below]
        if not members:
            continue
        x, y = min(s[0] for s in group), min(s[1] for s in group)
        w, h = max(s[0] + s[2] for s in group) - x, max(s[1] + s[3] for s in group) - y
        ka, kb = sum(s[5] for s in members), sum(s[6] for s in members)
        if ka < 0.25 * kb:
            kind = "addition"
        elif kb < 0.25 * ka:
            kind = "erasure"
        else:
            kind = "substitution"
        regions.append(
            Region(
                x, y, w, h,
                similarity=round(min(s[4] for s in members), 3),
                ink_ref=ka, ink_sub=kb, kind=kind,
                in_table=any(s[7] for s in members),
                glyphs=len(members),
            )
        )
    regions.sort(key=lambda r: (not r.in_table, r.similarity))
    return PageComparison(regions, round(quality, 4), len(scored), sigma, roi)

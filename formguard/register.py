"""Geometric registration of a submitted page onto its reference (original) page.

Two stages:
1. SIFT keypoints + RANSAC homography on downscaled copies. This handles scale, rotation
   (including 90/180 degree page rotations), cropping and phone-camera perspective.
2. Block matching on edge maps, filtered and smoothed, to absorb residual paper curl and lens
   distortion. Blocks are several times the size of a digit and the field is smooth, so the
   correction cannot "explain away" an edited digit, which is a local change.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MATCH_SIDE = 1200  # long side used for keypoint matching
RATIO = 0.75  # Lowe's ratio test
DARK = 0.5  # background (desk, scanner lid) is darker than this fraction of the paper's brightness


@dataclass
class Features:
    pts: np.ndarray  # keypoint coordinates in full working-resolution pixels
    desc: np.ndarray | None


@dataclass
class PageMatch:
    sub_index: int
    ref_index: int | None
    inliers: int
    H: np.ndarray | None  # maps submission pixels -> reference pixels


def features(img: np.ndarray, n: int = 5000) -> Features:
    scale = MATCH_SIDE / max(img.shape)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kps, desc = cv2.SIFT_create(n).detectAndCompute(small, None)
    pts = np.float32([k.pt for k in kps]).reshape(-1, 2) / scale
    return Features(pts, desc)


def homography(ref: Features, sub: Features) -> tuple[int, np.ndarray | None]:
    if ref.desc is None or sub.desc is None or len(ref.desc) < 2 or len(sub.desc) < 2:
        return 0, None
    pairs = cv2.BFMatcher().knnMatch(sub.desc, ref.desc, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < 12:
        return 0, None
    src = sub.pts[[m.queryIdx for m in good]]
    dst = ref.pts[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return 0, None
    return int(mask.sum()), H


def match_pages(ref_feats: list[Features], sub_feats: list[Features], min_inliers: int = 150) -> list[PageMatch]:
    """Assign every submitted page to the reference page it was derived from (or None).

    Submissions reorder, duplicate, rotate and drop pages, so each submitted page is matched
    independently against all reference pages; pages are few, so this is cheap.
    """
    out = []
    for j, sf in enumerate(sub_feats):
        best = PageMatch(j, None, 0, None)
        for i, rf in enumerate(ref_feats):
            inl, H = homography(rf, sf)
            if inl > best.inliers:
                best = PageMatch(j, i, inl, H)
        if best.inliers < min_inliers:
            best = PageMatch(j, None, best.inliers, None)
        out.append(best)
    return out


def paper_mask(img: np.ndarray, margin: float = 0.01) -> np.ndarray:
    """The sheet of paper in a photograph: everything but large dark areas reaching the image edge.

    Outside the sheet (desk, shadow, the curled edge of a printout) nothing on the form can be
    compared. Print, handwriting and table ruling are thin, so an opening removes them from the
    dark mask before background is looked for; a flat scan is all paper. The mask is shrunk by
    ``margin`` of the long side.
    """
    s = 600 / max(img.shape)
    small = cv2.GaussianBlur(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (0, 0), 2)
    # Much darker than the paper: lighting across a photographed sheet varies too much for a
    # threshold between paper and background to stay off the dimly lit side of the paper.
    dark = np.where(small < DARK * np.percentile(small, 90), 255, 0).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    h, w = small.shape
    sheet = np.full(small.shape, 255, np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= 0.01 * small.size and (x == 0 or y == 0 or x + bw == w or y + bh == h):
            sheet[labels == i] = 0
    if (sheet > 0).mean() < 0.3:  # not a sheet on a background after all
        sheet[:] = 255
    sheet[0, :] = sheet[-1, :] = sheet[:, 0] = sheet[:, -1] = 0  # so the margin is taken at the image edge too
    e = max(1, int(margin * max(small.shape)))
    sheet = cv2.erode(sheet, np.ones((2 * e + 1, 2 * e + 1), np.uint8))
    return cv2.resize(sheet, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def warp(sub: np.ndarray, H: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Warp the submission into the reference frame. Also returns the mask of comparable pixels:
    inside the warped image and on the paper of the photographed sheet."""
    h, w = shape
    warped = cv2.warpPerspective(sub, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    sheet = paper_mask(sub).astype(np.uint8) * 255
    valid = cv2.warpPerspective(sheet, H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return warped, valid > 0


def _agreement(a: np.ndarray, b: np.ndarray) -> float:
    """How well two registered pages line up: correlation of their edge strength at half scale."""
    s = 0.5
    def edges(im):
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
        return cv2.magnitude(cv2.Sobel(im, cv2.CV_32F, 1, 0), cv2.Sobel(im, cv2.CV_32F, 0, 1))
    x, y = edges(a), edges(b)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


def _block_shifts(ref: np.ndarray, sub: np.ndarray, step: int, block: int, search: int) -> tuple[np.ndarray, np.ndarray]:
    """Local shift of ``sub`` against ``ref`` on a grid of blocks, by correlation of edge maps.

    Returns (shift field [rows, cols, 2], confidence mask). Blocks without texture, or whose
    best match is weak, are marked unreliable rather than trusted.
    """
    def edges(im):
        im = cv2.GaussianBlur(im.astype(np.float32), (0, 0), 1.5)
        return cv2.magnitude(cv2.Sobel(im, cv2.CV_32F, 1, 0), cv2.Sobel(im, cv2.CV_32F, 0, 1))
    ea, eb = edges(ref), edges(sub)
    h, w = ref.shape
    ys = list(range(search, h - block - search, step))
    xs = list(range(search, w - block - search, step))
    field = np.zeros((len(ys), len(xs), 2), np.float32)
    ok = np.zeros((len(ys), len(xs)), bool)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            tpl = ea[y : y + block, x : x + block]
            if tpl.std() < 2.0:
                continue
            win = eb[y - search : y + block + search, x - search : x + block + search]
            r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            _, peak, _, (px, py) = cv2.minMaxLoc(r)
            if peak < 0.5:
                continue
            field[i, j] = (px - search, py - search)
            ok[i, j] = True
    return field, ok


def refine_maps(ref: np.ndarray, sub: np.ndarray, step: int = 96, block: int = 192, search: int = 16):
    """Smooth correction for residual misalignment (paper curl, lens), kept only if it helps.

    Block matching: each block of the page is located in the registered copy by correlation,
    unreliable blocks are dropped, outliers are removed by a median filter, and the field is
    smoothed over several blocks, so it follows the sheet's curl but cannot move one figure.
    Returns (maps for cv2.remap, max correction px); maps is None if the correction does not
    improve the pages' agreement.
    """
    field, ok = _block_shifts(ref, sub, step, block, search)
    if ok.sum() < 12:
        return None, 0.0
    fill = np.median(field[ok], axis=0)
    field[~ok] = fill
    for c in range(2):
        field[..., c] = cv2.medianBlur(field[..., c], 5) if min(field.shape[:2]) >= 5 else field[..., c]
    h, w = ref.shape
    gh, gw = field.shape[:2]
    up = cv2.resize(field, (gw * step, gh * step), interpolation=cv2.INTER_LINEAR)
    o = search + block // 2 - step // 2  # page position of the area the grid of blocks covers
    up = up[: h - o, : w - o]
    # Beyond the outermost blocks the field continues as it is at the edge (paper curls most
    # at the edges of the sheet), rather than falling back to the page's median.
    full = cv2.copyMakeBorder(up, o, h - o - up.shape[0], o, w - o - up.shape[1], cv2.BORDER_REPLICATE)
    full = cv2.GaussianBlur(full, (0, 0), step / 2)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    maps = (gx + full[..., 0], gy + full[..., 1])
    out = cv2.remap(sub, *maps, cv2.INTER_LINEAR, borderValue=255)
    if _agreement(ref, out) <= _agreement(ref, sub):
        return None, 0.0
    return maps, float(np.abs(full).max())


def refine(ref: np.ndarray, sub: np.ndarray) -> tuple[np.ndarray, float]:
    """Correct smooth residual misalignment. Returns (re-warped submission, max correction px)."""
    maps, moved = refine_maps(ref, sub)
    if maps is None:
        return sub, 0.0
    return cv2.remap(sub, *maps, cv2.INTER_LINEAR, borderValue=255), moved

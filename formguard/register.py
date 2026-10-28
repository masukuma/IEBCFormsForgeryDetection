"""Geometric registration of a submitted page onto its reference (original) page.

Two stages:
1. SIFT keypoints + RANSAC homography on downscaled copies. This handles scale, rotation
   (including 90/180 degree page rotations), cropping and phone-camera perspective.
2. Dense optical flow, heavily smoothed, to absorb residual paper curl and lens distortion.
   The smoothing keeps the correction low-frequency so it cannot "explain away" an edited
   digit, which is a local, high-frequency change.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MATCH_SIDE = 1200  # long side used for keypoint matching
RATIO = 0.75  # Lowe's ratio test


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


def warp(sub: np.ndarray, H: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Warp the submission into the reference frame. Also returns the valid-pixel mask."""
    h, w = shape
    warped = cv2.warpPerspective(sub, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    valid = cv2.warpPerspective(np.full(sub.shape, 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return warped, valid > 0


def refine(ref: np.ndarray, sub: np.ndarray, smooth: float = 25.0) -> tuple[np.ndarray, float]:
    """Correct smooth residual misalignment. Returns (re-warped submission, max correction px)."""
    s = 0.5
    r2 = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m2 = cv2.resize(sub, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(r2, m2, None)
    flow = cv2.GaussianBlur(flow, (0, 0), smooth)
    flow = cv2.resize(flow, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR) / s
    gx, gy = np.meshgrid(np.arange(ref.shape[1], dtype=np.float32), np.arange(ref.shape[0], dtype=np.float32))
    out = cv2.remap(sub, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderValue=255)
    return out, float(np.abs(flow).max())

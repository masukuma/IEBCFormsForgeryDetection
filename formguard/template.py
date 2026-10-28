"""The printed form: what every form of one type carries before anyone writes on it.

Alterations are made to the *entries* on a form (figures, names, ticks), not to its printed
labels. Separating the two matters on photographs of printouts, where the grey printed labels
often fade into the paper while the pen entries stay dark: faded print would otherwise score
like a change. The printed form is found by overlaying other filled-in forms of the same type
on the reference. Ink that most of them share at the same place is print; their entries
differ from form to form and drop out.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .register import features, homography, refine, warp

MIN_INLIERS = 100  # forms of one type share their whole printed layout, so this is easily met


def build_template(ref: np.ndarray, pool: list[np.ndarray], max_forms: int = 7) -> tuple[np.ndarray | None, int]:
    """The printed form, aligned on ``ref``: the pixel-wise median of other forms of its type.

    Each pool form is flattened (paper 1.0, ink 0.0), registered on the reference (homography,
    then the smooth local correction) and stacked; the median keeps what most of them share,
    the print, and drops their entries, which differ from form to form. Returns (image, forms
    used), or (None, n) when fewer than three forms could be registered, since two forms can
    share an entry by chance.
    """
    from .compare import normalize, stretch  # compare imports nothing from here

    rf = features(ref)
    stack = []
    for img in pool:
        if len(stack) == max_forms:
            break
        inliers, H = homography(rf, features(img))
        if H is None or inliers < MIN_INLIERS:
            continue
        warped, _ = warp(img, H, ref.shape)
        warped, _ = refine(ref, warped)
        stack.append(stretch(normalize(warped)))
    if len(stack) < 3:
        return None, len(stack)
    return np.median(np.stack(stack), axis=0).astype(np.float32), len(stack)


def load_pool(sources: list[str | Path], exclude: set[Path], long_side: int) -> list[np.ndarray]:
    """First page of each pool form, skipping the reference itself."""
    from .render import load_pages, parse_source

    out = []
    for s in sources:
        if parse_source(s)[0].resolve() in exclude:
            continue
        out.append(load_pages(s, long_side)[0].image)
    return out

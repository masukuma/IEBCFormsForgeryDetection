"""Was a questioned copy made from the same image as the reference, or is it an independent capture?

A form exists as one physical sheet; every photograph or scan of it is a separate *capture*
with its own framing, perspective, lighting and background (the desk, a hand, another sheet
under the form). Two captures of the same sheet share the handwriting, signatures and stamp
impressions, but not those capture-specific details. A copy *derived* from a capture (printed,
re-photographed, edited) inherits them.

So, after registration, two measurements:

* ``background``: correlation of the reference capture's outer band (beyond the form sheet,
  where the background shows) with the same band in the questioned copy. High correlation means
  the questioned copy contains the reference capture's background: it was made from that image.
  Reported as not measurable when the band is blank (nothing to compare).
* ``marks``: share of the reference's handwritten and stamped marks reproduced in place in the
  questioned copy. A copy derived from the reference reproduces nearly all of them; an unrelated
  form reproduces almost none.

Neither measurement says which version came first. That follows from the reference's custody
(signed portal upload, physical originals) and is argued in the report, not computed here.
"""
from __future__ import annotations

import cv2
import numpy as np

BAND = 0.05  # outer fraction of the reference frame treated as capture background


def background_correlation(ref: np.ndarray, sub: np.ndarray, valid: np.ndarray | None = None, tile: int = 48) -> dict:
    """Median tile-wise correlation of fine texture in the reference's outer band.

    Fine texture only (difference of Gaussians): the sheet's own edge lands in the same place in
    any registered photograph of the sheet, so large-scale structure would correlate even between
    independent captures. Correlating small tiles separately and taking the median keeps the few
    tiles crossed by that edge from deciding the result.
    """
    h, w = ref.shape
    b = int(BAND * min(h, w))
    band = np.zeros((h, w), bool)
    band[:b, :] = band[-b:, :] = True
    band[:, :b] = band[:, -b:] = True
    if valid is not None:
        band &= cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    dog = lambda im: cv2.GaussianBlur(im.astype(np.float32) / 255, (0, 0), 1.0) - cv2.GaussianBlur(im.astype(np.float32) / 255, (0, 0), 6.0)
    fa, fb = dog(ref), dog(sub)
    gx, gy = cv2.Sobel(fa, cv2.CV_32F, 1, 0), cv2.Sobel(fa, cv2.CV_32F, 0, 1)
    scores = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            if not band[y : y + tile, x : x + tile].all():
                continue
            a, c = fa[y : y + tile, x : x + tile], fb[y : y + tile, x : x + tile]
            if a.std() < 0.003:  # blank tile: nothing to compare
                continue
            # Skip tiles dominated by one straight edge (the sheet's border): any registered
            # photograph of the sheet has it in the same place.
            tx, ty = gx[y : y + tile, x : x + tile], gy[y : y + tile, x : x + tile]
            jxx, jyy, jxy = float((tx * tx).sum()), float((ty * ty).sum()), float((tx * ty).sum())
            if np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-12) > 0.7:
                continue
            a, c = a - a.mean(), c - c.mean()
            scores.append(float((a * c).sum() / (np.sqrt((a * a).sum() * (c * c).sum()) + 1e-9)))
    if len(scores) < 10:
        return {"measurable": False, "tiles": len(scores), "reason": "too little texture in the reference's background band"}
    scores = np.array(scores)
    return {"measurable": True, "zncc": round(float(np.median(scores)), 3), "tiles": len(scores), "tiles_above_0_5": round(float((scores > 0.5).mean()), 3)}


def assess(ref: np.ndarray, sub: np.ndarray, valid: np.ndarray | None, mark_similarities: np.ndarray) -> dict:
    """Capture-provenance facts for one registered page pair."""
    sims = np.asarray(mark_similarities, dtype=float)
    reproduced = float((sims >= 0.8).mean()) if sims.size else 0.0
    out = {
        "background": background_correlation(ref, sub, valid),
        "marks_compared": int(sims.size),
        "marks_reproduced": round(reproduced, 4),
    }
    bg = out["background"]
    if reproduced >= 0.75 and bg.get("measurable") and bg["zncc"] >= 0.5:
        # Derived from the reference image itself. The shared background decides it; fewer marks
        # are reproduced when figures were changed or the copy is a poor photograph.
        out["finding"] = "same_capture"
    elif reproduced >= 0.9:
        out["finding"] = "same_sheet"  # same marks; capture background not shown to be shared
    elif reproduced < 0.5:
        out["finding"] = "different_document"
    else:
        out["finding"] = "inconclusive"
    return out

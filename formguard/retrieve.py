"""Find the original a questioned page was made from, among many candidate originals.

All forms of one type share the same printed template, so template features match every
candidate. What singles out the true original is the *handwriting, signatures and stamps*:
they add hundreds of geometrically consistent matches that no other form has. On the 2022
Forms 34A, the true original scores 1,000-1,800 RANSAC inliers against a phone photograph
of a printout, while the best wrong candidate scores about 280. A match is accepted only if
it is both strong and well clear of the runner-up.

Candidate features are cached on disk, keyed by file hash, so a pool of thousands of
originals is processed once and then queried in seconds.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .register import RATIO, Features, features
from .render import load_pages, parse_source

LONG_SIDE = 1800


@dataclass
class Retrieval:
    reference: Path | None
    inliers: int
    runner_up: Path | None
    runner_up_inliers: int
    accepted: bool


def _key(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


class ReferenceIndex:
    def __init__(self, candidates: list[Path], cache_dir: Path | None = None):
        self.candidates = [Path(c) for c in candidates]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._feats: list[Features] | None = None

    def _load(self, path: Path) -> Features:
        cache = self.cache_dir / f"{_key(path)}.npz" if self.cache_dir else None
        if cache is not None and cache.exists():
            z = np.load(cache)
            return Features(z["pts"], z["desc"])
        f = features(load_pages(path, LONG_SIDE)[0].image)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, pts=f.pts, desc=f.desc if f.desc is not None else np.zeros((0, 128), np.float32))
        return f

    @property
    def feats(self) -> list[Features]:
        if self._feats is None:
            self._feats = [self._load(c) for c in self.candidates]
        return self._feats

    def query(self, image: np.ndarray, min_inliers: int = 300, margin: float = 2.0) -> Retrieval:
        """Best candidate for one page image (any resolution; rescaled to the index scale)."""
        scale = LONG_SIDE / max(image.shape)
        img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        q = features(img)
        flann = _page_index(q)
        scores = sorted(((_inliers(f, q, flann), i) for i, f in enumerate(self.feats)), reverse=True)
        (best, bi), (second, si) = scores[0], scores[1] if len(scores) > 1 else (0, None)
        accepted = best >= min_inliers and best >= margin * max(second, 1)
        return Retrieval(
            self.candidates[bi] if best else None,
            int(best),
            self.candidates[si] if si is not None else None,
            int(second),
            accepted,
        )


def _page_index(q: Features):
    """Approximate nearest-neighbour index over one questioned page's descriptors, built once
    and reused for every candidate (a brute-force matcher re-scans the page for each one)."""
    if q.desc is None or len(q.desc) < 2:
        return None
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 48})
    flann.add([q.desc.astype(np.float32)])
    flann.train()
    return flann


def _inliers(cand: Features, q: Features, flann) -> int:
    """RANSAC inliers between a candidate and the indexed questioned page (ratio-tested matches)."""
    if flann is None or cand.desc is None or len(cand.desc) < 2:
        return 0
    pairs = flann.knnMatch(cand.desc.astype(np.float32), k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < 12:
        return 0
    src = q.pts[[m.trainIdx for m in good]]
    dst = cand.pts[[m.queryIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return int(mask.sum()) if H is not None else 0


def query_source(index: ReferenceIndex, source: str | Path) -> list[tuple[int, Retrieval]]:
    """Retrieve an original for every page of a questioned source ("file.pdf#pages=46-130")."""
    path, _ = parse_source(source)
    return [(p.index + 1, index.query(p.image)) for p in load_pages(source, LONG_SIDE)]

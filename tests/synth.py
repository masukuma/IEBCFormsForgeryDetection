"""Synthetic stand-ins for scanned forms: a ruled results table on a textured security background."""
from __future__ import annotations

import cv2
import numpy as np

ROWS, COLS = 24, 6
X0, Y0, CW, RH = 120, 160, 190, 52  # table origin, column width, row height (px)
SIZE = (1800, 1400)  # page height, width


def background(seed: int = 0) -> np.ndarray:
    """Light microtext-like texture, like the guilloche/microprint behind real forms."""
    rng = np.random.default_rng(seed)
    page = np.full(SIZE, 250, np.uint8)
    for _ in range(900):
        x, y = int(rng.integers(0, SIZE[1])), int(rng.integers(0, SIZE[0]))
        cv2.putText(page, "IEBC", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, 215, 1, cv2.LINE_AA)
    return page


def values(seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 700, size=(ROWS, COLS))


def cell_box(r: int, c: int) -> tuple[int, int, int, int]:
    return X0 + c * CW, Y0 + r * RH, CW, RH


def draw(vals: np.ndarray, seed: int = 0) -> np.ndarray:
    page = background(seed)
    for r in range(ROWS + 1):
        cv2.line(page, (X0, Y0 + r * RH), (X0 + COLS * CW, Y0 + r * RH), 20, 2)
    for c in range(COLS + 1):
        cv2.line(page, (X0 + c * CW, Y0), (X0 + c * CW, Y0 + ROWS * RH), 20, 2)
    cv2.putText(page, "FORM 34B  COLLATION OF RESULTS", (X0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 2, cv2.LINE_AA)
    for r in range(ROWS):
        for c in range(COLS):
            x, y, w, h = cell_box(r, c)
            cv2.putText(page, str(vals[r, c]), (x + w - 90, y + h - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 15, 2, cv2.LINE_AA)
    return page


def degrade(page: np.ndarray, angle: float = 2.0, scale: float = 0.85, blur: float = 0.8, jpeg: int = 60) -> np.ndarray:
    """What happens to a form on its way to a whistleblower: re-photographed, resized, recompressed."""
    h, w = page.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    out = cv2.warpAffine(page, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    out = cv2.GaussianBlur(out, (0, 0), blur)
    out = np.clip(out.astype(np.float32) * 0.9 + 12, 0, 255).astype(np.uint8)  # contrast shift
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

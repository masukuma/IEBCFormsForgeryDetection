"""Optional digit reading for flagged regions, via the tesseract CLI if it is installed.

Only the handful of regions that were flagged are read, so OCR cost does not grow with the
size of the form. Detection never depends on OCR; it only annotates evidence ("26" -> "226").
"""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np


def available() -> bool:
    return shutil.which("tesseract") is not None


def _tesseract(img: np.ndarray) -> tuple[str, float]:
    ok, png = cv2.imencode(".png", img)
    if not ok:
        return "", 0.0
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789", "tsv"],
        input=png.tobytes(),
        capture_output=True,
        timeout=30,
        # One thread per call: callers already run in parallel, and tesseract's own OpenMP
        # threads oversubscribe the CPU badly (tens of runnable threads per core).
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    )
    words, confs = [], []
    for line in proc.stdout.decode(errors="ignore").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) == 12 and cols[11].strip() and float(cols[10]) >= 0:
            words.append(cols[11].strip())
            confs.append(float(cols[10]))
    return "".join(words), (float(np.mean(confs)) if confs else 0.0)


ROTATIONS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)  # upright first
UPRIGHT_OK = 60.0  # an upright reading this confident is kept
TURN_MARGIN = 20.0  # a sideways reading must beat a weak upright one by this much


def read_number(crop: np.ndarray, rotations=ROTATIONS) -> tuple[str, float, int | None]:
    """Read the digits in a grayscale crop, trying the upright and both 90-degree rotations.

    Tables on these forms are often printed sideways on the page, so orientation is unknown.
    Returns (text, confidence 0-100, the cv2 rotation code that read best or None).
    """
    if crop.size == 0:
        return "", 0.0, None
    big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    big = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Erase cell borders caught in the crop; they read as "1"s.
    ink = 255 - big
    h, w = ink.shape
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(w * 2 // 3, 1)), np.uint8))
    lines |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(h * 2 // 3, 1), 1), np.uint8))
    big[cv2.dilate(lines, np.ones((5, 5), np.uint8)) > 0] = 255
    big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    best = ("", 0.0, None)
    upright = 0.0
    for rot in rotations:
        text, conf = _tesseract(big if rot is None else cv2.rotate(big, rot))
        if rot is None:
            upright = conf if text else 0.0
        # Noise on a photographed copy can read "better" sideways: turning is accepted only when
        # the upright reading is weak and a sideways one is far more confident.
        elif not (upright < UPRIGHT_OK and conf >= upright + TURN_MARGIN):
            continue
        if text and conf > best[1]:
            best = (text, conf, rot)
    return best

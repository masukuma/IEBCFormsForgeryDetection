"""Load documents (PDF or image) as grayscale page rasters at a common working scale."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class Page:
    index: int
    image: np.ndarray  # uint8 grayscale, long side == working long side
    native_long_side: int  # long side, in pixels, of the best source raster behind this page


def _native_long_side(page: pymupdf.Page, fallback: int) -> int:
    """Pixel density of the page's largest raster, expressed as a full-page long side.

    Scans and phone photos are a single image filling the page; vector PDFs have none,
    in which case they can be rendered at any resolution and ``fallback`` is used.
    """
    page_long = max(page.rect.width, page.rect.height)
    best = 0.0
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        bbox_long = max(x1 - x0, y1 - y0)
        if bbox_long < 0.5 * page_long:  # logos, QR codes, MRC fragments
            continue
        best = max(best, max(info["width"], info["height"]) * page_long / bbox_long)
    return int(best) if best else fallback


def load_pages(path: str | Path, long_side: int = 2600) -> list[Page]:
    path = Path(path)
    if path.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"cannot read image {path}")
        native = max(img.shape)
        scale = long_side / native
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        return [Page(0, img, native)]

    pages = []
    with pymupdf.open(path) as doc:
        for i, pg in enumerate(doc):
            zoom = long_side / max(pg.rect.width, pg.rect.height)
            pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY, alpha=False)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, : pix.width].copy()
            pages.append(Page(i, img, _native_long_side(pg, fallback=long_side * 2)))
    return pages


def pdf_metadata(path: str | Path) -> dict:
    """Provenance facts that are cheap to read and often telling (producer, dates, raster layout)."""
    path = Path(path)
    if path.suffix.lower() in IMAGE_EXTS:
        return {"type": "image"}
    with pymupdf.open(path) as doc:
        meta = {k: v for k, v in (doc.metadata or {}).items() if v}
        rasters = []
        for pg in doc:
            imgs = pg.get_image_info()
            largest = max(imgs, key=lambda i: i["width"] * i["height"], default=None)
            rasters.append(
                {
                    "images": len(imgs),
                    "largest": f"{largest['width']}x{largest['height']}" if largest else None,
                    "native_long_side": _native_long_side(pg, fallback=0),
                }
            )
        return {"type": "pdf", "pages": doc.page_count, "metadata": meta, "rasters": rasters}

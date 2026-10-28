"""Load documents (PDF or image) as grayscale page rasters at a common working scale."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf

IMAGE_EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def parse_source(spec: str | Path) -> tuple[Path, list[int] | None]:
    """Split "file.pdf#page=75" or "file.pdf#pages=46-50,52" into (path, 1-based page numbers)."""
    m = re.match(r"^(.*)#pages?=([\d,\- ]+)$", str(spec))
    if not m:
        return Path(spec), None
    pages: list[int] = []
    for part in m.group(2).split(","):
        a, _, b = part.strip().partition("-")
        pages.extend(range(int(a), int(b or a) + 1))
    return Path(m.group(1)), pages


@dataclass
class Page:
    index: int  # 0-based page number within the source document
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


def load_pages(source: str | Path, long_side: int = 2600) -> list[Page]:
    """Render a document (or the pages named in a "#page=" suffix) as grayscale pages."""
    path, wanted = parse_source(source)
    if path.suffix.lower() in IMAGE_EXTS:
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"cannot read image {path}")
        native = max(img.shape)
        scale = long_side / native
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        return [Page(0, img, native)]

    pages = []
    with pymupdf.open(path) as doc:
        for i, pg in enumerate(doc):
            if wanted is not None and i + 1 not in wanted:
                continue
            zoom = long_side / max(pg.rect.width, pg.rect.height)
            pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY, alpha=False)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, : pix.width].copy()
            pages.append(Page(i, img, _native_long_side(pg, fallback=long_side * 2)))
    return pages


def pdf_metadata(source: str | Path) -> dict:
    """Provenance facts that are cheap to read and often telling (producer, dates, raster layout)."""
    path, _ = parse_source(source)
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

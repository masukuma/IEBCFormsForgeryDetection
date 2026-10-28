"""Compare a submitted document against its original and produce a verdict with evidence."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import ocr, provenance
from .compare import PageComparison, Region, compare_pages, sideways
from .register import features, homography, match_pages, paper_mask, refine, warp
from .render import load_pages, parse_source, pdf_metadata
from .template import build_template, load_pool

CLEAN, ALTERED, REVIEW, INCONCLUSIVE, IDENTICAL = "clean", "altered", "review", "inconclusive", "identical"


@dataclass
class Config:
    long_side: int = 2600  # working resolution (px on the page's long side); ~220 dpi on A3
    min_inliers: int = 150  # keypoint agreement required to accept a page match
    flag_below: float = 0.5  # similarity below which a change counts as an alteration
    possible_below: float = 0.65  # ... and below which it is a possible change, settled by reading the figure
    min_quality: float = 0.85  # median glyph similarity below which a page is too degraded to judge
    max_evidence: int = 50  # cells per page that get an evidence crop and an OCR reading
    ocr: bool = True  # read flagged numbers if tesseract is installed
    template: tuple[str, ...] = ()  # other forms of the original's type, to recognise its print


def sha256(source: str | Path) -> str:
    """SHA-256 of the file behind a source spec (a "#page=" suffix is ignored)."""
    path, _ = parse_source(source)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _crop(img: np.ndarray, r: Region, margin: int, min_w: int = 0, min_h: int = 0) -> np.ndarray:
    """The region plus a margin, widened about its centre to at least min_w x min_h."""
    H, W = img.shape
    w, h = max(r.w + 2 * margin, min_w), max(r.h + 2 * margin, min_h)
    x0 = min(max(0, r.x + r.w // 2 - w // 2), max(0, W - w))  # shifted inward at the page edge
    y0 = min(max(0, r.y + r.h // 2 - h // 2), max(0, H - h))
    return img[y0 : min(H, y0 + h), x0 : min(W, x0 + w)]


def _evidence_tile(ref: np.ndarray, sub: np.ndarray, r: Region, rotation: int | None) -> np.ndarray:
    """Original | submission, side by side, upright if OCR found the reading direction.

    A region can be one stroke of a number (a faded or erased digit), so the crop is never
    smaller than a whole three-digit figure and its neighbours above and below.
    """
    k = max(ref.shape) / 2600
    a, b = _crop(ref, r, 40, int(240 * k), int(150 * k)), _crop(sub, r, 40, int(240 * k), int(150 * k))
    if rotation is not None:
        a, b = cv2.rotate(a, rotation), cv2.rotate(b, rotation)
    gap = np.full((a.shape[0], 8), 0, np.uint8)
    tile = cv2.cvtColor(np.hstack([a, gap, b]), cv2.COLOR_GRAY2BGR)
    bar = np.full((22, tile.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, "original", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(bar, "submission", (a.shape[1] + 12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    return np.vstack([bar, tile])


def _overview(ref: np.ndarray, sub: np.ndarray, flagged: list[Region], other: list[Region]) -> np.ndarray:
    a, b = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR), cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
    for regions, color in ((other, (0, 160, 255)), (flagged, (0, 0, 255))):
        for r in regions:
            for img in (a, b):
                cv2.rectangle(img, (r.x - 8, r.y - 8), (r.x + r.w + 8, r.y + r.h + 8), color, 4)
    both = np.hstack([a, np.zeros((a.shape[0], 20, 3), np.uint8), b])
    return cv2.resize(both, None, fx=1400 / both.shape[1], fy=1400 / both.shape[1], interpolation=cv2.INTER_AREA)


def _page_verdict(pc: PageComparison, cfg: Config) -> str:
    if pc.quality < cfg.min_quality:
        return INCONCLUSIVE
    if any(r.on_figure for r in pc.flagged(cfg.flag_below, table_only=True)):
        return ALTERED
    if pc.flagged(cfg.possible_below, table_only=False):
        return REVIEW  # possible changes to be read, marks on blank paper, or changes outside the table
    return CLEAN


_TEMPLATES: dict[tuple, tuple[np.ndarray | None, int]] = {}
_POOLS: dict[tuple, list[np.ndarray]] = {}


def _pool(original: Path, cfg: Config) -> list[np.ndarray]:
    key = (cfg.template, cfg.long_side, str(original))
    if key not in _POOLS:
        _POOLS.clear()
        _POOLS[key] = load_pool(list(cfg.template[:12]), {parse_source(original)[0].resolve()}, cfg.long_side)
    return _POOLS[key]


def _template(original: Path, page: int, ref: np.ndarray, cfg: Config) -> tuple[np.ndarray | None, int]:
    """Printed form of one original page, built once and reused for every exhibit of a case."""
    key = (str(original), page, cfg.long_side, cfg.template)
    if key not in _TEMPLATES:
        if len(_TEMPLATES) > 4:
            _TEMPLATES.clear()
        _TEMPLATES[key] = build_template(ref, _pool(original, cfg))
    return _TEMPLATES[key]


def _upside_down(ref: np.ndarray, original: Path, cfg: Config) -> bool:
    """Whether an original is stored upside down: other forms of its type register onto it turned
    by about 180 degrees. Its evidence images would otherwise be shown upside down."""
    rf = features(ref)
    angles = []
    for img in _pool(original, cfg)[:4]:
        n, H = homography(features(img), rf)
        if H is not None and n >= 100:
            angles.append(abs(np.degrees(np.arctan2(H[1, 0], H[0, 0]))))
        if len(angles) == 2:
            break
    return bool(angles) and float(np.median(angles)) > 135


def compare_documents(original: Path, submission: Path, out_dir: Path | None = None, cfg: Config = Config()) -> dict:
    """Full comparison of one submitted document against one original. Returns a JSON-able report."""
    t0 = time.time()
    report: dict = {
        "submission": str(submission),
        "original": str(original),
        "submission_sha256": sha256(submission),
        "original_sha256": sha256(original),
        "submission_meta": pdf_metadata(submission),
        "original_meta": pdf_metadata(original),
    }
    whole_files = parse_source(submission)[1] is None and parse_source(original)[1] is None
    if whole_files and report["submission_sha256"] == report["original_sha256"]:
        report.update(verdict=IDENTICAL, pages=[], notes=["byte-identical to the original"], seconds=round(time.time() - t0, 1))
        return report

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    use_ocr = cfg.ocr and ocr.available()

    ref_pages = load_pages(original, cfg.long_side)
    turned = set()
    if cfg.template:
        for p in ref_pages:
            if _upside_down(p.image, original, cfg):
                p.image = cv2.rotate(p.image, cv2.ROTATE_180)
                turned.add(p.index)
    sub_pages = load_pages(submission, cfg.long_side)
    matches = match_pages([features(p.image) for p in ref_pages], [features(p.image) for p in sub_pages], cfg.min_inliers)

    pages = []
    for m in matches:
        entry: dict = {"submission_page": sub_pages[m.sub_index].index + 1, "inliers": m.inliers}
        if m.ref_index is None:
            entry.update(original_page=None, verdict=INCONCLUSIVE, note="no page of the original matches this page")
            pages.append(entry)
            continue
        ref = ref_pages[m.ref_index].image
        sub = sub_pages[m.sub_index].image
        warped, valid = warp(sub, m.H, ref.shape)
        # Where the copy's image lies at all: the provenance test needs the background beyond
        # the sheet, which the comparison of entries leaves out.
        covered = cv2.warpPerspective(np.full(sub.shape, 255, np.uint8), m.H, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_NEAREST) > 0
        valid &= paper_mask(ref)  # originals photographed by the KIEMS kit show the desk too
        warped, residual = refine(ref, warped)
        tpl, tpl_forms = _template(original, ref_pages[m.ref_index].index, ref, cfg) if cfg.template else (None, 0)
        pc = compare_pages(ref, warped, valid, template=tpl)
        verdict = _page_verdict(pc, cfg)
        # Clear changes, and possible ones (a "3" scores fairly high against the "8" it was made
        # from) that a reading of both figures settles.
        flagged = pc.flagged(cfg.possible_below, table_only=False)
        entry.update(
            original_page=ref_pages[m.ref_index].index + 1,
            original_turned_180=ref_pages[m.ref_index].index in turned,
            verdict=verdict,
            quality=pc.quality,
            glyphs_compared=pc.glyphs,
            printed_glyphs=pc.printed,
            template_forms=tpl_forms,
            residual_warp_px=round(residual, 1),
            blur_sigma=pc.blur_sigma,
            provenance=provenance.assess(ref, warped, covered, pc.similarities),
            large_marks=pc.large_marks,
            changes=[],
        )
        tag = f"p{entry['submission_page']}"
        # Text orientation: upright unless the page's table is printed sideways; which way it
        # turns is then fixed by the first confident reading.
        page_rotation: tuple | None = None if sideways(ref) else (None,)
        for k, r in enumerate(flagged):
            change = r.to_dict()
            change["clear"] = r.similarity < cfg.flag_below
            rotation = None
            if use_ocr and k < cfg.max_evidence:
                rots = page_rotation or ocr.ROTATIONS[1:]
                before, conf_b, rotation = ocr.read_number(_crop(ref, r, 4), rots)
                after, conf_a, rot_a = ocr.read_number(_crop(warped, r, 4), (rotation,) if before else rots)
                rotation = rotation if before else rot_a
                if page_rotation is None and before and conf_b >= 85:
                    page_rotation = (rotation,)
                change.update(
                    read_original=before,
                    read_submission=after,
                    ocr_confidence=round(min(conf_b, conf_a) if before and after else max(conf_b, conf_a), 1),
                    # Same reading on both sides: the figure was re-typeset in place, value unchanged.
                    value_changed=None if not (before and after) else before != after,
                )
            if out_dir is not None and k < cfg.max_evidence:
                name = f"{tag}_change{k + 1:02d}.png"
                cv2.imwrite(str(out_dir / name), _evidence_tile(ref, warped, r, rotation))
                change["evidence"] = name
            entry["changes"].append(change)
        if out_dir is not None and flagged:
            name = f"{tag}_overview.jpg"
            in_tab = [r for r in flagged if r.in_table and r.on_figure]
            cv2.imwrite(str(out_dir / name), _overview(ref, warped, in_tab, [r for r in flagged if r not in in_tab]), [cv2.IMWRITE_JPEG_QUALITY, 80])
            entry["overview"] = name
        pages.append(entry)

    matched = {m.ref_index for m in matches if m.ref_index is not None}
    notes = []
    swaps = find_swaps(pages)
    if swaps:
        notes.append(f"{len(swaps)} pair(s) of figures swapped between cells")
    missing = [ref_pages[i].index + 1 for i in range(len(ref_pages)) if i not in matched]
    if missing:
        notes.append(f"original page(s) {missing} not present in the submission")
    dupes = len([m for m in matches if m.ref_index is not None]) - len(matched)
    if dupes:
        notes.append(f"{dupes} duplicated page(s) in the submission")
    unmatched = [p["submission_page"] for p in pages if p.get("original_page") is None]
    if unmatched:
        notes.append(f"submission page(s) {unmatched} have no counterpart in the original")

    verdicts = {p["verdict"] for p in pages}
    for verdict in (ALTERED, REVIEW, INCONCLUSIVE, CLEAN):
        if verdict in verdicts:
            break
    else:
        verdict = INCONCLUSIVE  # empty document
    report.update(
        verdict=verdict,
        altered_numbers=sum(1 for p in pages for c in p.get("changes", []) if c["in_table"] and c["on_figure"] and c["clear"]),
        possible_numbers=sum(1 for p in pages for c in p.get("changes", []) if c["in_table"] and c["on_figure"] and not c["clear"]),
        values_changed=sum(1 for p in pages for c in p.get("changes", []) if c["in_table"] and c["on_figure"] and c.get("value_changed")),
        swaps=swaps,
        pages=pages,
        notes=notes,
        seconds=round(time.time() - t0, 1),
    )
    if out_dir is not None:
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default))
    return report


def find_swaps(pages: list[dict]) -> list[dict]:
    """Pairs of changed cells whose figures were exchanged (A shows B's value and B shows A's).

    A tell-tale of deliberate tampering: votes moved between candidates or polling stations
    rather than mis-transcribed. Needs OCR readings; returns [] without them.
    """
    cells = [
        (p["submission_page"], i + 1, c["read_original"], c["read_submission"])
        for p in pages
        for i, c in enumerate(p.get("changes", []))
        if c["in_table"] and c.get("read_original") and c.get("read_submission") and c["read_original"] != c["read_submission"]
    ]
    swaps, used = [], set()
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            if a in used or b in used:
                continue
            pa, ia, oa, sa = cells[a]
            pb, ib, ob, sb = cells[b]
            if oa == sb and ob == sa:
                swaps.append({"a": {"page": pa, "change": ia, "value": oa}, "b": {"page": pb, "change": ib, "value": ob}})
                used.update((a, b))
    return swaps


def _json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(type(o))

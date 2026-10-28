import hashlib
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from formguard.compare import compare_pages
from formguard.dropbox_sync import content_hash
from formguard.pairing import key, pair
from formguard.pipeline import Config, compare_documents, find_swaps
from formguard.register import features, match_pages, refine, warp
from formguard.render import load_pages

from . import synth

NO_OCR = Config(ocr=False)


def _write(path: Path, img: np.ndarray) -> Path:
    cv2.imwrite(str(path), img)
    return path


def _overlaps(change: dict, box: tuple[int, int, int, int], scale: float) -> bool:
    x, y, w, h = (v * scale for v in box)
    return change["x"] < x + w and x < change["x"] + change["w"] and change["y"] < y + h and y < change["y"] + change["h"]


@pytest.fixture(scope="module")
def original(tmp_path_factory) -> tuple[Path, np.ndarray]:
    vals = synth.values()
    return _write(tmp_path_factory.mktemp("o") / "original.png", synth.draw(vals)), vals


def test_degraded_but_unaltered_copy_is_clean(original, tmp_path):
    path, vals = original
    sub = _write(tmp_path / "copy.png", synth.degrade(synth.draw(vals)))
    rep = compare_documents(path, sub, tmp_path / "out", NO_OCR)
    assert rep["verdict"] == "clean", rep["pages"]
    assert rep["altered_numbers"] == 0


@pytest.mark.parametrize(
    "edit",
    [
        {(3, 2): 417},  # digits replaced
        {(10, 1): 5, (10, 4): 5},  # number shortened (erasure of digits)
        {(17, 0): None},  # digit prepended, as in 26 -> 226
    ],
)
def test_edited_cells_are_found_and_nothing_else(original, tmp_path, edit):
    path, vals = original
    forged = vals.copy()
    for (r, c), v in edit.items():
        forged[r, c] = v if v is not None else int("2" + str(vals[r, c]))
    assert (forged != vals).any()
    sub = _write(tmp_path / "forged.jpg", synth.degrade(synth.draw(forged)))
    rep = compare_documents(path, sub, tmp_path / "out", NO_OCR)
    assert rep["verdict"] == "altered"
    changes = [c for p in rep["pages"] for c in p["changes"] if c["in_table"]]
    scale = 2600 / max(synth.SIZE)
    edited = [synth.cell_box(r, c) for (r, c) in edit]
    for box in edited:
        assert any(_overlaps(c, box, scale) for c in changes), f"missed edit in {box}"
    for c in changes:
        assert any(_overlaps(c, box, scale) for box in edited), f"false alarm at {c}"
    assert (tmp_path / "out" / "report.json").exists()
    assert all((tmp_path / "out" / c["evidence"]).exists() for c in changes)


def test_swapped_values_between_cells(original, tmp_path):
    path, vals = original
    forged = vals.copy()
    forged[5, 1], forged[5, 2] = vals[5, 2], vals[5, 1]
    if forged[5, 1] == vals[5, 1]:
        pytest.skip("random values happen to be equal")
    sub = _write(tmp_path / "swapped.jpg", synth.degrade(synth.draw(forged), angle=-1.5))
    rep = compare_documents(path, sub, tmp_path / "out", NO_OCR)
    assert rep["verdict"] == "altered"
    assert rep["altered_numbers"] == 2


def test_find_swaps_pairs_exchanged_readings():
    pages = [
        {
            "submission_page": 1,
            "changes": [
                {"in_table": True, "read_original": "189", "read_submission": "75"},
                {"in_table": True, "read_original": "26", "read_submission": "226"},
                {"in_table": True, "read_original": "75", "read_submission": "189"},
            ],
        }
    ]
    swaps = find_swaps(pages)
    assert len(swaps) == 1
    assert {swaps[0]["a"]["value"], swaps[0]["b"]["value"]} == {"189", "75"}


def test_identical_file_short_circuits(original, tmp_path):
    path, _ = original
    dup = tmp_path / "dup.png"
    dup.write_bytes(path.read_bytes())
    assert compare_documents(path, dup, None, NO_OCR)["verdict"] == "identical"


def test_pages_are_matched_across_reordering_and_rotation(tmp_path):
    a, b = synth.draw(synth.values(1), seed=1), synth.draw(synth.values(2), seed=2)
    ref = tmp_path / "ref.pdf"
    sub = tmp_path / "sub.pdf"
    for path, imgs in ((ref, [a, b]), (sub, [cv2.rotate(b, cv2.ROTATE_90_CLOCKWISE), a, a])):
        doc = pymupdf.open()
        for img in imgs:
            ok, png = cv2.imencode(".png", img)
            page = doc.new_page(width=img.shape[1] / 2, height=img.shape[0] / 2)
            page.insert_image(page.rect, stream=png.tobytes())
        doc.save(path)
    R, S = load_pages(ref, 1800), load_pages(sub, 1800)
    matches = match_pages([features(p.image) for p in R], [features(p.image) for p in S])
    assert [m.ref_index for m in matches] == [1, 0, 0]
    rep = compare_documents(ref, sub, None, NO_OCR)
    assert rep["verdict"] == "clean"
    assert any("duplicated" in n for n in rep["notes"])


def test_unrelated_page_is_not_compared(tmp_path):
    a = synth.draw(synth.values(1), seed=1)
    other = np.full_like(a, 255)
    cv2.putText(other, "a completely different document", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    rep = compare_documents(_write(tmp_path / "a.png", a), _write(tmp_path / "b.png", other), None, NO_OCR)
    assert rep["verdict"] == "inconclusive"
    assert rep["pages"][0]["original_page"] is None


def test_registration_recovers_geometry():
    page = synth.draw(synth.values())
    sub = synth.degrade(page, angle=3, scale=0.8)
    (m,) = match_pages([features(page)], [features(sub)])
    warped, valid = warp(sub, m.H, page.shape)
    warped, _ = refine(page, warped)
    pc = compare_pages(page, warped, valid)
    assert pc.quality > 0.95
    assert not pc.flagged(0.5, table_only=False)


def test_pairing_keys_survive_naming_differences():
    assert key("IEBC Webuye West.pdf") == key("Webuye-West.pdf")
    assert key("Mt Elgon.pdf") == key("Mt.-Elgon.pdf")
    assert key("Sirisiya.pdf") == key("Sirisia.pdf")
    assert key("Balambala.pdf") == key("Mbalambala.pdf")
    pairs, unpaired = pair([Path("x/Teso North.pdf")], [Path("Teso-North.pdf"), Path("Luanda.pdf")])
    assert [p[1].name for p in pairs] == ["Teso-North.pdf"]
    assert [u.name for u in unpaired] == ["Luanda.pdf"]


def test_dropbox_content_hash(tmp_path):
    data = b"a" * (4 * 1024 * 1024) + b"tail"
    f = tmp_path / "f.bin"
    f.write_bytes(data)
    blocks = hashlib.sha256(data[: 4 * 1024 * 1024]).digest() + hashlib.sha256(b"tail").digest()
    assert content_hash(f) == hashlib.sha256(blocks).hexdigest()

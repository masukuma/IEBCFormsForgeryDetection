import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from formguard import llm
from formguard.affidavit import write as write_affidavit
from formguard.cases import discover, run_manifest
from formguard.pipeline import Config

from . import synth


def _pdf(path: Path, images: list[np.ndarray]) -> Path:
    doc = pymupdf.open()
    for img in images:
        ok, png = cv2.imencode(".png", img)
        page = doc.new_page(width=img.shape[1] / 2, height=img.shape[0] / 2)
        page.insert_image(page.rect, stream=png.tobytes())
    doc.save(path)
    return path


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Three originals, and a 'questioned affidavit' with a faithful copy and a forged copy of #2."""
    root = tmp_path_factory.mktemp("case")
    (root / "ref").mkdir()
    vals = [synth.values(s) for s in (11, 12, 13)]
    for i, v in enumerate(vals):
        _pdf(root / "ref" / f"original_{i}.pdf", [synth.draw(v, seed=20 + i, capture=i)])
    forged = vals[1].copy()
    forged[4, 1] = int("1" + str(vals[1][4, 1]))  # prefix a digit, as in 26 -> 126
    cover = np.full((synth.SIZE[0], synth.SIZE[1]), 255, np.uint8)
    cv2.putText(cover, "AFFIDAVIT", (300, 600), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5)
    pages = [
        cover,
        synth.degrade(synth.draw(vals[1], seed=21, capture=1)),  # faithful copy of the original's image
        synth.degrade(synth.draw(forged, seed=21, capture=1), angle=-2),  # that image, with one figure edited
        synth.degrade(synth.draw(vals[1], seed=21, capture=99), angle=1),  # a separate photograph of the same sheet
    ]
    _pdf(root / "questioned.pdf", pages)
    return root


def test_discover_finds_the_original_of_each_page(workspace):
    m = discover(str(workspace / "questioned.pdf#pages=1-4"), sorted((workspace / "ref").glob("*.pdf")), workspace / "cache", base=workspace)
    assert len(m["cases"]) == 1
    case = m["cases"][0]
    assert case["reference"] == "ref/original_1.pdf"
    assert [e["source"] for e in case["exhibits"]] == [f"questioned.pdf#page={n}" for n in (2, 3, 4)]
    assert [u["page"] for u in m["unmatched_pages"]] == [1]  # the cover page matches nothing
    (workspace / "manifest.json").write_text(json.dumps({"matter": {"cause": "TEST"}, **m}))


def test_run_manifest_separates_faithful_and_forged_copies(workspace):
    results = run_manifest(workspace / "manifest.json", workspace / "runs", Config(ocr=False))
    (case,) = results
    faithful, forged, other_photo = case["exhibit_results"]
    assert faithful["verdict"] == "clean"
    assert forged["verdict"] == "altered" and forged["altered_numbers"] == 1
    # A different photograph differs outside the form (its surround) but not in the results.
    assert other_photo["verdict"] in ("clean", "review") and other_photo["altered_numbers"] == 0
    prov = {k: r["pages"][0]["provenance"] for k, r in (("faithful", faithful), ("forged", forged), ("other", other_photo))}
    assert all(p["marks_reproduced"] > 0.9 for p in prov.values())
    assert prov["faithful"]["finding"] == prov["forged"]["finding"] == "same_capture"  # made from original #2's image
    assert prov["other"]["finding"] == "same_sheet"  # same marks, but a different capture's surround
    assert case["reference_integrity"]["sha256"]


def test_affidavit_is_written_from_results(workspace, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    md, docx = write_affidavit(workspace / "runs" / "results.json", {"name": "TEST DEPONENT"}, workspace / "affidavit")
    text = md.read_text()
    assert docx.exists() and docx.stat().st_size > 10_000
    assert "TEST DEPONENT" in text and "THAT" in text
    assert "one figure differs" in text and "agrees with the original" in text
    assert "drafted with" not in text  # no model used without a key
    from formguard.llm import plain

    paragraphs = [line for line in text.splitlines() if "THAT" in line]
    assert paragraphs and all(plain(p) for p in paragraphs)  # the templates obey the same style rules as model drafts


def test_llm_draft_rejects_invented_numbers(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test")
    facts = {"votes_original": 60, "votes_questioned": 32}
    fallback = "THAT the original reads 60 and the exhibit reads 32."
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "THAT the exhibit shows 32 votes instead of 60.")
    assert llm.draft("x", facts, fallback)[1] != "template"
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "THAT 28 votes were moved, out of 345 cast.")
    assert llm.draft("x", facts, fallback) == (fallback, "template")


def test_llm_draft_rejects_rhetoric(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test")
    facts = {"votes_original": 60, "votes_questioned": 32}
    fallback = "THAT the original reads 60 and the exhibit reads 32."
    for text in (
        "THAT the exhibit clearly shows 32 votes instead of 60.",
        "THAT the exhibit — a printout — shows 32 votes, not 60.",
        "THAT, notably, the exhibit shows 32 votes and the original 60.",
        "THAT the exhibit shows **32** votes and the original 60.",
    ):
        monkeypatch.setattr(llm, "chat", lambda *a, _t=text, **k: _t)
        assert llm.draft("x", facts, fallback) == (fallback, "template"), text


def test_model_ladder_escalates_only_on_invalid_output(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test")
    calls = []

    def fake_chat(model, messages, fmt=None, timeout=300, task=""):
        calls.append(model)
        llm.USAGE.append({"model": model, "task": task})
        if model == llm.VISION_LADDER[0]:
            return json.dumps({"original": "060", "questioned": "", "legible": True})  # empty reading: invalid
        return json.dumps({"original": "060", "questioned": "032", "legible": True})

    monkeypatch.setattr(llm, "chat", fake_chat)
    r = llm.read_figures(b"png")
    assert calls == llm.VISION_LADDER[:2]
    assert (r["original"], r["questioned"], r["by"]) == ("060", "032", llm.VISION_LADDER[1])

    calls.clear()
    monkeypatch.setattr(llm, "chat", lambda model, *a, **k: calls.append(model) or json.dumps({"original": "284", "questioned": "284", "legible": True}))
    assert llm.read_figures(b"png")["by"] == llm.VISION_LADDER[0] and calls == llm.VISION_LADDER[:1]  # cheapest answer accepted

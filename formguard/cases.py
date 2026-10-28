"""Case manifests: which exhibits are checked against which original, and what was claimed.

A manifest is JSON:

    {
      "matter": {"court": "...", "cause": "...", "parties": [...]},
      "template": ["ref/34A/<another form>.PDF", ...],
      "cases": [
        {
          "id": "AO-04",
          "title": "Ngoron Primary School polling station 1 of 1 (Tiaty, Baringo)",
          "reference": "ref/34A/1_34_A_030157078100401_....PDF",
          "claim": {"by": "...", "where": "para 17(4)", "text": "...", "figures": {...}},
          "exhibits": [
            {"label": "Annexure p. 50", "caption": "IEBC PORTAL", "source": "docs/affidavit.pdf#page=50"},
            {"label": "Annexure p. 75", "caption": "AGENT COPY",  "source": "docs/affidavit.pdf#page=75"}
          ],
          "readings": {"p75_change03": {"original": "060", "questioned": "032", "by": "examiner"}}
        }
      ]
    }

Paths are relative to the manifest. ``template`` lists other forms of the originals' type
(optional); they let the comparison tell the printed form from the entries written on it. ``discover`` writes a draft manifest by retrieving the
original of every questioned page from a pool of candidates; an examiner then fills in the
claim and checks the captions.
"""
from __future__ import annotations

import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from .integrity import file_facts, pdf_signatures
from .pipeline import Config, compare_documents
from .render import load_pages, parse_source
from .retrieve import ReferenceIndex


def _resolve(base: Path, source: str) -> str:
    path, pages = parse_source(source)
    path = path if path.is_absolute() else base / path
    return f"{path}#pages={','.join(map(str, pages))}" if pages else str(path)


def run_case(case: dict, base: Path, out_dir: Path, cfg: Config) -> dict:
    """Compare every exhibit of one case with its reference and record the reference's integrity."""
    ref = _resolve(base, case["reference"])
    case_dir = out_dir / case["id"]
    result = dict(case)
    result["reference_integrity"] = {**file_facts(ref), "signatures": pdf_signatures(ref)}
    result["exhibit_results"] = []
    for ex in case["exhibits"]:
        src = _resolve(base, ex["source"])
        slug = re.sub(r"[^A-Za-z0-9]+", "-", ex["label"]).strip("-").lower()
        rep = compare_documents(ref, src, case_dir / slug, cfg)
        rep["exhibit"] = ex
        rep["evidence_dir"] = str((case_dir / slug).relative_to(out_dir))
        rep["exhibit_integrity"] = file_facts(src)
        result["exhibit_results"].append(rep)
    case_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, default=str)
    (case_dir / "case.json").write_text(text)
    return json.loads(text)  # plain JSON types only: results cross process boundaries


def _run(args):
    case, base, out_dir, cfg = args
    try:
        return run_case(case, base, out_dir, cfg)
    except Exception as e:
        return {**case, "error": f"{type(e).__name__}: {e}"}


def run_manifest(manifest_path: Path, out_dir: Path, cfg: Config = Config(), workers: int = 1) -> list[dict]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    if manifest.get("template") and not cfg.template:
        cfg = replace(cfg, template=tuple(_resolve(base, t) for t in manifest["template"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, base, out_dir, cfg) for c in manifest["cases"]]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_run, jobs))
    else:
        results = [_run(j) for j in jobs]
    (out_dir / "results.json").write_text(json.dumps({"matter": manifest.get("matter", {}), "cases": results}, indent=2, default=str))
    return results


_INDEX: ReferenceIndex | None = None  # shared with forked workers instead of pickled per task


def _query(args):
    page_no, image = args
    return page_no, _INDEX.query(image)


def discover(questioned: str, candidates: list[Path], cache_dir: Path, workers: int = 1, base: Path | None = None) -> dict:
    """Draft manifest: retrieve the original of each questioned page and group pages by original."""
    global _INDEX
    _INDEX = ReferenceIndex(candidates, cache_dir)
    _INDEX.feats  # build or load the cache once, before forking
    pages = load_pages(questioned, 1800)
    jobs = [(p.index + 1, p.image) for p in pages]
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            hits = list(pool.map(_query, jobs))
    else:
        hits = [_query(j) for j in jobs]

    path, _ = parse_source(questioned)
    rel = lambda p: str(Path(p).relative_to(base)) if base else str(p)
    groups: dict[str, dict] = {}
    unmatched = []
    for page_no, r in hits:
        if not r.accepted:
            unmatched.append({"page": page_no, "best": rel(r.reference) if r.reference else None, "inliers": r.inliers, "runner_up_inliers": r.runner_up_inliers})
            continue
        code = re.search(r"_A_(\d{15})", r.reference.name)
        key = str(r.reference)
        g = groups.setdefault(key, {"id": "", "title": "", "station_code": code.group(1) if code else None, "reference": rel(r.reference), "claim": {}, "exhibits": []})
        g["exhibits"].append({"label": f"Page {page_no}", "caption": "", "source": f"{rel(path)}#page={page_no}", "retrieval": {"inliers": r.inliers, "runner_up_inliers": r.runner_up_inliers}})
    cases = sorted(groups.values(), key=lambda c: int(re.search(r"page=(\d+)", c["exhibits"][0]["source"]).group(1)))
    for i, c in enumerate(cases, 1):
        c["id"] = f"C{i:02d}"
    # Forms of the same type that are not among the originals: they show the printed form.
    used = {Path(k) for k in groups}
    spare = [c for c in candidates if c not in used]
    template = [rel(c) for c in spare[:: max(1, len(spare) // 12)][:12]]
    return {"template": template, "cases": cases, "unmatched_pages": unmatched}

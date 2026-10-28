"""Read the figures in every evidence image of a case run with the vision-model ladder."""
from __future__ import annotations

import json
from pathlib import Path

from . import llm


def read_all(results_path: Path, confirm: bool = False, existing: dict | None = None, save_to: Path | None = None) -> dict:
    """{case id: {"p75_change03": {"original": "060", "questioned": "032", "by": "gemma4:31b", ...}}}.

    Entries already in ``existing`` (e.g. typed in by the examiner) are kept and not re-read.
    With ``save_to``, progress is written after each case, so an interrupted run resumes.
    """
    results = json.loads(Path(results_path).read_text())
    run_dir = Path(results_path).parent
    out: dict = json.loads(json.dumps(existing or {}))
    for case in results["cases"]:
        for rep in case.get("exhibit_results", []):
            ev = run_dir / rep["evidence_dir"]
            for page in rep.get("pages", []):
                for c in page.get("changes", []):
                    # Figures only: marks on blank paper and areas under added marks are not read.
                    if not c.get("in_table") or not c.get("on_figure", True) or not c.get("evidence"):
                        continue
                    key = c["evidence"].rsplit(".", 1)[0]
                    if key in out.get(case["id"], {}):
                        continue
                    r = llm.read_figures((ev / c["evidence"]).read_bytes(), confirm=confirm)
                    out.setdefault(case["id"], {})[key] = r
        if save_to is not None:
            Path(save_to).write_text(json.dumps(out, indent=2))
    out["_usage"] = llm.usage_summary()
    return out

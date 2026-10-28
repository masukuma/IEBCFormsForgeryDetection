"""Markdown for the doc's summary table and per-station findings, written with the same sentence
functions as the DOCX (formguard.affidavit), so the two cannot drift apart.

usage: doc_md.py results.json readings.json start_para out_dir [manifest.json]
The manifest, if given, supplies the claim of each case (claims are edited there after a run).
writes summary.md, stations/<id>.md (numbered continuously), numbering.json, marks.json
"""
import json
import sys
from pathlib import Path

from formguard.affidavit import PROVENANCE_SHORT, _claim, _count, _exhibit_sentence, _opinion, _plain_title, _sig_sentence, _that, annexure_marks, exhibit_facts

results = json.load(open(sys.argv[1]))
readings = json.load(open(sys.argv[2])) if Path(sys.argv[2]).exists() else {}
n = int(sys.argv[3])
out = Path(sys.argv[4])
claims = {c["id"]: c.get("claim") or {} for c in json.load(open(sys.argv[5]))["cases"]} if len(sys.argv) > 5 else {}
(out / "stations").mkdir(parents=True, exist_ok=True)

marks = annexure_marks(results, "[XX]", readings)
cases = [c for c in results["cases"] if "error" not in c]


def bold_that(p: str) -> str:
    return "**THAT**" + p[4:] if p.startswith("THAT") else p


rows = ["| Polling station | Annexed page | Caption | Result | How the page was made |", "|---|---|---|---|---|"]
numbering = {}
pages = {}  # case id -> {exhibit label: paragraph number}
for c in cases:
    paras = []
    claim = _claim(claims.get(c["id"], c.get("claim") or {}))
    if claim:
        paras.append(claim)
    paras.append(_that(_sig_sentence(c)))
    for rep in c["exhibit_results"]:
        f = exhibit_facts(c, rep, readings.get(c["id"], {}))
        pages.setdefault(c["id"], {})[f["label"]] = n + len(paras)
        paras.append(_exhibit_sentence(c, f, marks.get(c["id"])))
        op = _opinion(c, f)
        if op:
            paras.append(op)
        changed = [x for x in f["differences_in_results"] if not (x["original_reads"] and x["original_reads"] == x["questioned_reads"])]
        result = "not matched" if not f["matched"] else ("agrees with the original" if not changed else _count(len(changed), "figure") + (" differs" if len(changed) == 1 else " differ"))
        made = PROVENANCE_SHORT.get(f["provenance_finding"] or "", "not determined") if changed else ""
        station = _plain_title(c["title"]).split(",")[0]
        rows.append(f"| {station} | {f['label']} | {f['caption'] or 'none'} | {result} | {made} |")
    lines = [f"### {_plain_title(c['title'])}", ""]
    first = n
    for p in paras:
        lines.append(f"{n}. {bold_that(p)}")
        n += 1
    numbering[c["id"]] = [first, n - 1]
    (out / "stations" / f"{c['id']}.md").write_text("\n".join(lines) + "\n")

(out / "summary.md").write_text("\n".join(rows) + "\n")
json.dump({"numbering": numbering, "pages": pages, "next": n}, open(out / "numbering.json", "w"), indent=1)
json.dump(marks, open(out / "marks.json", "w"), indent=1)
print("stations", len(cases), "next paragraph", n, "annexures", len(marks))

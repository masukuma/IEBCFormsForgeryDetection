"""Batch summary: one JSON file and one self-contained HTML page linking to the evidence."""
from __future__ import annotations

import html
import json
from pathlib import Path

ORDER = {"altered": 0, "review": 1, "inconclusive": 2, "error": 3, "no_original": 4, "clean": 5, "identical": 6}
COLORS = {"altered": "#c62828", "review": "#ef6c00", "inconclusive": "#6d4c41", "error": "#6d4c41", "no_original": "#546e7a", "clean": "#2e7d32", "identical": "#2e7d32"}


def write_summary(results: list[dict], out_dir: Path) -> Path:
    results = sorted(results, key=lambda r: (ORDER.get(r["verdict"], 9), -r.get("altered_numbers", 0), r["name"]))
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    rows = []
    for r in results:
        name = html.escape(r["name"])
        link = f'<a href="{html.escape(r["dir"])}/report.json">{name}</a>' if r.get("dir") else name
        detail = []
        for p in r.get("pages", []):
            for c in p.get("changes", []):
                if not c["in_table"]:
                    continue
                read = ""
                if c.get("value_changed") is False:
                    read = f'{html.escape(c["read_original"])}, re-typeset in place'
                elif c.get("read_original") or c.get("read_submission"):
                    read = f'<b>{html.escape(c.get("read_original") or "?")} &rarr; {html.escape(c.get("read_submission") or "?")}</b>'
                img = f'<img loading="lazy" src="{html.escape(r["dir"])}/{html.escape(c["evidence"])}">' if c.get("evidence") else ""
                detail.append(f'<figure>{img}<figcaption>p{p["submission_page"]} &middot; {c["kind"]} &middot; sim {c["similarity"]:.2f} {read}</figcaption></figure>')
        notes = "; ".join(html.escape(n) for n in r.get("notes", []))
        rows.append(
            f'<tr><td>{link}</td><td><span class="v" style="background:{COLORS.get(r["verdict"], "#555")}">{r["verdict"]}</span></td>'
            f'<td class="n">{r.get("altered_numbers", "")}</td><td class="n">{r.get("values_changed", "")}</td><td>{notes}</td></tr>'
            + (f'<tr class="ev"><td colspan="5"><details><summary>{len(detail)} changed numbers</summary><div class="grid">{"".join(detail)}</div></details></td></tr>' if detail else "")
        )
    tiles = " ".join(f'<span class="v" style="background:{COLORS.get(k, "#555")}">{k}: {v}</span>' for k, v in sorted(counts.items(), key=lambda kv: ORDER.get(kv[0], 9)))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Form Alteration Scan</title><style>
body{{font:14px/1.4 system-ui,sans-serif;margin:16px;color:#222;background:#fff}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}}
.v{{color:#fff;border-radius:4px;padding:2px 8px;font-size:12px;white-space:nowrap}}.n{{text-align:right}}
.grid{{display:flex;flex-wrap:wrap;gap:8px}}figure{{margin:0;max-width:420px}}figure img{{max-width:100%;border:1px solid #ccc}}
figcaption{{font-size:12px;color:#555}}tr.ev td{{border-bottom:2px solid #bbb}}
</style></head><body><h1>Form alteration scan</h1><p>{tiles}</p>
<table><tr><th>Document</th><th>Verdict</th><th>Altered cells</th><th>Values changed<br><small>(OCR)</small></th><th>Notes</th></tr>{"".join(rows)}</table></body></html>"""
    path = out_dir / "index.html"
    path.write_text(page)
    return path

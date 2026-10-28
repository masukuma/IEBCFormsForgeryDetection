"""Command line entry point.

    python -m formguard scan --originals DIR --submissions DIR --out DIR [--workers N]
    python -m formguard compare ORIGINAL SUBMISSION --out DIR
    python -m formguard sync --remote /Work/Kenya/2022/Forms/34B --local data/originals

Case work (exhibits annexed to affidavits, checked against the originals they claim to show):
    python -m formguard discover "affidavit.pdf#pages=46-130" --candidates ref/34A --out manifest.json
    python -m formguard cases manifest.json --out runs/
    python -m formguard affidavit runs/results.json --deponent deponent.json --out affidavit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .pairing import documents, key, pair
from .pipeline import Config, compare_documents
from .report import write_summary


def _job(args: tuple[Path, Path, Path, Config]) -> dict:
    original, submission, out, cfg = args
    name = key(submission.name).replace(" ", "-")
    try:
        rep = compare_documents(original, submission, out / name, cfg)
    except Exception as e:  # one bad file must not sink a batch of hundreds
        return {"name": name, "verdict": "error", "dir": name, "notes": [f"{type(e).__name__}: {e}"], "trace": traceback.format_exc()}
    rep["name"] = name
    rep["dir"] = name
    return rep


def _forms(folder: str | None) -> tuple[str, ...]:
    if not folder:
        return ()
    return tuple(str(p) for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in (".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"))


def _config(a: argparse.Namespace) -> Config:
    return Config(long_side=a.long_side, flag_below=a.flag_below, ocr=not a.no_ocr, template=_forms(a.template))


def cmd_scan(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs, unpaired = pair(documents(a.originals), documents(a.submissions))
    print(f"{len(pairs)} submissions have an original; {len(unpaired)} do not", file=sys.stderr)
    cfg = _config(a)
    results = [{"name": key(s.name).replace(" ", "-"), "verdict": "no_original", "submission": str(s), "notes": ["no original found to compare against"]} for s in unpaired]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(_job, (o, s, out, cfg)) for o, s in pairs]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            results.append(r)
            print(f"[{i}/{len(pairs)}] {r['name']}: {r['verdict']} ({r.get('altered_numbers', 0)} numbers, {r.get('seconds', '?')}s)", file=sys.stderr)
    index = write_summary(results, out)
    print(index)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    rep = compare_documents(Path(a.original), Path(a.submission), Path(a.out), _config(a))
    print(json.dumps({k: rep[k] for k in ("verdict", "altered_numbers", "notes") if k in rep}, indent=2))
    return 0 if rep["verdict"] in ("clean", "identical") else 1


def cmd_sync(a: argparse.Namespace) -> int:
    from .dropbox_sync import sync

    token = a.token or os.environ.get("DROPBOX_TOKEN")
    if not token:
        print("set DROPBOX_TOKEN or pass --token", file=sys.stderr)
        return 2
    n = sync(token, a.remote, Path(a.local))
    print(f"{n} file(s) downloaded", file=sys.stderr)
    return 0


def cmd_discover(a: argparse.Namespace) -> int:
    from .cases import discover

    cands = sorted(p for p in Path(a.candidates).rglob("*") if p.suffix.lower() in (".pdf", ".jpg", ".jpeg", ".png"))
    out = Path(a.out)
    m = discover(a.questioned, cands, Path(a.cache or out.parent / ".formguard-cache"), workers=a.workers, base=out.parent.resolve())
    out.write_text(json.dumps(m, indent=2))
    print(f"{len(m['cases'])} case(s), {len(m['unmatched_pages'])} unmatched page(s) -> {out}", file=sys.stderr)
    return 0


def cmd_cases(a: argparse.Namespace) -> int:
    from .cases import run_manifest

    results = run_manifest(Path(a.manifest), Path(a.out), _config(a), workers=a.workers)
    for r in results:
        verdicts = [x.get("verdict") for x in r.get("exhibit_results", [])]
        print(f"{r['id']}: {r.get('title', '')} -> {verdicts or r.get('error')}", file=sys.stderr)
    return 0


def cmd_readings(a: argparse.Namespace) -> int:
    from . import llm
    from .readings import read_all

    if not llm.available():
        print("OLLAMA_API_KEY is not set", file=sys.stderr)
        return 2
    existing = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else None
    out = read_all(Path(a.results), confirm=a.confirm, existing=existing)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["_usage"], indent=2), file=sys.stderr)
    return 0


def cmd_affidavit(a: argparse.Namespace) -> int:
    from .affidavit import write

    deponent = json.loads(Path(a.deponent).read_text()) if a.deponent else {}
    md, docx = write(Path(a.results), deponent, Path(a.out), Path(a.readings) if a.readings else None, use_llm=not a.no_llm)
    from . import llm

    if llm.USAGE:
        print(json.dumps(llm.usage_summary(), indent=2), file=sys.stderr)
    print(f"{md}\n{docx}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="formguard", description="Detect alterations in submitted copies of election forms.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--long-side", type=int, default=2600, help="working resolution, px on the page's long side")
        sp.add_argument("--flag-below", type=float, default=Config.flag_below, help="glyph similarity below which a change is an alteration")
        sp.add_argument("--no-ocr", action="store_true", help="skip reading flagged numbers with tesseract")
        sp.add_argument("--template", help="folder of other forms of the originals' type, to recognise their print (photographed copies)")

    s = sub.add_parser("scan", help="compare every submission with its original")
    s.add_argument("--originals", required=True)
    s.add_argument("--submissions", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    common(s)
    s.set_defaults(fn=cmd_scan)

    c = sub.add_parser("compare", help="compare one submission with one original")
    c.add_argument("original")
    c.add_argument("submission")
    c.add_argument("--out", required=True)
    common(c)
    c.set_defaults(fn=cmd_compare)

    y = sub.add_parser("sync", help="mirror a Dropbox folder locally (incremental, by content hash)")
    y.add_argument("--remote", required=True)
    y.add_argument("--local", required=True)
    y.add_argument("--token")
    y.set_defaults(fn=cmd_sync)

    d = sub.add_parser("discover", help="find the original of each questioned page; write a draft case manifest")
    d.add_argument("questioned", help='questioned document, optionally with pages: "file.pdf#pages=46-130"')
    d.add_argument("--candidates", required=True, help="folder of candidate originals")
    d.add_argument("--out", required=True, help="manifest JSON to write")
    d.add_argument("--cache", help="feature cache folder (default: next to the manifest)")
    d.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    d.set_defaults(fn=cmd_discover)

    k = sub.add_parser("cases", help="run every case in a manifest")
    k.add_argument("manifest")
    k.add_argument("--out", required=True)
    k.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    common(k)
    k.set_defaults(fn=cmd_cases)

    r = sub.add_parser("readings", help="read flagged figures with Ollama Cloud vision models (cheapest first)")
    r.add_argument("results", help="results.json written by 'cases'")
    r.add_argument("--out", required=True, help="readings JSON (existing entries are kept)")
    r.add_argument("--confirm", action="store_true", help="also get an independent second reading from the next model up")
    r.set_defaults(fn=cmd_readings)

    f = sub.add_parser("affidavit", help="draft an affidavit (DOCX + Markdown) from case results")
    f.add_argument("results", help="results.json written by 'cases'")
    f.add_argument("--deponent", help="JSON with name, address, profession, instructed_by")
    f.add_argument("--readings", help="JSON of examiner/model readings of the flagged figures")
    f.add_argument("--out", required=True, help="output path without extension")
    f.add_argument("--no-llm", action="store_true", help="do not use Ollama Cloud even if OLLAMA_API_KEY is set")
    f.set_defaults(fn=cmd_affidavit)

    a = p.parse_args(argv)
    return a.fn(a)

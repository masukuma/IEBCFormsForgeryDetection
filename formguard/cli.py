"""Command line entry point.

    python -m formguard scan --originals DIR --submissions DIR --out DIR [--workers N]
    python -m formguard compare ORIGINAL SUBMISSION --out DIR
    python -m formguard sync --remote /Work/Kenya/2022/Forms/34B --local data/originals
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


def _config(a: argparse.Namespace) -> Config:
    return Config(long_side=a.long_side, flag_below=a.flag_below, ocr=not a.no_ocr)


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="formguard", description="Detect alterations in submitted copies of election forms.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--long-side", type=int, default=2600, help="working resolution, px on the page's long side")
        sp.add_argument("--flag-below", type=float, default=0.5, help="glyph similarity below which a change is an alteration")
        sp.add_argument("--no-ocr", action="store_true", help="skip reading flagged numbers with tesseract")

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

    a = p.parse_args(argv)
    return a.fn(a)

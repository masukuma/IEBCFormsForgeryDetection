"""Pair each submitted document with its original, by constituency name.

File names on the two sides differ in spacing, case and prefixes ("IEBC Webuye West.pdf" vs
"Webuye-West.pdf"), so names are reduced to a canonical key. Pairing is only a candidate:
the page matcher still has to find strong geometric agreement before anything is compared,
so a wrong pairing surfaces as "no matching pages", never as a false alteration.
"""
from __future__ import annotations

import re
from pathlib import Path

from .render import IMAGE_EXTS

DOC_EXTS = {".pdf"} | IMAGE_EXTS
STRIP_PREFIXES = ("iebc ",)
ALIASES: dict[str, str] = {
    # Spelling variants seen between the 2022 originals and the WhistleBlower submissions.
    "sirisiya": "sirisia",
    "balambala": "mbalambala",
}


def key(name: str) -> str:
    s = Path(name).stem.lower()
    s = re.sub(r"[-_.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for p in STRIP_PREFIXES:
        if s.startswith(p):
            s = s[len(p) :]
    return ALIASES.get(s, s)


def documents(root: str | Path) -> list[Path]:
    """All documents under root, skipping derived folders (page extracts, labelling sets)."""
    skip = {"extracted", "for labling", "for labeling", "website"}
    out = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in DOC_EXTS and not skip & {q.lower() for q in p.relative_to(root).parts[:-1]}:
            out.append(p)
    return out


def pair(originals: list[Path], submissions: list[Path]) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Returns ([(original, submission)], [submissions without an original])."""
    index: dict[str, Path] = {}
    for o in originals:
        index.setdefault(key(o.name), o)
    pairs, unpaired = [], []
    for s in submissions:
        o = index.get(key(s.name))
        if o is None:
            unpaired.append(s)
        else:
            pairs.append((o, s))
    return pairs, unpaired

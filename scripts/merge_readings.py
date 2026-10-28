"""Combine the examiner's readings (field names, values) with the vision model's independent
readings, and record whether they agree.

usage: merge_readings.py examiner.json model.json out.json
The model reads every number in a panel, top to bottom; the examiner's value agrees with it when it
is one of those numbers (leading zeros ignored), or when the model split it ("3,45" for 345), on
each side. The model's figure closest to the examiner's is recorded as original_figure and
questioned_figure, so a disagreement can be quoted for the figure in question. Entries marked
same_as (a second region of a number already read) are carried over but not counted.
"""
import json
import sys

ex = json.load(open(sys.argv[1]))
model = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
out, stats = {}, {"agree": 0, "disagree": 0, "model_unread": 0, "not_numbers": 0}


def nums(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def bare(x):
    return x.lstrip("0") or "0"


def agrees(value, read):
    ns = nums(read)
    return bare(value) in {bare(n) for n in ns} or bare("".join(ns)) == bare(value)


def edits(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def common(a, b):
    """Length of the longest common subsequence."""
    row = [0] * (len(b) + 1)
    for ca in a:
        prev = 0
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], prev + 1 if ca == cb else max(row[j], row[j - 1])
    return row[-1]


def closest(value, read):
    ns = nums(read)
    return min(ns, key=lambda n: (-common(n, value), edits(n, value))) if ns else None


for cid, figs in ex.items():
    for stem, r in figs.items():
        r = dict(r)
        m = model.get(cid, {}).get(stem) or {}
        if m.get("original") is not None:
            r["model"] = {"original": m["original"], "questioned": m["questioned"], "by": m.get("by")}
            if r["original"].isdigit() and r["questioned"].isdigit():
                r["model_agrees"] = agrees(r["original"], m["original"]) and agrees(r["questioned"], m["questioned"])
                r["model"]["original_figure"] = closest(r["original"], m["original"])
                r["model"]["questioned_figure"] = closest(r["questioned"], m["questioned"])
                if not r.get("same_as"):
                    stats["agree" if r["model_agrees"] else "disagree"] += 1
            else:
                r["model_agrees"] = nums(m["original"]) == nums(m["questioned"]) if r["original"] == r["questioned"] else None
                stats["not_numbers"] += 1
        elif r["original"].isdigit() and not r.get("same_as"):
            stats["model_unread"] += 1
        out.setdefault(cid, {})[stem] = r
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(stats)
for cid, figs in out.items():
    for stem, r in figs.items():
        if r.get("model_agrees") is False and not r.get("same_as"):
            print(" ", cid, stem, f"{r['original']} -> {r['questioned']}", "| model", f"{r['model']['original_figure']} -> {r['model']['questioned_figure']}",
                  f"({r['model']['original']} -> {r['model']['questioned']}, {r['model']['by']})")

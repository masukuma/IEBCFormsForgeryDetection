"""Combine the examiner's readings (field names, values) with the vision model's independent
readings, and record whether they agree.

usage: merge_readings.py examiner.json model.json out.json
The model reads every number in a panel, top to bottom; the examiner's value agrees with it when it
is one of those numbers (leading zeros ignored) on each side.
"""
import json
import sys

ex = json.load(open(sys.argv[1]))
model = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
out, stats = {}, {"agree": 0, "disagree": 0, "model_unread": 0, "not_numbers": 0}


def nums(s):
    return [x.strip().lstrip("0") or "0" for x in (s or "").split(",") if x.strip()]


for cid, figs in ex.items():
    for stem, r in figs.items():
        r = dict(r)
        m = model.get(cid, {}).get(stem) or {}
        if m.get("original") is not None:
            r["model"] = {"original": m["original"], "questioned": m["questioned"], "by": m.get("by")}
            if r["original"].isdigit() and r["questioned"].isdigit():
                o, q = r["original"].lstrip("0") or "0", r["questioned"].lstrip("0") or "0"
                r["model_agrees"] = o in nums(m["original"]) and q in nums(m["questioned"])
                stats["agree" if r["model_agrees"] else "disagree"] += 1
            else:
                r["model_agrees"] = nums(m["original"]) == nums(m["questioned"]) if r["original"] == r["questioned"] else None
                stats["not_numbers"] += 1
        elif r["original"].isdigit():
            stats["model_unread"] += 1
        out.setdefault(cid, {})[stem] = r
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(stats)

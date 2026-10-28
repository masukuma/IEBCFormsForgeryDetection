"""Carry readings keyed by one run's evidence names over to another run, matching each figure by its
box on the original (evidence numbering shifts between runs).

usage: remap.py old_results.json new_results.json readings.json out.json
prints the new figures (in table, on an entry) that have no reading
"""
import json
import sys


def boxes(results):
    out = {}
    for c in results["cases"]:
        for rep in c.get("exhibit_results", []):
            for p in rep.get("pages", []):
                for x in p.get("changes", []):
                    if x.get("evidence"):
                        out[(c["id"], x["evidence"].rsplit(".", 1)[0])] = (p["submission_page"], x["x"], x["y"], x["w"], x["h"], x["in_table"] and x.get("on_figure", True))
    return out


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter or 1)


old, new = boxes(json.load(open(sys.argv[1]))), boxes(json.load(open(sys.argv[2])))
readings = json.load(open(sys.argv[3]))
out, missing = {}, []
for (cid, stem), (page, *box, fig) in new.items():
    best, score = None, 0.0
    for (ocid, ostem), (opage, *obox, _) in old.items():
        if ocid == cid and opage == page:
            s = iou(box, obox)
            if s > score:
                best, score = ostem, s
    r = readings.get(cid, {}).get(best) if best and score >= 0.4 else None
    if r is not None:
        r = dict(r)
        if r.get("same_as"):  # point at the new name of the figure it duplicates
            target = next((s for (c2, s), v in new.items() if c2 == cid and old.get((cid, r["same_as"])) and v[0] == page and iou(v[1:5], old[(cid, r["same_as"])][1:5]) >= 0.4), None)
            r["same_as"] = target or r["same_as"]
        out.setdefault(cid, {})[stem] = r
    elif fig:
        missing.append(f"{cid} {stem} page {page} box {box}")
json.dump(out, open(sys.argv[4], "w"), indent=1)
print(len(missing), "figures without a reading")
print("\n".join(missing))

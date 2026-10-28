"""Turn case results into a draft expert affidavit (DOCX and Markdown) with annexed evidence.

Every statement of fact in the document is generated from the measured results: hashes,
signature checks, match strengths, similarity scores, the cells that differ and their readings.
Narrative paragraphs come from fixed templates; if an Ollama Cloud key is configured a language
model may redraft them, under the guard in ``llm.draft`` (no number may appear that is not in the
facts). Placeholders in [SQUARE BRACKETS] are for the deponent, who must verify every paragraph
before swearing it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import llm

ALTERED_KINDS = {"substitution": "different figure", "addition": "mark added", "erasure": "mark removed"}


@dataclass
class Doc:
    """Minimal document model rendered to both DOCX and Markdown."""

    blocks: list[tuple] = field(default_factory=list)
    para_no: int = 0

    def heading(self, text: str, level: int = 1):
        self.blocks.append(("heading", text, level))

    def center(self, text: str, bold: bool = False):
        self.blocks.append(("center", text, bold))

    def text(self, text: str):
        self.blocks.append(("text", text))

    def para(self, text: str, source: str = "template") -> int:
        self.para_no += 1
        self.blocks.append(("para", self.para_no, text, source))
        return self.para_no

    def sub(self, letter: str, text: str):
        self.blocks.append(("sub", letter, text))

    def table(self, header: list[str], rows: list[list[str]]):
        self.blocks.append(("table", header, rows))

    def image(self, path: Path, caption: str, width_in: float = 6.0):
        self.blocks.append(("image", Path(path), caption, width_in))

    def page_break(self):
        self.blocks.append(("break",))


# ---------------------------------------------------------------------------------------------
# facts

def _changes(rep: dict) -> list[dict]:
    return [c | {"page": p["submission_page"]} for p in rep.get("pages", []) for c in p.get("changes", []) if c.get("in_table")]


def _prov(rep: dict) -> dict:
    provs = [p.get("provenance") for p in rep.get("pages", []) if p.get("provenance")]
    return provs[0] if provs else {}


def exhibit_facts(case: dict, rep: dict, readings: dict) -> dict:
    ex = rep["exhibit"]
    prov = _prov(rep)
    bg = prov.get("background", {})
    changes = []
    for c in _changes(rep):
        key = c["evidence"].rsplit(".", 1)[0] if c.get("evidence") else None
        r = readings.get(key, {}) if key else {}
        changes.append(
            {
                "evidence": c.get("evidence"),
                "kind": ALTERED_KINDS.get(c["kind"], c["kind"]),
                "similarity": c["similarity"],
                "original_reads": r.get("original") or c.get("read_original") or None,
                "questioned_reads": r.get("questioned") or c.get("read_submission") or None,
                "read_by": r.get("by") or ("OCR (tesseract)" if c.get("read_original") or c.get("read_submission") else None),
                "field": r.get("field"),
            }
        )
    page = (rep.get("pages") or [{}])[0]
    return {
        "label": ex["label"],
        "caption": ex.get("caption", ""),
        "verdict": rep.get("verdict"),
        "matched": page.get("original_page") is not None,
        "match_inliers": page.get("inliers"),
        "glyph_similarity_median": page.get("quality"),
        "marks_compared": prov.get("marks_compared"),
        "marks_reproduced_pct": round(100 * prov["marks_reproduced"], 1) if "marks_reproduced" in prov else None,
        "background_measurable": bg.get("measurable"),
        "background_correlation": bg.get("zncc"),
        "provenance_finding": prov.get("finding"),
        "differences_in_results": changes,
        "sha256": rep.get("exhibit_integrity", {}).get("sha256"),
    }


# ---------------------------------------------------------------------------------------------
# wording

def _sig_sentence(case: dict) -> str:
    sigs = case["reference_integrity"].get("signatures") or []
    f = case["reference_integrity"]
    base = f"The reference original is the file {f['file']} ({f['bytes']:,} bytes, SHA-256 {f['sha256']})."
    if not sigs:
        return base + " It carries no embedded digital signature."
    s = sigs[0]
    if not s.get("checked"):
        return base + f" It carries an embedded digital signature, which I was unable to validate ({s.get('error') or s.get('reason')})."
    state = "intact and cryptographically valid" if s.get("intact") and s.get("valid") else "NOT intact"
    cover = "covering the entire file" if s.get("coverage") == "ENTIRE_FILE" else f"with coverage {s.get('coverage')}"
    return (
        base + f" It carries an embedded digital signature by \"{s.get('signer')}\" (issuer \"{s.get('issuer')}\"), "
        f"signing time {s.get('signing_time')}, digest algorithm {s.get('digest_algorithm')}. On validation the signature is "
        f"{state}, {cover}, which means the file has not been modified since it was signed. "
        + ("The signer's certificate chain could not be traced to a trusted root with the materials available to me." if not s.get("trusted") else "")
    ).strip()


def _exhibit_sentence(case: dict, f: dict) -> str:
    cap = f" (captioned \"{f['caption']}\")" if f["caption"] else ""
    if not f["matched"]:
        return f"{f['label']}{cap} could not be matched to the reference original with sufficient confidence; I make no finding on it."
    head = (
        f"{f['label']}{cap} was registered onto the reference original using {f['match_inliers']:,} matching image features. "
        f"Of the {f['marks_compared']:,} handwritten and printed marks compared, {f['marks_reproduced_pct']}% are reproduced at the same "
        f"positions with the same shapes"
    )
    if f["background_measurable"]:
        head += f", and the background outside the form sheet in the reference capture correlates with this exhibit at {f['background_correlation']:.2f}"
    head += "."
    n = len(f["differences_in_results"])
    if f["verdict"] in ("clean", "identical", "review") and n == 0:
        return head + " There is no difference between this exhibit and the reference original anywhere in the results section of the form."
    diffs = []
    for d in f["differences_in_results"]:
        if d["original_reads"] or d["questioned_reads"]:
            what = f"\"{d['original_reads'] or '?'}\" in the original against \"{d['questioned_reads'] or '?'}\" in the exhibit"
        else:
            what = "a figure that differs (see the evidence image)"
        where = f" ({d['field']})" if d.get("field") else ""
        diffs.append(f"{what}{where} [{d['evidence']}]")
    return head + f" The exhibit differs from the reference original at {n} place(s) in the results section: " + "; ".join(diffs) + "."


def _opinion(case: dict, f: dict) -> str | None:
    if not f["matched"]:
        return None
    n = len(f["differences_in_results"])
    if f["provenance_finding"] == "same_capture" and n:
        return (
            f"In my opinion {f['label']} was produced from the same photographic capture as the reference original uploaded to the "
            f"public portal, and not from a separate carbon copy or a separate photograph of the form: it reproduces the reference's "
            f"handwriting, signatures and stamp impressions in place and also carries the reference capture's own background. The "
            f"{n} difference(s) listed above are confined to the figures, and therefore represent alterations made to an image of the "
            f"reference original."
        )
    if f["provenance_finding"] == "same_capture":
        return f"In my opinion {f['label']} is a reproduction of the reference original and agrees with it."
    if f["provenance_finding"] == "same_sheet" and n:
        return (
            f"In my opinion {f['label']} depicts the same physical form as the reference original (its handwriting, signatures and "
            f"stamp impressions are reproduced in place), but with the {n} figure(s) listed above altered."
        )
    if f["provenance_finding"] == "same_sheet":
        return f"In my opinion {f['label']} depicts the same physical form as the reference original and agrees with it."
    return None


# ---------------------------------------------------------------------------------------------
# document

METHOD = [
    "Retrieval. Each questioned page was matched against every candidate original for the constituencies concerned "
    "using scale- and rotation-invariant image features (SIFT) and a robust geometric fit (RANSAC). All Forms 34A share a "
    "printed template, so every candidate matches to some extent; the true original is singled out by the hand-written "
    "entries, signatures and stamps, which add many further consistent matches. A match was accepted only where it was "
    "both strong and at least twice as strong as the next candidate.",
    "Registration. The questioned page was then mapped onto the reference original by a projective transformation, which "
    "removes differences of scale, rotation, cropping and camera angle, followed by a heavily smoothed correction for paper "
    "curl. The smoothing allows gentle, page-wide distortion to be corrected but cannot move an individual figure, so it "
    "cannot hide an alteration.",
    "Glyph comparison. Both images were normalised for paper tone and contrast and brought to the same sharpness. Every "
    "mark (digit, letter, stroke) on either page was then compared with the same location on the other using zero-mean "
    "normalised cross-correlation, which is insensitive to brightness and contrast. An unchanged mark scores close to 1; a "
    "different figure, an added mark or a removed mark scores below 0.5, which is the threshold used. Differing marks were "
    "grouped into figures and reported with side-by-side evidence images.",
    "Capture provenance. A form exists as one physical sheet, and every photograph or scan of it is a separate capture with "
    "its own framing, lighting and background. Two captures of the same sheet share its handwriting and stamps but not "
    "those capture-specific details; an image made from a capture (printed, re-photographed or edited) inherits them. I "
    "therefore measured (i) the share of the reference's marks reproduced in place in the questioned page and (ii) the "
    "correlation between the background visible outside the form sheet in the reference capture and the same area of the "
    "questioned page.",
    "Integrity of the reference. The SHA-256 hash of each reference file was recorded, and where the file carries an "
    "embedded digital signature it was validated to establish whether the file has changed since it was signed.",
    "Readings. The figures shown in each evidence image were read from that image; the reader of each figure is recorded. "
    "The detection of differences does not depend on these readings.",
]

LIMITS = [
    "The comparison establishes where a questioned page differs from the reference original and whether it was made from the "
    "same capture. Which version is authentic follows from the custody of the reference (a digitally signed upload to the "
    "public portal, and the physical originals), not from the image comparison alone.",
    "The SHA-1 digest used by the KIEMS signatures is deprecated; collision attacks on SHA-1 are known. The signature check "
    "is therefore supporting, not conclusive, evidence of the reference files' integrity.",
    "Photographs of printouts lose detail. Where a questioned page is too degraded to compare reliably, or cannot be matched "
    "to its original, I have said so and made no finding on it.",
    "The thresholds used (match acceptance, similarity 0.5) were calibrated on genuine and altered copies of IEBC forms; "
    "the software, its tests and the full numerical results are produced as part of my annexures.",
]


def build(results: dict, deponent: dict, readings: dict | None = None, use_llm: bool = True) -> Doc:
    matter = results.get("matter", {})
    readings = readings or {}
    d = Doc()
    for line in matter.get("heading", ["REPUBLIC OF KENYA", "IN THE SUPREME COURT OF KENYA AT NAIROBI"]):
        d.center(line, bold=True)
    if matter.get("cause"):
        d.center(matter["cause"], bold=True)
    for party in matter.get("parties", []):
        d.text(party)
    title = matter.get("title", "AFFIDAVIT ON THE FORENSIC EXAMINATION OF QUESTIONED FORMS")
    d.center(title.replace("[DEPONENT]", deponent.get("name", "[DEPONENT NAME]")), bold=True)
    d.text(
        f"I, {deponent.get('name', '[DEPONENT NAME]')}, of {deponent.get('address', '[ADDRESS]')} in the Republic of Kenya, "
        "do hereby make oath and state as follows:"
    )

    d.para(f"THAT I am {deponent.get('profession', '[PROFESSION AND QUALIFICATIONS: forensic document / digital image examiner; years of practice; relevant training and prior testimony]')}, and I am competent to swear this affidavit.")
    d.para(f"THAT I was instructed by {deponent.get('instructed_by', '[INSTRUCTING PARTY]')} to examine the Forms 34A annexed to {matter.get('questioned_affidavit', '[QUESTIONED AFFIDAVIT]')} and to determine whether they agree with the Forms 34A uploaded to the 1st Respondent's public portal. I understand my duty is to the Court and my opinions are my own.")

    d.heading("MATERIALS EXAMINED")
    d.para(f"THAT I examined the following materials, whose SHA-256 hashes I recorded so that they can be identified exactly: {matter.get('materials_summary', '[LIST OF MATERIALS AND THEIR SOURCE]')}.")
    d.para(f"THAT the reference originals are copies of Forms 34A as published on the 1st Respondent's public portal {matter.get('reference_source', '[SOURCE AND DATE OF ACQUISITION, e.g. obtained from https://forms.iebc.or.ke on ...]')}.")

    d.heading("METHOD")
    for m in METHOD:
        d.para("THAT " + m[0].lower() + m[1:])

    cases = [c for c in results["cases"] if "error" not in c]
    d.heading("SUMMARY OF FINDINGS")
    rows = []
    for c in cases:
        for rep in c["exhibit_results"]:
            f = exhibit_facts(c, rep, readings.get(c["id"], {}))
            n = len(f["differences_in_results"])
            rows.append([c["id"], c["title"], f"{f['label']}" + (f" ({f['caption']})" if f["caption"] else ""),
                         "no match" if not f["matched"] else ("agrees with original" if n == 0 else f"{n} figure(s) differ"),
                         (f["provenance_finding"] or "").replace("_", " ")])
    d.para(f"THAT I examined {len(cases)} polling station(s). My findings are summarised in the table below and set out station by station in the paragraphs that follow.")
    d.table(["Case", "Polling station", "Exhibit", "Result", "Provenance"], rows)

    d.heading("FINDINGS BY POLLING STATION")
    for c in cases:
        d.heading(f"{c['id']}: {c['title']}", level=2)
        claim = c.get("claim") or {}
        if claim.get("text"):
            d.para(f"THAT {claim.get('by', 'the deponent')} states at {claim.get('where', '[REFERENCE]')} that {claim['text']}")
        d.para("THAT " + _sig_sentence(c)[0].lower() + _sig_sentence(c)[1:])
        for rep in c["exhibit_results"]:
            f = exhibit_facts(c, rep, readings.get(c["id"], {}))
            fallback = "THAT " + _exhibit_sentence(c, f)
            text, src = (llm.draft("Restate these comparison results as one affidavit paragraph beginning with 'THAT'.", f, fallback) if use_llm else (fallback, "template"))
            d.para(text, src)
            op = _opinion(c, f)
            if op:
                d.para("THAT " + op)

    d.heading("CONCLUSIONS")
    altered = sorted({c["id"] for c in cases for rep in c["exhibit_results"] if exhibit_facts(c, rep, {})["differences_in_results"]})
    d.para(f"THAT of the {len(cases)} polling station(s) examined, the annexed forms for {len(altered)} differ from the reference originals in their figures, as set out above ({', '.join(altered) or 'none'}).")
    d.heading("LIMITATIONS")
    for l in LIMITS:
        d.para("THAT " + l[0].lower() + l[1:])
    d.para("THAT what is deponed to herein is true to the best of my knowledge, information and belief, save for matters of opinion, which are my honest opinions.")
    d.text("SWORN at [PLACE] by the said [DEPONENT NAME] this [DAY] day of [MONTH] [YEAR]")
    d.text("BEFORE ME\n\n\nCOMMISSIONER FOR OATHS")
    d.text("DRAWN & FILED BY: [ADVOCATES]")

    d.page_break()
    d.heading("ANNEXURES: EVIDENCE IMAGES")
    d.text("Each evidence image shows the reference original on the left and the questioned exhibit on the right, at the same location after registration.")
    return d


# ---------------------------------------------------------------------------------------------
# rendering

def add_evidence(d: Doc, results: dict, run_dir: Path, prefix: str = "EX"):
    k = 0
    for c in results["cases"]:
        if "error" in c:
            continue
        for rep in c["exhibit_results"]:
            ev = run_dir / rep["evidence_dir"]
            changes = _changes(rep)
            if not changes:
                continue
            k += 1
            d.heading(f"{prefix}-{k}: {c['id']} {c['title']}, {rep['exhibit']['label']}", level=2)
            for p in rep["pages"]:
                if p.get("overview"):
                    d.image(ev / p["overview"], f"{prefix}-{k}: page overview (reference left, exhibit right; red boxes mark differences)")
            for ch in changes:
                if ch.get("evidence"):
                    d.image(ev / ch["evidence"], f"{prefix}-{k} [{ch['evidence']}]", width_in=3.2)


def to_markdown(d: Doc) -> str:
    out = []
    for b in d.blocks:
        kind = b[0]
        if kind == "heading":
            out.append(f"\n{'#' * (b[2] + 1)} {b[1]}\n")
        elif kind == "center":
            out.append(f"**{b[1]}**" if b[2] else b[1])
        elif kind == "text":
            out.append(b[1] + "\n")
        elif kind == "para":
            out.append(f"{b[1]}. {b[2]}" + (f" *(drafted with {b[3]})*" if b[3] not in ("template",) else "") + "\n")
        elif kind == "sub":
            out.append(f"    ({b[1]}) {b[2]}")
        elif kind == "table":
            out.append("| " + " | ".join(b[1]) + " |\n|" + "---|" * len(b[1]))
            out += ["| " + " | ".join(str(x) for x in r) + " |" for r in b[2]]
            out.append("")
        elif kind == "image":
            out.append(f"![{b[2]}]({b[1]})\n\n*{b[2]}*\n")
        elif kind == "break":
            out.append("\n---\n")
    return "\n".join(out)


def to_docx(d: Doc, path: Path):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.paragraph_format.line_spacing = 1.5
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1)
    for b in d.blocks:
        kind = b[0]
        if kind == "heading":
            p = doc.add_paragraph()
            r = p.add_run(b[1])
            r.bold = True
            r.underline = b[2] == 1
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if b[2] == 1 else WD_ALIGN_PARAGRAPH.LEFT
        elif kind == "center":
            p = doc.add_paragraph()
            r = p.add_run(b[1])
            r.bold = b[2]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif kind == "text":
            doc.add_paragraph(b[1])
        elif kind == "para":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.4)
            p.paragraph_format.first_line_indent = Inches(-0.4)
            p.add_run(f"{b[1]}.\t")
            text = b[2]
            if text.startswith("THAT "):
                r = p.add_run("THAT")
                r.bold = True
                r.underline = True
                text = text[4:]
            p.add_run(text)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "table":
            t = doc.add_table(rows=1, cols=len(b[1]))
            t.style = "Table Grid"
            for i, h in enumerate(b[1]):
                t.rows[0].cells[i].text = h
                for r in t.rows[0].cells[i].paragraphs[0].runs:
                    r.bold = True
            for row in b[2]:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = str(v)
            for row in t.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.line_spacing = 1.0
                        for r in p.runs:
                            r.font.size = Pt(9)
        elif kind == "image":
            if b[1].exists():
                doc.add_picture(str(b[1]), width=Inches(b[3]))
                cap = doc.add_paragraph(b[2])
                cap.runs[0].italic = True
                cap.runs[0].font.size = Pt(9)
        elif kind == "break":
            doc.add_page_break()
    doc.save(str(path))


def write(results_path: Path, deponent: dict, out_stem: Path, readings_path: Path | None = None, use_llm: bool = True) -> tuple[Path, Path]:
    results = json.loads(Path(results_path).read_text())
    readings = json.loads(Path(readings_path).read_text()) if readings_path else {}
    d = build(results, deponent, readings, use_llm=use_llm)
    add_evidence(d, results, Path(results_path).parent)
    md, docx = out_stem.with_suffix(".md"), out_stem.with_suffix(".docx")
    md.write_text(to_markdown(d))
    to_docx(d, docx)
    return md, docx

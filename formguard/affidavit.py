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

ALTERED_KINDS = {
    "substitution": "different figure",
    "addition": "mark added",
    "erasure": "mark removed",
    "obscured": "covered by a mark added to the copy",
}


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

def _changes(rep: dict, on_entries: bool = True) -> list[dict]:
    """Differences in the results section: on the original's entries, or (on_entries=False) marks
    added to the copy on blank paper or over its print, such as captions, page numbers, stamps."""
    return [
        c | {"page": p["submission_page"]}
        for p in rep.get("pages", [])
        for c in p.get("changes", [])
        if c.get("in_table") and c.get("on_figure", True) == on_entries
    ]


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
        if r.get("same_as"):  # a second region of a figure already listed (one number split in two)
            continue
        if not c.get("clear", True) and not (r.get("original") and r.get("questioned") and r["original"] != r["questioned"]):
            continue  # a possible change counts only when the figure reads differently
        changes.append(
            {
                "evidence": c.get("evidence"),
                "kind": ALTERED_KINDS.get(c["kind"], c["kind"]),
                "similarity": c["similarity"],
                # Readings recorded for the evidence image (examiner or model); OCR of photographed
                # handwriting is too unreliable to quote and stays in the technical report.
                "original_reads": r.get("original") or None,
                "questioned_reads": r.get("questioned") or None,
                "read_by": r.get("by"),
                "field": r.get("field"),
                "clear": c.get("clear", True),
                "model_agrees": r.get("model_agrees"),
                "model_reading": (f"{r['model']['original']} on the original and {r['model']['questioned']} on the page" if r.get("model") else None),
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
        "marks_added_elsewhere": len(_changes(rep, on_entries=False)),
        "sha256": rep.get("exhibit_integrity", {}).get("sha256"),
    }


# ---------------------------------------------------------------------------------------------
# wording

def _that(sentence: str) -> str:
    """'The file ...' -> 'THAT the file ...'; 'I ...' keeps its capital."""
    return "THAT " + (sentence if sentence.startswith("I ") else sentence[0].lower() + sentence[1:])


def _local(iso: str) -> str:
    """ISO time in UTC -> '9th August 2022 at 22:54:53 East Africa Time'."""
    from datetime import datetime, timedelta

    t = datetime.fromisoformat(iso) + timedelta(hours=3)
    suffix = "th" if 11 <= t.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(t.day % 10, "th")
    return f"{t.day}{suffix} {t:%B %Y} at {t:%H:%M:%S} East Africa Time"


def _sig_sentence(case: dict) -> str:
    """The reference file and its signature; who signs and what validation shows is explained in the method."""
    sigs = case["reference_integrity"].get("signatures") or []
    f = case["reference_integrity"]
    base = f"The original of this form is the file {f['file']}, of {f['bytes']:,} bytes, with the SHA-256 hash value {f['sha256']}."
    if not sigs:
        return base + " It carries no digital signature."
    s = sigs[0]
    if not s.get("checked"):
        return base + f" It carries a digital signature, which I was unable to validate ({s.get('error') or s.get('reason')})."
    when = f" Its digital signature was applied on {_local(s['signing_time'])}." if s.get("signing_time") else ""
    if s.get("intact") and s.get("valid") and s.get("coverage") == "ENTIRE_FILE":
        return base + when + " The signature validates as intact and covers the whole file, so the file has not changed since it was signed."
    return base + when + " The signature does not validate as covering an unchanged file."


WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _count(n: int, noun: str) -> str:
    word = WORDS[n] if n < len(WORDS) else f"{n:,}"
    return f"{word} {noun}" + ("" if n == 1 else "s")


def _num(x: float) -> str:
    """0.48, \u22120.13; never \u22120.00."""
    s = f"{x:.2f}"
    return "0.00" if s == "-0.00" else s.replace("-", "\u2212")


def _claim(claim: dict) -> str | None:
    """What the questioned affidavit says about a station, or how it presents the station's pages."""
    if claim.get("sentence"):
        return "THAT " + claim["sentence"]
    if claim.get("text"):
        return f"THAT the affidavit of {claim.get('by', '[DEPONENT]')} states at {claim.get('where', '[REFERENCE]')} that {claim['text']}"
    return None


def _field_order(d: dict) -> tuple:
    """Candidates' votes before totals, so each station's figures are listed the same way."""
    field = d.get("field") or ""
    return (not field.startswith("the votes of"), field)


def _exhibit_sentence(case: dict, f: dict, mark: str | None) -> str:
    cap = f", captioned \"{f['caption']}\"," if f["caption"] else ""
    text = f"THAT the page at {f['label']}{cap} shows this form."
    if not f["matched"]:
        return text[:-1] + " but could not be matched to the original with enough confidence. I make no finding on it."
    text += (
        f" It matches the original at {f['match_inliers']:,} image features and reproduces "
        f"{f['marks_reproduced_pct']:.0f}% of its {f['marks_compared']:,} marks in place."
    )
    if f["background_measurable"]:
        text += f" The background beyond the edge of the sheet correlates with that of the original at {_num(f['background_correlation'])}."
    q = f.get("glyph_similarity_median")
    if q is not None and q < 0.85:
        text += (
            f" The page is degraded. The median similarity of its marks is {_num(q)}, below the 0.85 I require for a "
            f"complete comparison, so some differences may not be visible."
        )
    diffs = f["differences_in_results"]
    same = [d for d in diffs if d["original_reads"] and d["original_reads"] == d["questioned_reads"]]
    changed = sorted(_differing(f), key=_field_order)
    if not changed:
        text += " Its entries agree with those of the original."
    else:
        shown = f", shown in annexure {mark}" if mark else ""
        text += f" It differs from the original in {_count(len(changed), 'figure')}{shown}."
        for d in changed:
            sim = _num(d["similarity"])
            if d["original_reads"] and d["questioned_reads"]:
                what = f"The figure for {d['field']}" if d.get("field") else "A figure"
                text += f" {what} reads {d['original_reads']} on the original and {d['questioned_reads']} on the page, with a similarity of {sim}."
            else:
                text += f" A further figure differs, with a similarity of {sim}."
            if not d.get("clear", True):
                text += " That similarity is in the range where I rely on the reading."
            if d.get("model_agrees") is False:
                text += f" The second reader read it as {d['model_reading']}."
    if same:
        n = _count(len(same), "mark")
        n = f"a further {n}" if changed else n
        text += f" {n[0].upper()}{n[1:]} scored below 0.65 but {'reads' if len(same) == 1 else 'read'} the same on both images."
    return text


PROVENANCE_SHORT = {"same_capture": "made from the portal photograph", "same_sheet": "shows the same sheet"}  # summary table


def _differing(f: dict) -> list[dict]:
    """The differences of exhibit_facts that remain once read: a figure that reads the same on both is not one."""
    return [d for d in f["differences_in_results"] if not (d["original_reads"] and d["original_reads"] == d["questioned_reads"])]


def _opinion(case: dict, f: dict) -> str | None:
    changed = _differing(f)
    if not f["matched"] or not changed:
        return None
    if f["provenance_finding"] == "same_capture":
        return (
            "THAT in my opinion this page was made from the same photograph as the portal original and not from a separate "
            "photograph of the form or from a carbon copy. It reproduces the original's handwriting, signatures and stamps in "
            "place, together with the background of that photograph beyond the edge of the sheet. The figures listed in the "
            "preceding paragraph were therefore changed on a copy of the portal image."
        )
    if f["provenance_finding"] == "same_sheet":
        return (
            "THAT in my opinion this page shows the same handwritten sheet as the portal original. Its handwriting, signatures "
            "and stamps coincide with those of the original in shape and position, and only the figures listed in the "
            "preceding paragraph differ."
        )
    return None


# ---------------------------------------------------------------------------------------------
# document

METHOD = [
    "I examined each annexed page in the steps set out in this and the following six paragraphs. Each step can be repeated "
    "with the same files and the software I used, which I produce as annexure {software}. I first identified the original of "
    "each annexed page. I compared the page with every candidate original using image features that are unaffected by "
    "changes of scale, rotation and camera angle. All Forms 34A share one printed layout, so every candidate matches a page "
    "to some degree. The original of a page is distinguished by its handwriting, signatures and stamps, which add several "
    "hundred further matching points. I accepted a match only where it was strong and at least twice as strong as the next "
    "best candidate.",
    "I then aligned each annexed page with its original by a geometric transformation that removes differences of size, "
    "angle, cropping and perspective, followed by a smooth correction for the curl of the paper. The correction is measured "
    "over areas several times the size of a figure and cannot move a single figure, so it cannot conceal an alteration. I "
    "compared only the paper of each image and excluded the desk or other background visible around a photographed sheet.",
    "I then corrected the two images for differences of paper tone, contrast and sharpness, area by area, since a "
    "photograph is not equally sharp across the sheet. I compared every mark on the form, that is each digit, letter and "
    "stroke, with the same place on the other image, using a correlation measure that is unaffected by brightness and "
    "contrast. The measure runs from \u22121 to 1, and an unchanged mark scores close to 1. I measured every figure containing a doubtful mark "
    "again in windows the size of a single digit, so that a changed digit is measured on its own even where it touches the "
    "next one. A figure scoring below 0.5 differs from the original. A figure scoring between 0.5 and 0.65 may differ, as "
    "when a 3 is made from an 8, and I report it as differing only where it reads differently on the two images. For each "
    "figure that differs I prepared an image showing the original and the annexed page side by side.",
    "Photographs of printed copies often lose the grey printed lettering of a form while the handwritten entries remain "
    "dark. So as not to mistake faded print for an alteration, I identified the printed parts of each original by "
    "overlaying other Forms 34A on it, since the print is common to all of them while the handwritten entries are not. I did "
    "not count faded print, marks that are present but fainter, or marks added to a copy for the court bundle, such as "
    "captions, page numbers and stamps, as differences in the entries.",
    "A Form 34A is a single sheet of paper, and every photograph or scan of it is a separate capture with its own framing, "
    "lighting and background. Two separate captures of the same sheet share its handwriting and stamps but not those "
    "features of the capture. An image made from a capture, whether by printing, re-photographing or editing it, carries "
    "them over. For each annexed page I therefore measured the proportion of the original's marks that it reproduces in "
    "place, and the correlation between the background visible beyond the edge of the sheet in the original and the same "
    "area of the annexed page.",
    "I recorded the SHA-256 hash value of each original and validated its digital signature, which shows whether the file "
    "has changed since it was signed. The Forms 34A on the portal are signed when transmitted from the KIEMS kit, with a "
    "certificate issued by \"CA_APPS\" of Smartmatic International and the SHA-1 digest.",
    "The figures I quote in my findings were read from the side-by-side images twice: by me, and independently by a vision "
    "language model run through Ollama Cloud. Where the two readings differ, I say so. The finding that a figure differs "
    "does not depend on how it is read.",
]

LIMITS = [
    "The comparison shows where an annexed page differs from its original and whether it was made from the same capture. "
    "Which version is authentic follows from the custody of the original, being a digitally signed upload to the portal and "
    "the physical form, and not from the comparison alone.",
    "The SHA-1 digest used in the KIEMS signatures is no longer considered secure against deliberate collisions. I therefore "
    "treat the signature check as supporting evidence of the integrity of the originals and not as proof on its own.",
    "Photographs of printed copies lose detail. Where an annexed page was too degraded to compare completely, or could not be "
    "matched to its original, I have said so.",
    "I set the thresholds I applied, for accepting a match and for similarity, by testing the method on genuine and altered "
    "copies of IEBC forms. The software, its tests and the full numerical results form part of annexure {software}.",
]


def _plain_title(title: str) -> str:
    """'X polling station 1 of 1 (Tiaty Constituency, Baringo County)' -> 'X polling station 1 of 1, Tiaty Constituency, Baringo County'."""
    title = re.sub(r"'([A-Z])(?=[a-z])", lambda m: "'" + m.group(1).lower(), title)  # Kong'Or -> Kong'or
    return re.sub(r"\s*\((.*)\)\s*$", r", \1", title)


def annexure_marks(results: dict, prefix: str, readings: dict | None = None) -> dict[str, str]:
    """One annexure per polling station with a figure that differs from the original; the software is annexure 1."""
    readings = readings or {}
    marks, k = {}, 1
    for c in results["cases"]:
        if "error" in c:
            continue
        if any(_differing(exhibit_facts(c, rep, readings.get(c["id"], {}))) for rep in c["exhibit_results"]):
            k += 1
            marks[c["id"]] = f"{prefix}-{k}"
    return marks


def build(results: dict, deponent: dict, readings: dict | None = None, use_llm: bool = True) -> Doc:
    matter = results.get("matter", {})
    readings = readings or {}
    prefix = deponent.get("initials", "[XX]")
    software = f"{prefix}-1"
    marks = annexure_marks(results, prefix, readings)
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

    d.para(f"THAT I am {deponent.get('profession', '[PROFESSION AND QUALIFICATIONS]')}, and I am competent to swear this affidavit.")
    d.para(
        f"THAT I was instructed by {deponent.get('instructed_by', '[INSTRUCTING PARTY]')} to examine the Forms 34A annexed to "
        f"{matter.get('questioned_affidavit', '[QUESTIONED AFFIDAVIT]')}, and to state whether they agree with the Forms 34A that "
        "the 1st Respondent published on its public portal."
    )
    d.para("THAT my duty is to this Honourable Court. Where a statement below is my opinion rather than a measurement, I say so.")

    d.heading("MATERIALS EXAMINED")
    d.para(f"THAT I examined {matter.get('materials_summary', '[THE QUESTIONED AFFIDAVIT, WITH ITS SHA-256 HASH VALUE]')}.")
    d.para(f"THAT as reference originals I used the Forms 34A published on the 1st Respondent's public portal, {matter.get('reference_source', '[SOURCE AND DATE OF ACQUISITION]')}.")

    d.heading("METHOD")
    for m in METHOD:
        d.para(_that(m.format(software=software)))

    cases = [c for c in results["cases"] if "error" not in c]
    d.heading("SUMMARY OF FINDINGS")
    rows = []
    for c in cases:
        for rep in c["exhibit_results"]:
            f = exhibit_facts(c, rep, readings.get(c["id"], {}))
            changed = _differing(f)
            result = "not matched" if not f["matched"] else ("agrees with the original" if not changed else _count(len(changed), "figure") + " differ" + ("s" if len(changed) == 1 else ""))
            made = PROVENANCE_SHORT.get(f["provenance_finding"] or "", "not determined") if changed else ""
            rows.append([_plain_title(c["title"]).split(",")[0], f["label"], f["caption"] or "none", result, made])
    d.para(f"THAT I examined the annexed pages for {len(cases)} polling stations. I summarise my findings in the following table and set them out station by station in the paragraphs after it.")
    d.table(["Polling station", "Annexed page", "Caption", "Result", "How the page was made"], rows)

    d.heading("FINDINGS FOR EACH POLLING STATION")
    for c in cases:
        d.heading(_plain_title(c["title"]), level=2)
        claim = _claim(c.get("claim") or {})
        if claim:
            d.para(claim)
        d.para(_that(_sig_sentence(c)))
        for rep in c["exhibit_results"]:
            f = exhibit_facts(c, rep, readings.get(c["id"], {}))
            fallback = _exhibit_sentence(c, f, marks.get(c["id"]))
            if use_llm:
                facts = {**f, "annexure": marks.get(c["id"])}
                text, src = llm.draft(
                    "Restate these comparison results as one affidavit paragraph beginning with 'THAT'. Keep every figure, "
                    "reading, measurement and annexure mark exactly as given.",
                    facts,
                    fallback,
                )
            else:
                text, src = fallback, "template"
            d.para(text, src)
            op = _opinion(c, f)
            if op:
                d.para(op)

    d.heading("CONCLUSIONS AND LIMITATIONS")
    altered = [c for c in cases if c["id"] in marks]
    d.para(
        f"THAT of the {len(cases)} polling stations examined, the annexed pages for {len(altered)} differ from the portal "
        "originals in their figures, as set out above."
    )
    for l in LIMITS:
        d.para(_that(l.format(software=software)))
    d.para("THAT what is deponed to in this affidavit is true to the best of my knowledge, save for matters of opinion, which are my honest opinions.")
    d.text("SWORN at [PLACE] by the said [DEPONENT NAME] this [DAY] day of [MONTH] [YEAR]")
    d.text("BEFORE ME\n\n\nCOMMISSIONER FOR OATHS")
    d.text("DRAWN AND FILED BY: [ADVOCATES]")

    d.page_break()
    d.heading("ANNEXURES")
    d.text(
        "Each image shows the portal original on the left and the annexed page on the right, at the same place on the form "
        "after alignment."
    )
    return d


# ---------------------------------------------------------------------------------------------
# rendering

def add_evidence(d: Doc, results: dict, run_dir: Path, prefix: str = "[XX]", readings: dict | None = None):
    readings = readings or {}
    marks = annexure_marks(results, prefix, readings)
    for c in results["cases"]:
        if c.get("id") not in marks:
            continue
        d.heading(f"Annexure {marks[c['id']]}: {_plain_title(c['title'])}", level=2)
        for rep in c["exhibit_results"]:
            ev = run_dir / rep["evidence_dir"]
            for ch in sorted(_differing(exhibit_facts(c, rep, readings.get(c["id"], {}))), key=_field_order):
                if ch.get("evidence"):
                    d.image(ev / ch["evidence"], f"{rep['exhibit']['label']}: original (left) and annexed page (right)", width_in=4.5)


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
    add_evidence(d, results, Path(results_path).parent, deponent.get("initials", "[XX]"), readings)
    md, docx = out_stem.with_suffix(".md"), out_stem.with_suffix(".docx")
    md.write_text(to_markdown(d))
    to_docx(d, docx)
    return md, docx

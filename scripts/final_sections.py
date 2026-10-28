"""Markdown for the sections after the station findings: inconsistencies in the questioned
affidavit, conclusions and limitations, the jurat, and the annexure list. Counts and cross-
references are computed from the run, the readings and doc_md.py's numbering, not typed in.

usage: final_sections.py results.json readings.json manifest.json numbering.json annex_index.json commit out_dir
"""
import json
import sys
from collections import Counter
from pathlib import Path

from formguard.affidavit import LIMITS, _differing, _local, _that, exhibit_facts

results, readings, manifest, numbering, annex, commit, out = (
    json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), json.load(open(sys.argv[3])),
    json.load(open(sys.argv[4])), json.load(open(sys.argv[5])), sys.argv[6], Path(sys.argv[7]),
)
out.mkdir(parents=True, exist_ok=True)
cases = {c["id"]: c for c in results["cases"] if "error" not in c}
claims = {c["id"]: c.get("claim") or {} for c in manifest["cases"]}
pages = numbering["pages"]
n = numbering["next"]
software = "[XX]-1"
kagera_mark = f"[XX]-{len(annex) + 2}"


def page_para(cid, pdf_page):
    return next(v for k, v in pages[cid].items() if k.startswith(f"PDF page {pdf_page} "))


def sig(cid):
    c = cases[cid]
    s = c["reference_integrity"]["signatures"][0]
    assert s["intact"] and s["valid"] and s["coverage"] == "ENTIRE_FILE"
    return c["reference_integrity"]["file"], _local(s["signing_time"])


def reading(cid, field):
    vals = {(r["original"], r["questioned"]) for r in readings.get(cid, {}).values() if r.get("field") == field and not r.get("same_as")}
    assert len(vals) == 1, (cid, field, vals)
    return vals.pop()


# --- facts the text rests on, checked rather than assumed --------------------------------------
facts = {cid: [(rep, exhibit_facts(c, rep, readings.get(cid, {}))) for rep in c["exhibit_results"]] for cid, c in cases.items()}
by_caption = Counter()
for cid, fs in facts.items():
    for rep, f in fs:
        by_caption[(f["caption"] or "none", bool(_differing(f)))] += 1
agent_total = by_caption[("AGENT COPY", True)] + by_caption[("AGENT COPY", False)]
agent_differ = by_caption[("AGENT COPY", True)]
others_differ = sum(v for (cap, d), v in by_caption.items() if d and cap not in ("AGENT COPY", "IEBC (INITIAL) PORTAL"))
assert others_differ == 0, by_caption
agent_same = [cid for cid, fs in facts.items() for rep, f in fs if f["caption"] == "AGENT COPY" and not _differing(f)]
assert agent_same == ["C47"], agent_same

ruto, totals, raila = [], [], []
for cid, fs in facts.items():
    for rep, f in fs:
        for d in _differing(f):
            field = d.get("field") or ""
            pair = (int(d["original_reads"]), int(d["questioned_reads"]))
            (ruto if field == "the votes of William Ruto" else raila if "Raila" in field else totals).append(pair)
assert ruto and all(o > q for o, q in ruto) and all(o > q for o, q in totals) and not raila

table_agree = 0
for cid, cl in claims.items():
    fig = cl.get("figures")
    if fig and fig["agent"].get("Ruto") is not None:
        assert reading(cid, "the votes of William Ruto") and {(int(a), int(b)) for a, b in [reading(cid, "the votes of William Ruto")]} == {(fig["portal"]["Ruto"], fig["agent"]["Ruto"])}, cid
        table_agree += 1
for cid, inc in (("C27", 100), ("C28", 83), ("C41", 60)):
    o, q = reading(cid, "the votes of William Ruto")
    assert int(o) - int(q) == inc, (cid, o, q)

prov = Counter()
pages_total = 0
for cid, fs in facts.items():
    for rep, f in fs:
        if _differing(f):
            pages_total += any((d.get("field") or "").startswith("the total") for d in _differing(f))
            prov[f["provenance_finding"] if f["provenance_finding"] in ("same_capture", "same_sheet") else "other"] += 1
differing_pages = sum(prov.values())

assert reading("C48", "the votes of William Ruto") == ("195", "115")
assert reading("C48", "the total number of valid votes cast") == ("256", "176")
kad_file, kad_time = sig("C42")
mar_file, mar_time = sig("C47")
kag_file, kag_time = sig("C48")
dates = Counter(_local(c["reference_integrity"]["signatures"][0]["signing_time"]).split(" at ")[0] for c in cases.values())
assert set(dates) == {"9th August 2022", "10th August 2022"}, dates


# --- inconsistencies ---------------------------------------------------------------------------
def para(text):
    global n
    line = f"{n}. **THAT**{text[4:]}" if text.startswith("THAT") else f"{n}. {text}"
    n += 1
    return line


s6 = ["## INCONSISTENCIES IN THE AFFIDAVIT OF ARNOLD OCHIENG OGINGA", ""]
s6.append(para(
    "THAT the affidavit of Arnold Ochieng Oginga gives two different increases in the votes of William Ruto for five polling "
    "stations in Baringo County. One is in the list at paragraph 17 and the other is in the table at pages 8 to 13:"))
s6 += ["", "| Polling station | Increase in paragraph 17 | Increase in the table | William Ruto on the agent's form, in the table | William Ruto on the portal form, in the table |",
       "|---|---|---|---|---|",
       "| Kapedo Primary School 1 of 1 | 1 | 3 | 1 | 4 |",
       "| Korelach Primary School 1 of 1 | 1 | 81 | 99 | 180 |",
       "| Kamsino Primary School 1 of 1 | 81 | 2 | 1 | 3 |",
       "| Chepkirial Primary School 1 of 1 | 2 | 3 | 0 | 3 |",
       "| Koipasiran Village 1 of 1 | 32 | 34 | 34 | 66 |", ""]
s6.append(para(
    "THAT the table also disagrees with itself. For Koipasiran Village its own figures of 34 and 66 give an increase of 32, not "
    "the 34 it states. For Chepkirial Primary School it records 62 votes for Raila Odinga on the agent's form and 61 on the portal "
    "form, but shows no variance for Raila Odinga. The agent's copy annexed at PDF page 94 reads 061 for Raila Odinga, as the "
    "original does."))
s6.append(para(
    "THAT the Baringo totals in the table do not follow from its rows. Its rows give 3,420 votes for Raila Odinga on the portal "
    "forms, but the total it states is 3,421. Its totals for William Ruto, 268 on the agents' forms and 625 on the portal forms, "
    "differ by 357. The total variance it states is 359, which is the sum of its variance column. The difference of 2 comes from "
    "the row for Koipasiran Village."))
s6.append(para(
    "THAT for Nairobi County paragraph 17 lists fifteen polling stations, while the table lists twelve. The table leaves out "
    "Kabete Vetlab Primary School polling stations 2, 3 and 4 of 12, for which paragraph 17 gives increases of 100, 83 and 106 "
    "votes. The Nairobi totals in the table are 3,470 votes for Raila Odinga, 1,843 votes for William Ruto on the portal forms and "
    "a variance of 1,217. Its twelve rows add up to 2,778, 1,487 and 928. The differences are 692, 356 and 289."))
s6.append(para(
    "THAT the portal originals for the three stations left out record 233, 235 and 224 votes for Raila Odinga, which add up to "
    "692. They record 123, 118 and 115 votes for William Ruto, which add up to 356. The increases that paragraph 17 gives for "
    "them add up to 289. The Nairobi totals in the table therefore include the three stations that its rows leave out. For Kabete "
    "Vetlab Primary School 12 of 12 and St Martin's School Kibarage 4 of 7 the table gives a variance but no figure for William "
    "Ruto on the agent's form."))
s6.append(para(
    f"THAT paragraph 21 of the affidavit states that the Form 34A for \"Kadondo Grounds - Mishomoroni Polling Station 2 of 9\" has "
    f"not been uploaded in the portal. The Mombasa archive listed in paragraph 5 contains a Form 34A for Kadongo Grounds - "
    f"Mishomoroni polling station 2 of 9. It is the file {kad_file}, signed on {kad_time}, and its signature validates. The page "
    f"at PDF page 46 of the affidavit, which has no caption, shows the same form. Its entries agree with those of that file, as I "
    f"state at paragraph {page_para('C42', 46)}."))
s6.append(para(
    f"THAT paragraph 20 of the affidavit states that neither the agent's Form 34A for Maragi Primary School polling station 1 of 4 "
    f"nor the portal form states any results. The portal original, the file {mar_file}, was signed on {mar_time}. It has no "
    f"figures in its column of votes for each candidate. Its polling station counts are filled in, and give 548 registered voters, "
    f"3 rejected ballot papers and 344 valid votes cast. The agent's copy annexed at PDF page 129 agrees with it, as I state at "
    f"paragraph {page_para('C47', 129)}. The forms therefore bear out paragraph 20 as to the votes for each candidate."))
s6.append(para(
    f"THAT two pages are annexed for Kagera Primary School polling station 1 of 2 in Gatundu South Constituency, one captioned "
    f"\"IEBC PORTAL\" and the other \"IEBC (INITIAL) PORTAL\". Paragraph 7 of the affidavit states that Forms 34A on the portal were "
    f"replaced with forms \"containing different results from what they had initially posted\". The portal original, the file "
    f"{kag_file}, was signed on {kag_time}. The page captioned \"IEBC PORTAL\" agrees with it, as I state at paragraph "
    f"{page_para('C48', 130)}. The page captioned \"IEBC (INITIAL) PORTAL\" differs from it in three figures, as I state at "
    f"paragraph {page_para('C48', 131)}. It gives William Ruto 115 votes against 195 on the original, and both totals of valid "
    f"votes as 176 against 256."))
s6.append(para(
    f"THAT in my opinion both pages annexed for Kagera Primary School were printed from the portal photograph. In the portal "
    f"original the top edge of the sheet is curled over, against a dark background. Both annexed pages show the top edge curled in "
    f"the same way against the same background, as annexure {kagera_mark} shows. My measurements for the page captioned \"IEBC "
    f"(INITIAL) PORTAL\" fall short of the levels I set for finding that it was made from the portal photograph, so this opinion "
    f"rests on my examination of the images. If it is right, the figures on that page were changed on a copy of the signed portal "
    f"image, and the page does not show an earlier version of the form."))
s6.append(para(
    "THAT paragraph 7 also states that Forms 34A were deleted and uploaded as late as 19th August 2022. Each original I relied on "
    "carries a digital signature applied on 9th or 10th August 2022 which validates, so each of those files is as it was when it "
    "was signed. Examining images cannot show what the portal displayed at other times. I have not examined annexure \"AOO1\", "
    "which is a print-out of portal timestamps and not a Form 34A."))
s6.append(para(
    "THAT statements resting on server logs, on access to the 1st Respondent's systems or on the accounts of witnesses, such as "
    "those in the affidavit of John Mark Githongo, cannot be tested by examining images of forms. I express no opinion on them."))

# --- conclusions -------------------------------------------------------------------------------
s7 = ["## CONCLUSIONS AND LIMITATIONS", ""]
s7.append(para(
    f"THAT I examined the annexed pages for {len(cases)} polling stations. For {len(annex)} of them an annexed page differs from "
    f"the portal original in its figures."))
s7.append(para(
    f"THAT the pages that differ are {agent_differ} of the {agent_total} pages captioned \"AGENT COPY\" and the page captioned "
    f"\"IEBC (INITIAL) PORTAL\" for Kagera Primary School. The one page captioned \"AGENT COPY\" that agrees with its original is "
    f"the page for Maragi Primary School. Every other annexed page, including every page captioned \"IEBC PORTAL\", agrees in its "
    f"entries with the portal original."))
s7.append(para(
    "THAT on every page that differs, the votes of William Ruto are lower than on the original"
    + (", and so are the totals of valid votes." if pages_total == differing_pages else
       f". On {pages_total} of those pages at least one total of valid votes is lower as well.")
    + " No figure for Raila Odinga differs on any page."))
s7.append(para(
    f"THAT for the {table_agree} polling stations for which the table in the affidavit of Arnold Ochieng Oginga gives the votes "
    f"of William Ruto on both forms, its figure for the portal form is the figure on the portal original. Its figure for the agent's "
    f"form is the figure on the annexed page captioned \"AGENT COPY\". For Kabete Vetlab Primary School 2 and 3 of 12 and St Martin's "
    f"School Kibarage 4 of 7, the increase the affidavit gives is the difference between the original and the annexed copy. The "
    f"variances in that affidavit are therefore differences between each portal original and an annexed copy that differs from it, "
    f"and not differences between two versions of a form on the portal."))
s7.append(para(
    f"THAT of the {differing_pages} pages that differ, {prov['same_capture']} reproduce the original's handwriting together with the "
    f"background of the portal photograph, and in my opinion were made from that photograph. A further {prov['same_sheet']} show "
    f"the same handwritten sheet as the original. For the other {prov['other']} my measurements do not settle how the page was "
    f"made. A photograph of a printed copy often loses the background of the image it was printed from, so a low background "
    f"correlation does not show that a page was made from a different photograph."))
for limit in LIMITS:
    s7.append(para(_that(limit.format(software=software))))
s7.append(para("THAT what is deponed to in this affidavit is true to the best of my knowledge, save for matters of opinion, which are my honest opinions."))

s8 = ["**SWORN** at [PLACE] by the said [DEPONENT NAME] this [DAY] day of [MONTH] [YEAR]", "", "[SIGNATURE]", "", "**DEPONENT**", "",
      "**BEFORE ME**", "", "[SIGNATURE AND STAMP]", "", "**COMMISSIONER FOR OATHS**", "", "DRAWN AND FILED BY: [ADVOCATES]"]

s9 = ["## ANNEXURES", "",
      f"Annexure {software} is the formguard software at commit {commit}, with its tests and the full numerical results of this "
      f"examination, produced in electronic form.", "",
      "Each annexure that follows shows, for each figure that differs, the portal original on the left and the annexed page on the "
      "right, at the same place on the form after alignment. Below each pair are the figure, how it reads on each image and its "
      "measured similarity."]

(out / "s6.md").write_text("\n".join(s6) + "\n")
(out / "s7.md").write_text("\n".join(s7) + "\n")
(out / "s8.md").write_text("\n".join(s8) + "\n")
(out / "s9_intro.md").write_text("\n".join(s9) + "\n")
json.dump({"kagera_mark": kagera_mark, "commit": commit, "next": n, "by_caption": {f"{k[0]}|{k[1]}": v for k, v in by_caption.items()}, "prov": prov,
           "table_agree": table_agree}, open(out / "final_facts.json", "w"), indent=1)
print("paragraphs to", n - 1, "| kagera", kagera_mark, "| agent", agent_differ, "of", agent_total, "| table agree", table_agree, "| prov", dict(prov))

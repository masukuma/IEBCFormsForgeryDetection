# Draft affidavit: sources

The draft affidavit is in `build/` as Word, PDF and Markdown, and as a Claude Doc with the same text.
This folder holds what it was generated from, so every number in it can be traced and regenerated.

| Path | What it is |
|---|---|
| `../case/results.json`, `../case/C*/` | results of the final run, `formguard cases case/manifest.json` |
| `../case/manifest.json` | the 48 cases: original, annexed pages, what the Ochieng affidavit says about each |
| `../case/ochieng_table.json` | the table at pages 8 to 13 of the Ochieng affidavit, as typed from the page images |
| `../case/readings_examiner.json` | the figure in each evidence image, read by eye; `same_as` marks a second region of one number |
| `../case/readings_model.json` | the same figures read by a vision model on Ollama Cloud (second reader) |
| `../case/readings_merged.json` | both readings side by side, with whether they agree and the examiner's remarks |
| `build/` | the affidavit: `.docx`, `.pdf` and the master `.md` they are made from |
| `sections/01_front.md` | heading, parties and paragraphs 1 to 15 |
| `annexures/` | one image per station whose annexed page differs; `kagera_top.png` is the Kagera top-of-sheet annexure |
| `numbering.json`, `final_facts.json` | paragraph numbers per station and page, and the counts the conclusions state |

Regenerate (needs the case run directory, the 150 dpi Kagera renders and LibreOffice):

```
python scripts/merge_readings.py case/readings_examiner.json case/readings_model.json case/readings_merged.json
scripts/build.sh <run dir> case/readings_merged.json case/manifest.json <kagera renders> <commit> out
```

`final_sections.py` checks the facts its text rests on (every differing figure lowers William
Ruto's votes, the table's figures match the readings, the signature dates) and stops if one fails.
Paragraphs 1 to 15 (deponent, materials, method) were written in the doc directly; their method
text matches `formguard/affidavit.py` METHOD.

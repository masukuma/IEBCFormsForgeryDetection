# Draft affidavit: sources

The draft affidavit itself is a Claude Doc (exportable to Word). This folder holds what its
sections 4 to 9 were generated from, so every number in them can be traced and regenerated.

| Path | What it is |
|---|---|
| `../case/results.json`, `../case/C*/` | results of the final run, `formguard cases case/manifest.json` |
| `../case/manifest.json` | the 48 cases: original, annexed pages, what the Ochieng affidavit says about each |
| `../case/ochieng_table.json` | the table at pages 8 to 13 of the Ochieng affidavit, as typed from the page images |
| `../case/readings_examiner.json` | the figure in each evidence image, read by eye; `same_as` marks a second region of one number |
| `../case/readings_model.json` | the same figures read by a vision model on Ollama Cloud (second reader) |
| `sections/` | the markdown of sections 4 to 9, as placed in the doc |
| `annexures/` | one image per station whose annexed page differs; `kagera_top.png` is the Kagera top-of-sheet annexure |
| `numbering.json`, `final_facts.json` | paragraph numbers per station and page, and the counts the conclusions state |

Regenerate, from the repository root with the case data in `case/`:

```
python scripts/doc_md.py case/results.json case/readings_examiner.json 17 out case/manifest.json
python scripts/annex_images.py case case/readings_examiner.json out/annexures
python scripts/final_sections.py case/results.json case/readings_examiner.json case/manifest.json \
    out/numbering.json out/annexures/index.json <commit> out
python scripts/merge_readings.py case/readings_examiner.json case/readings_model.json out/readings_merged.json
```

`final_sections.py` checks the facts its text rests on (every differing figure lowers William
Ruto's votes, the table's figures match the readings, the signature dates) and stops if one fails.
Paragraphs 1 to 15 (deponent, materials, method) were written in the doc directly; their method
text matches `formguard/affidavit.py` METHOD.

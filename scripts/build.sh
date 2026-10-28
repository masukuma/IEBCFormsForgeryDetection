#!/bin/sh
# Build the affidavit (Markdown, Word, PDF) from a case run and the merged readings.
# usage: scripts/build.sh run_dir readings_merged.json manifest.json kagera_pages_dir commit out_dir
# kagera_pages_dir holds the 150 dpi renders kagH_orig-1.png, kagH130-130.png, kagH131-131.png.
set -e
run=$1; readings=$2; manifest=$3; pages=$4; commit=$5; out=$6
here=$(dirname "$0")
mkdir -p "$out"
python "$here/doc_md.py" "$run/results.json" "$readings" 17 "$out" "$manifest"
python "$here/annex_images.py" "$run" "$readings" "$out/annexures"
python "$here/final_sections.py" "$run/results.json" "$readings" "$manifest" "$out/numbering.json" "$out/annexures/index.json" "$commit" "$out"
mark=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['kagera_mark'])" "$out/final_facts.json")
python "$here/kagera_annex.py" "$pages/kagH_orig-1.png" "$pages/kagH130-130.png" "$pages/kagH131-131.png" "$mark" "$out/annexures/kagera_top.png"
python "$here/assemble.py" "$here/../affidavit/sections/01_front.md" "$run/results.json" "$readings" "$out" "$out/Affidavit_Forensic_Examination_Forms_34A_DRAFT"

#!/usr/bin/env bash
# =================================================================
# Build: draft_proposal_skripsi_en.md -> draft_proposal_skripsi_en.docx
# (English edition; padanan build_docx.sh)
#
# Memakai _docx_build.py mode "en":
#   - front matter Inggris (Cover, Approval Sheet, Abstract, Table of
#     Contents otomatis) dari metadata_en.yaml
#   - label "CHAPTER I/II/III", caption "Figure"/"Table"
#   - figur pipeline (TikZ) + tabel jadwal dirender versi Inggris ke
#     figures_en/; resnet/vit dipakai bersama dari gambar/
#   - reference.docx (Times New Roman 12, A4, margin 4-3-3-3 cm, spasi ganda)
# Usage:  ./build_docx_en.sh
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_DIR/draft_proposal_skripsi_en.md"
BODY="$SCRIPT_DIR/body_docx_en.md"
REF="$SCRIPT_DIR/reference.docx"
OUT="$SCRIPT_DIR/draft_proposal_skripsi_en.docx"

# Render PNG versi Inggris (pipeline TikZ + tabel jadwal) ke figures_en/
"$SCRIPT_DIR/render_figures.sh" "$SRC" "$SCRIPT_DIR/figures_en" >/dev/null

# Bangun reference.docx + body_docx_en.md (front matter + isi)
python3 "$SCRIPT_DIR/_docx_build.py" en

echo "[docx:en] pandoc -> docx ..."
pandoc "$BODY" \
    --from=markdown+tex_math_dollars+tex_math_single_backslash+implicit_figures+raw_attribute+fancy_lists+startnum \
    --to=docx \
    --reference-doc="$REF" \
    --shift-heading-level-by=-1 \
    --resource-path="$SCRIPT_DIR:$REPO_DIR" \
    --output="$OUT"

rm -f "$BODY"
echo "[docx:en] Done: $OUT"

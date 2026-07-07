#!/usr/bin/env bash
# =================================================================
# Build: draft_proposal_skripsi.md -> draft_proposal_skripsi.docx
#
# Berbeda dengan konversi pandoc polos (yang membuat struktur Word
# "hancur": tanpa front matter, bab jadi Heading 2, tanpa daftar isi),
# skrip ini memakai _docx_build.py untuk:
#   - menyusun front matter dari metadata.yaml (Sampul, Lembar
#     Persetujuan, Abstrak ID + Abstract EN, Daftar Isi otomatis Word)
#   - memberi label "BAB I/II/III" dan, lewat --shift-heading-level-by=-1,
#     menjadikan bab = Heading 1, subbab = Heading 2/3 (outline benar)
#   - menukar figure/tabel LaTeX -> PNG dan \refitem -> daftar pustaka
#   - memakai reference.docx (Times New Roman 12, A4, margin 4-3-3-3 cm,
#     spasi ganda) sesuai Pedoman Undiksha
# Usage:  ./build_docx.sh
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BODY="$SCRIPT_DIR/body_docx.md"
REF="$SCRIPT_DIR/reference.docx"
OUT="$SCRIPT_DIR/draft_proposal_skripsi.docx"

# Render PNG (figur TikZ + tabel jadwal) bila belum ada
"$SCRIPT_DIR/render_figures.sh" >/dev/null

# Bangun reference.docx + body_docx.md (front matter + isi)
python3 "$SCRIPT_DIR/_docx_build.py"

echo "[docx] pandoc -> docx ..."
pandoc "$BODY" \
    --from=markdown+tex_math_dollars+tex_math_single_backslash+implicit_figures+raw_attribute \
    --to=docx \
    --reference-doc="$REF" \
    --shift-heading-level-by=-1 \
    --resource-path="$SCRIPT_DIR:$REPO_DIR" \
    --output="$OUT"

rm -f "$BODY"
echo "[docx] Done: $OUT"

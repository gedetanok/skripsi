#!/usr/bin/env bash
# =================================================================
# Build: makalah_soft_exudate.md -> makalah_soft_exudate.pdf
# Usage:  ./build_makalah.sh
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="$REPO_DIR/makalah_soft_exudate.md"
META="$SCRIPT_DIR/metadata_makalah.yaml"
TEMPLATE="$SCRIPT_DIR/template_makalah.tex"
OUT="$SCRIPT_DIR/makalah_soft_exudate.pdf"

if [[ ! -f "$SRC" ]]; then
    echo "Error: source not found: $SRC" >&2
    exit 1
fi

# 1. Render diagram TikZ + tabel dari markdown -> PNG
echo "[build] render figures ..."
"$SCRIPT_DIR/render_figures.sh" "$SRC" "$SCRIPT_DIR/figures"

# 2. Markdown -> PDF (xelatex). ## -> \section via shift-heading-level.
echo "[build] pandoc -> PDF via xelatex ..."
pandoc "$SRC" \
    --from=markdown+tex_math_dollars+tex_math_single_backslash+fancy_lists+startnum \
    --to=latex \
    --metadata-file="$META" \
    --template="$TEMPLATE" \
    --pdf-engine=xelatex \
    --shift-heading-level-by=-1 \
    --top-level-division=section \
    --resource-path="$SCRIPT_DIR:$REPO_DIR" \
    --variable=graphics-path="$REPO_DIR/" \
    --variable=graphics-path="$SCRIPT_DIR/figures/" \
    --output="$OUT"

echo "[build] Done: $OUT"

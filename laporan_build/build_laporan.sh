#!/usr/bin/env bash
# =================================================================
# Build: laporan_soft_exudate.md -> laporan_soft_exudate.pdf
# Format laporan BAB I-IV (book class). Usage:  ./build_laporan.sh
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="$REPO_DIR/laporan_soft_exudate.md"
META="$SCRIPT_DIR/metadata_laporan.yaml"
TEMPLATE="$SCRIPT_DIR/template_laporan.tex"
OUT="$SCRIPT_DIR/laporan_soft_exudate.pdf"

if [[ ! -f "$SRC" ]]; then
    echo "Error: source not found: $SRC" >&2
    exit 1
fi

echo "[build] render figures ..."
"$SCRIPT_DIR/render_figures.sh" "$SRC" "$SCRIPT_DIR/figures"

# Headings: "## BAB I PENDAHULUAN" -> "## PENDAHULUAN", "### 1.1 Judul" -> "### Judul"
# (LaTeX menomori otomatis via \chapter/\section). Shift ## -> \chapter.
BODY="$SCRIPT_DIR/_body_laporan.md"
sed -E \
    -e 's/^(#{2})[[:space:]]+BAB[[:space:]]+[IVX]+[[:space:]]+/\1 /' \
    -e 's/^(#{3,6})[[:space:]]+[0-9]+(\.[0-9]+)+[[:space:]]+/\1 /' \
    "$SRC" > "$BODY"

echo "[build] pandoc -> PDF via xelatex ..."
pandoc "$BODY" \
    --from=markdown+tex_math_dollars+tex_math_single_backslash+fancy_lists+startnum \
    --to=latex \
    --metadata-file="$META" \
    --template="$TEMPLATE" \
    --pdf-engine=xelatex \
    --shift-heading-level-by=-1 \
    --top-level-division=chapter \
    --resource-path="$SCRIPT_DIR:$REPO_DIR" \
    --variable=graphics-path="$REPO_DIR/" \
    --variable=graphics-path="$SCRIPT_DIR/figures/" \
    --output="$OUT"

rm -f "$BODY"
echo "[build] Done: $OUT"

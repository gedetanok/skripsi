#!/usr/bin/env bash
# =================================================================
# Build script: draft_proposal_skripsi.md -> draft_proposal_skripsi.pdf
# Usage:  ./build.sh
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SRC="$REPO_DIR/draft_proposal_skripsi.md"
BODY="$SCRIPT_DIR/body.md"
META="$SCRIPT_DIR/metadata.yaml"
TEMPLATE="$SCRIPT_DIR/template.tex"
OUT="$SCRIPT_DIR/draft_proposal_skripsi.pdf"

if [[ ! -f "$SRC" ]]; then
    echo "Error: source file not found: $SRC" >&2
    exit 1
fi

# Strip the top 7 lines (H1 title + working title + horizontal rule) so
# pandoc sees BAB headings as the top level, then strip manual numbering
# prefixes from headings (e.g. "## CHAPTER I INTRODUCTION" -> "## INTRODUCTION",
# "### 1.1 Background" -> "### Background"). LaTeX auto-numbers these via
# \chapter / \section so leaving manual numbers in would produce duplicates.
tail -n +8 "$SRC" | sed -E \
    -e 's/^(#{2,6})[[:space:]]+CHAPTER[[:space:]]+[IVX]+[[:space:]]+/\1 /' \
    -e 's/^(#{3,6})[[:space:]]+[0-9]+(\.[0-9]+)+[[:space:]]+/\1 /' \
    > "$BODY"

echo "[build] pandoc -> PDF via xelatex ..."
pandoc "$BODY" \
    --from=markdown+tex_math_dollars+tex_math_single_backslash \
    --to=latex \
    --metadata-file="$META" \
    --template="$TEMPLATE" \
    --pdf-engine=xelatex \
    --shift-heading-level-by=-1 \
    --top-level-division=chapter \
    --resource-path="$SCRIPT_DIR:$REPO_DIR" \
    --output="$OUT"

rm -f "$BODY"

echo "[build] Done: $OUT"

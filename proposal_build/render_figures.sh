#!/usr/bin/env bash
# =================================================================
# Render SEMUA diagram TikZ + tabel jadwal dari draft_proposal_skripsi.md
# menjadi PNG resolusi tinggi di proposal_build/figures/.
#
# Bersifat generik: setiap \begin{figure}...\label{fig:X}...\end{figure}
# dirender ke fig_X.png; tabel jadwal ke fig_jadwal.png. Tambah diagram
# baru di markdown -> otomatis ikut ter-render tanpa mengubah skrip ini.
# Usage:  ./render_figures.sh
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Argumen opsional: $1 = file sumber markdown, $2 = folder output PNG.
# Default: sumber Indonesia -> proposal_build/figures/.
SRC="${1:-$REPO_DIR/draft_proposal_skripsi.md}"
FIGDIR="${2:-$SCRIPT_DIR/figures}"
mkdir -p "$FIGDIR"

echo "[fig] mengekstrak blok TikZ + tabel ..."
SRC="$SRC" FIGDIR="$FIGDIR" python3 <<'PY'
import os, re
src = open(os.environ["SRC"], encoding="utf-8").read()
figdir = os.environ["FIGDIR"]

PRE_TIKZ = r"""\documentclass[border=8pt]{standalone}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,fit,backgrounds}
\hyphenpenalty=10000\exhyphenpenalty=10000\tolerance=2000
\begin{document}
"""
PRE_TBL = r"""\documentclass[border=8pt]{standalone}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{array}
\usepackage[table]{xcolor}
\usepackage{calc}
\begin{document}
"""
POST = "\n\\end{document}\n"

names = []
# semua figure -> fig_<labelsuffix>.tex (pakai tikzpicture di dalamnya)
for m in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}', src, re.DOTALL):
    env = m.group(0)
    lab = re.search(r'\\label\{fig:([A-Za-z0-9_]+)\}', env)
    tik = re.search(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', env, re.DOTALL)
    if not (lab and tik):
        continue
    name = "fig_" + lab.group(1)
    open(f"{figdir}/{name}.tex", "w", encoding="utf-8").write(PRE_TIKZ + tik.group(0) + POST)
    names.append(name)

# semua tabel -> fig_<labelsuffix>.tex (key dari \label{tab:X})
for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}', src, re.DOTALL):
    env = m.group(0)
    lab = re.search(r'\\label\{tab:([A-Za-z0-9_]+)\}', env)
    inner = re.search(r'(\\small.*?\\end\{tabular\})', env, re.DOTALL)
    if not (lab and inner):
        continue
    name = "fig_" + lab.group(1)
    open(f"{figdir}/{name}.tex", "w", encoding="utf-8").write(PRE_TBL + inner.group(1) + POST)
    names.append(name)

print("  blok: " + ", ".join(names))
open(f"{figdir}/.fignames", "w").write("\n".join(names) + "\n")
PY

cd "$FIGDIR"
while read -r f; do
    [[ -z "$f" ]] && continue
    echo "[fig] kompilasi $f ..."
    xelatex -interaction=nonstopmode -halt-on-error "$f.tex" >"$f.log" 2>&1
    pdftocairo -png -r 250 -singlefile "$f.pdf" "$f"
    echo "[fig]   -> $f.png"
done < .fignames

rm -f *.aux *.log *.tex *.pdf .fignames
echo "[fig] Selesai. PNG ada di $FIGDIR/"

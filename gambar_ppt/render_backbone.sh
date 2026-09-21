#!/usr/bin/env bash
# =================================================================
# Render diagram kedua backbone jadi PNG resolusi tinggi untuk slide.
#
#   resnet50.png  -> diekstrak dari blok TikZ \label{fig:resnet}
#                    di draft_proposal_skripsi.md (selalu sinkron)
#   vit_b16.png   -> digambar di skrip ini, gaya sama dgn ResNet
#                    (proposal masih memakai screenshot paper)
#
# Usage:  ./gambar_ppt/render_backbone.sh
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_DIR/draft_proposal_skripsi.md"
OUT="$SCRIPT_DIR/backbone"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"

PREAMBLE='\documentclass[border=8pt]{standalone}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,fit,backgrounds}
\hyphenpenalty=10000\exhyphenpenalty=10000\tolerance=2000
\begin{document}'

# ----------------------------------------------------------------- #
# 1. ResNet-50: ambil blok TikZ dari markdown                        #
# ----------------------------------------------------------------- #
echo "[bb] mengekstrak TikZ ResNet-50 dari draft ..."
SRC="$SRC" WORK="$WORK" PREAMBLE="$PREAMBLE" python3 <<'PY'
import os, re
src = open(os.environ["SRC"], encoding="utf-8").read()
for m in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}', src, re.DOTALL):
    env = m.group(0)
    if not re.search(r'\\label\{fig:resnet\}', env):
        continue
    tik = re.search(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', env, re.DOTALL)
    path = os.path.join(os.environ["WORK"], "resnet50.tex")
    open(path, "w", encoding="utf-8").write(
        os.environ["PREAMBLE"] + "\n" + tik.group(0) + "\n\\end{document}\n")
    break
else:
    raise SystemExit("blok \\label{fig:resnet} tidak ditemukan di draft")
PY

# ----------------------------------------------------------------- #
# 2. ViT-B/16: digambar di sini                                      #
# ----------------------------------------------------------------- #
cat > "$WORK/vit_b16.tex" <<'TEX'
\documentclass[border=8pt]{standalone}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,fit,backgrounds}
\hyphenpenalty=10000\exhyphenpenalty=10000\tolerance=2000
\begin{document}
\begin{tikzpicture}[
  font=\footnotesize,
  box/.style={draw, rounded corners=2pt, minimum height=1.9cm, text width=2.0cm, align=center, font=\scriptsize, inner sep=2pt},
  arr/.style={-{Stealth[length=2mm]}, semithick},
  dim/.style={font=\tiny, text=gray}
]
\node[box, fill=gray!12] (in) {Input fundus\\$224{\times}224{\times}3$};
\node[box, fill=cyan!12, right=0.4cm of in] (pa) {patch\\$16{\times}16$\\$N=196$};
\node[box, fill=orange!12, right=0.4cm of pa] (pr) {linear\\projection\\$\to 768$-d};
\node[box, fill=orange!18, right=0.4cm of pr] (em) {$+$ position emb.\\prepend [class]};
\node[box, fill=orange!24, right=0.4cm of em] (en) {Transformer\\encoder\\$\times 12$};
\node[box, fill=green!12, right=0.4cm of en] (cl) {[class]\\token};
\node[box, fill=red!12, right=0.4cm of cl] (hd) {MLP head\\5 (ICDR)};
\foreach \a/\b in {in/pa,pa/pr,pr/em,em/en,en/cl,cl/hd}{\draw[arr] (\a)--(\b);}
\foreach \n/\d in {pa/{$196$ patch},pr/{$196{\times}768$},em/{$197{\times}768$},cl/{768-d}}{\node[dim, below=2pt of \n] {\d};}
\coordinate (bc) at ($(pr.south)!0.5!(em.south)+(0,-2.1)$);
\begin{scope}[font=\tiny,
   cb/.style={draw, rounded corners=1.5pt, fill=blue!12, text width=1.9cm, align=center, minimum height=0.5cm, inner sep=1.5pt},
   ar/.style={-{Stealth[length=1.5mm]}, semithick}]
\node (z) at (bc) {$z$};
\node[cb, fill=yellow!18, below=2.5mm of z] (n1) {layer norm};
\node[cb, fill=green!18, below=2.5mm of n1] (at) {multi-head\\self-attention};
\node[draw, circle, below=2.5mm of at, inner sep=0.8pt] (s1) {$+$};
\node[cb, fill=yellow!18, below=2.5mm of s1] (n2) {layer norm};
\node[cb, below=2.5mm of n2] (ml) {MLP (GELU)};
\node[draw, circle, below=2.5mm of ml, inner sep=0.8pt] (s2) {$+$};
\node[below=2.5mm of s2] (out) {$z'$};
\foreach \a/\b in {z/n1,n1/at,at/s1,s1/n2,n2/ml,ml/s2,s2/out}{\draw[ar] (\a)--(\b);}
\draw[ar] (z.east) -- ++(1.15,0) |- (s1.east);
\draw[ar] (s1.west) -- ++(-1.15,0) |- (s2.west);
\node[right=1.2cm of at, align=left] {residual\\$z$};
\node[left=1.2cm of ml, align=right] {residual};
\node[below=3pt of out, font=\scriptsize, align=center]
  {Blok \emph{Transformer encoder}: $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(QK^{\top}/\sqrt{d_k}\right)V$\\
   ViT-B/16: $L=12$ blok, 12 \emph{head}, $d=768$};
\end{scope}
\draw[arr, dashed, gray] (en.south) -- (z.north);
\end{tikzpicture}
\end{document}
TEX

# ----------------------------------------------------------------- #
cd "$WORK"
for f in resnet50 vit_b16; do
    echo "[bb] kompilasi $f ..."
    xelatex -interaction=nonstopmode -halt-on-error "$f.tex" >"$f.log" 2>&1
    pdftocairo -png -r 250 -singlefile "$f.pdf" "$f"
    cp "$f.png" "$OUT/$f.png"
    echo "[bb]   -> $OUT/$f.png"
done

echo "[bb] Selesai."

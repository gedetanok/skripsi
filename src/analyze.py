"""Analisis statistik lintas dataset sesuai Bagian 3.7 proposal.

Masukannya satu CSV per run berkolom dataset, technique, backbone, seed, qwk.
QWK tiap sel dirata-ratakan antar-seed lebih dulu, lalu matriks konfigurasi x
dataset itulah yang diuji.

Tahapannya:
  1. Uji omnibus Friedman atas seluruh konfigurasi lintas dataset.
  2. Urutan rata-rata peringkat, dilaporkan secara deskriptif.
  3. Faktor backbone: QWK dirata-ratakan atas teknik, lalu Wilcoxon signed-rank
     sebagai satu perbandingan terencana.
  4. Faktor preprocessing: QWK dirata-ratakan atas backbone, Friedman, lalu tiap
     teknik dibandingkan terhadap baseline dengan Wilcoxon + koreksi Holm.
  5. Interaksi: urutan preprocessing diperiksa terpisah di tiap backbone.

Jalankan:
    python -m src.analyze --runs results/run_144_qwk.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def cell_means(runs: pd.DataFrame) -> pd.DataFrame:
    """Rerata antar-seed per sel, dibentuk jadi matriks konfigurasi x dataset."""
    runs = runs.copy()
    runs["config"] = runs.technique + " x " + runs.backbone
    return runs.pivot_table(index="config", columns="dataset", values="qwk", aggfunc="mean")


def friedman(matrix: pd.DataFrame, label: str) -> float:
    """Friedman atas baris matriks (tiap kolom satu dataset)."""
    stat, p = stats.friedmanchisquare(*[matrix.loc[i].to_numpy() for i in matrix.index])
    n_methods, n_datasets = matrix.shape
    print(f"\nFriedman {label}: chi2 = {stat:.3f}, p = {p:.4f} "
          f"(k={n_methods} metode, N={n_datasets} dataset)")
    print("  signifikan pada 0,05" if p < 0.05 else "  TIDAK signifikan pada 0,05")
    return p


def average_ranks(matrix: pd.DataFrame) -> pd.Series:
    """Rata-rata peringkat tiap metode; peringkat 1 = QWK tertinggi dalam dataset."""
    return matrix.rank(ascending=False, axis=0).mean(axis=1).sort_values()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("results/run_144_qwk.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results/analysis"))
    args = p.parse_args()

    runs = pd.read_csv(args.runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(runs.dataset.unique())
    print("=" * 70)
    print(f"{len(datasets)} dataset: {', '.join(datasets)}")
    print(f"{runs.groupby(['dataset','technique','backbone']).ngroups} sel, "
          f"{len(runs)} run, {runs.seed.nunique()} seed per sel")
    print("=" * 70)

    # --- 1-2. omnibus seluruh konfigurasi ------------------------------------
    matrix = cell_means(runs)
    matrix.round(4).to_csv(args.out_dir / "matriks_qwk.csv")
    print("\nMatriks QWK (rerata antar-seed)")
    print(matrix.round(4).to_string())

    friedman(matrix, "seluruh konfigurasi")
    ranks = average_ranks(matrix)
    print("\nRata-rata peringkat (deskriptif, 1 = terbaik):")
    for config, rank in ranks.items():
        print(f"  {rank:5.2f}  {config}")

    # --- 3. faktor backbone ---------------------------------------------------
    by_backbone = runs.pivot_table(index="backbone", columns="dataset", values="qwk", aggfunc="mean")
    print("\n" + "-" * 70)
    print("FAKTOR BACKBONE (dirata-ratakan atas enam teknik)")
    print(by_backbone.round(4).to_string())

    a, b = by_backbone.loc["resnet50"], by_backbone.loc["vit_b16"]
    stat, pval = stats.wilcoxon(a, b)
    print(f"\n  ResNet-50 menang di {int((a > b).sum())} dari {len(a)} dataset")
    print(f"  selisih median: {float((a - b).median()):+.4f}")
    print(f"  Wilcoxon signed-rank: W = {stat:.1f}, p = {pval:.4f}")
    if len(a) < 6:
        print(f"  CATATAN: pada N={len(a)} dataset, p terkecil yang mungkin adalah "
              f"{2 / 2 ** len(a):.4f}; signifikansi belum dapat dicapai.")

    # --- 4. faktor preprocessing ---------------------------------------------
    by_technique = runs.pivot_table(index="technique", columns="dataset", values="qwk", aggfunc="mean")
    print("\n" + "-" * 70)
    print("FAKTOR PREPROCESSING (dirata-ratakan atas kedua backbone)")
    print(by_technique.round(4).to_string())
    friedman(by_technique, "preprocessing")

    print("\n  Tiap teknik dibandingkan terhadap baseline (Wilcoxon + Holm):")
    others = [t for t in by_technique.index if t != "baseline"]
    pvals = [stats.wilcoxon(by_technique.loc[t], by_technique.loc["baseline"])[1] for t in others]
    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    for technique, p_raw, p_corr, sig in zip(others, pvals, p_adj, reject):
        delta = float((by_technique.loc[technique] - by_technique.loc["baseline"]).mean())
        print(f"    {technique:<18} selisih {delta:+.4f}  p = {p_raw:.4f}  "
              f"p_Holm = {p_corr:.4f}  {'signifikan' if sig else 'tidak'}")

    # --- 5. interaksi ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("INTERAKSI: urutan preprocessing di dalam tiap backbone")
    for backbone in sorted(runs.backbone.unique()):
        subset = runs[runs.backbone == backbone]
        m = subset.pivot_table(index="technique", columns="dataset", values="qwk", aggfunc="mean")
        order = average_ranks(m)
        spread = m.mean(axis=1).max() - m.mean(axis=1).min()
        print(f"\n  {backbone} (sebaran QWK rerata {spread:.4f}):")
        for technique, rank in order.items():
            print(f"    {rank:5.2f}  {technique:<18} QWK {m.loc[technique].mean():.4f}")

    ranks.round(3).to_csv(args.out_dir / "peringkat_rata_rata.csv", header=["avg_rank"])
    print(f"\nkeluaran disimpan di {args.out_dir}")


if __name__ == "__main__":
    main()

"""Jalankan seluruh konfigurasi preprocessing x backbone pada satu dataset.

Setiap sel dijalankan pada beberapa seed dan QWK yang dilaporkan adalah
reratanya. Ini bukan kehati-hatian berlebihan: pengukuran pada APTOS menunjukkan
simpangan baku antar-seed 0,0114 sedangkan jarak antar-konfigurasi hanya 0,0070,
sehingga peringkat dari satu seed berbalik arah ketika seed diganti.

Runner ini resumable: konfigurasi yang metrics.json-nya sudah ada dilewati.
Sifat itu penting karena sesi Kaggle dibatasi 12 jam, sehingga sebuah grid yang
terputus di tengah bisa dilanjutkan dengan menjalankan ulang perintah yang sama
alih-alih mengulang dari nol.

Kegagalan satu konfigurasi tidak menghentikan sisanya; galatnya dicatat dan
runner lanjut ke konfigurasi berikutnya, supaya satu bug pada satu kombinasi
tidak membatalkan hasil semalam.

Jalankan:
    python -m src.run_grid --dataset aptos2019 --cache-dir cache --out-dir results
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

from src.train import run, run_name


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def summarise(out_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kumpulkan metrik seluruh run, per run dan dirata-ratakan antar-seed."""
    rows = []
    for metrics_file in sorted((out_dir / dataset).glob("*/metrics.json")):
        m = json.loads(metrics_file.read_text())
        rows.append({
            "technique": m["technique"], "backbone": m["backbone"], "seed": m["seed"],
            "qwk": m["test"]["qwk"], "accuracy": m["test"]["accuracy"],
            "macro_f1": m["test"]["macro_f1"],
            "epochs": m["epochs_ran"], "minutes": round(m["train_minutes"], 1),
        })
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    per_run = pd.DataFrame(rows)
    agg = (per_run.groupby(["technique", "backbone"])
                  .agg(qwk_mean=("qwk", "mean"), qwk_std=("qwk", "std"),
                       accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
                       n_seeds=("seed", "nunique"), minutes=("minutes", "sum"))
                  .sort_values("qwk_mean", ascending=False)
                  .round(4).reset_index())
    return per_run, agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--epochs", type=int)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, help="batasi citra per partisi (uji cepat)")
    p.add_argument("--only-backbone", choices=["resnet50", "vit_b16"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                   help="seed yang dijalankan per sel; reratanya yang dilaporkan")
    p.add_argument("--time-budget-hours", type=float,
                   help="berhenti rapi sebelum batas ini terlampaui, agar sesi batch "
                        "sempat menyimpan hasil yang sudah jadi alih-alih dibunuh "
                        "sistem dan kehilangan semuanya")
    args = p.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    backbones = [args.only_backbone] if args.only_backbone else list(cfg["backbones"])

    combos = [(t, b, s) for t in cfg["preprocessing"] for b in backbones for s in args.seeds]
    print(f"{args.dataset}: {len(combos)} run "
          f"({len(cfg['preprocessing'])} preprocessing x {len(backbones)} backbone "
          f"x {len(args.seeds)} seed)\n", flush=True)

    started = time.time()
    done = failed = skipped = 0
    for i, (technique, backbone, seed) in enumerate(combos, 1):
        elapsed = (time.time() - started) / 3600
        if args.time_budget_hours and elapsed > args.time_budget_hours:
            print(f"anggaran waktu {args.time_budget_hours} jam terlampaui "
                  f"({elapsed:.1f} jam); berhenti agar hasil yang sudah jadi tersimpan. "
                  "Jalankan ulang perintah yang sama untuk melanjutkan.", flush=True)
            break

        name = run_name(technique, backbone, seed)
        if (args.out_dir / args.dataset / name / "metrics.json").exists():
            print(f"[{i}/{len(combos)}] {name} -- sudah ada, dilewati", flush=True)
            skipped += 1
            continue

        print(f"[{i}/{len(combos)}] {name}", flush=True)
        try:
            run(dataset=args.dataset, technique=technique, backbone=backbone,
                cache_dir=args.cache_dir, out_dir=args.out_dir,
                epochs=args.epochs or train_cfg["max_epochs"],
                batch_size=train_cfg["batch_size"], lr=float(train_cfg["lr"]),
                weight_decay=float(train_cfg["weight_decay"]),
                warmup=train_cfg["warmup_epochs"],
                patience=train_cfg["early_stopping_patience"],
                workers=args.workers, seed=seed, limit=args.limit)
            done += 1
        except Exception:
            failed += 1
            print(f"  GAGAL {technique} x {backbone}:\n{traceback.format_exc()}", flush=True)
        print(flush=True)

    print(f"ringkasan: {done} selesai, {skipped} dilewati, {failed} gagal\n", flush=True)

    per_run, agg = summarise(args.out_dir, args.dataset)
    if not agg.empty:
        per_run.to_csv(args.out_dir / args.dataset / "runs.csv", index=False)
        agg.to_csv(args.out_dir / args.dataset / "summary.csv", index=False)
        print(agg.to_string(index=False), flush=True)

        if agg.n_seeds.min() > 1:
            spread = agg.qwk_mean.max() - agg.qwk_mean.min()
            noise = float(agg.qwk_std.mean())
            print(f"\njarak antar-konfigurasi : {spread:.4f}")
            print(f"derau antar-seed        : {noise:.4f}")
            if noise >= spread:
                print("derau melebihi jarak: konfigurasi tidak terbedakan pada dataset ini.")


if __name__ == "__main__":
    main()

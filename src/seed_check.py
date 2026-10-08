"""Ukur seberapa besar QWK berubah hanya karena pergantian seed.

Seluruh grid pada satu dataset hanya terbentang 0,038 QWK, sedangkan selisih
konfigurasi terbaik terhadap baseline sekitar 0,015. Angka sekecil itu hanya
berarti bila variasi antar-seed lebih kecil lagi. Skrip ini melatih beberapa
konfigurasi terpilih pada beberapa seed, lalu melaporkan rerata dan simpangan
bakunya, sehingga terlihat apakah urutan peringkat bertahan atau tenggelam.

Pembacaan hasilnya: bila simpangan baku antar-seed sebanding atau lebih besar
daripada jarak antar-konfigurasi, maka peringkat pada satu seed tidak dapat
ditafsirkan, dan desainnya perlu dilaporkan dengan beberapa seed per sel.

Jalankan:
    python -m src.seed_check --dataset aptos2019 \\
        --configs lab_ace:resnet50 clahe:resnet50 baseline:resnet50 \\
        --seeds 42 43 44 --cache-dir cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.train import run, run_name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--configs", nargs="+", required=True,
                   help="daftar 'teknik:backbone', mis. lab_ace:resnet50")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--config-file", type=Path, default=Path("configs/experiment.yaml"))
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument("--out-dir", type=Path, default=Path("seed_check"))
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    train_cfg = yaml.safe_load(args.config_file.read_text())["training"]
    pairs = [tuple(c.split(":")) for c in args.configs]

    rows = []
    total = len(pairs) * len(args.seeds)
    n = 0
    for technique, backbone in pairs:
        for seed in args.seeds:
            n += 1
            out_dir = args.out_dir / f"seed{seed}"
            done = out_dir / args.dataset / run_name(technique, backbone, seed) / "metrics.json"

            if done.exists():
                print(f"[{n}/{total}] {technique} x {backbone} seed={seed} -- sudah ada", flush=True)
                metrics = json.loads(done.read_text())
            else:
                print(f"[{n}/{total}] {technique} x {backbone} seed={seed}", flush=True)
                metrics = run(
                    dataset=args.dataset, technique=technique, backbone=backbone,
                    cache_dir=args.cache_dir, out_dir=out_dir,
                    epochs=train_cfg["max_epochs"], batch_size=train_cfg["batch_size"],
                    lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]),
                    warmup=train_cfg["warmup_epochs"],
                    patience=train_cfg["early_stopping_patience"],
                    workers=args.workers, seed=seed, limit=None)

            rows.append({"technique": technique, "backbone": backbone, "seed": seed,
                         "qwk": metrics["test"]["qwk"], "accuracy": metrics["test"]["accuracy"]})

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / f"{args.dataset}_runs.csv", index=False)

    summary = (df.groupby(["technique", "backbone"])
                 .qwk.agg(["mean", "std", "min", "max"])
                 .sort_values("mean", ascending=False).round(4))
    summary.to_csv(args.out_dir / f"{args.dataset}_summary.csv")

    print("\n" + "=" * 62)
    print("QWK per konfigurasi lintas seed")
    print("=" * 62)
    print(summary.to_string())

    spread = summary["mean"].max() - summary["mean"].min()
    noise = float(df.groupby(["technique", "backbone"]).qwk.std().mean())
    print(f"\njarak antar-konfigurasi (rerata terbaik - terburuk) : {spread:.4f}")
    print(f"simpangan baku antar-seed (rerata)                  : {noise:.4f}")

    if noise >= spread:
        print("\nKESIMPULAN: derau seed setara atau melebihi jarak antar-konfigurasi.")
        print("Peringkat dari satu seed TIDAK dapat ditafsirkan; laporkan beberapa")
        print("seed per sel, atau nyatakan bahwa konfigurasi tidak terbedakan.")
    elif noise >= spread / 3:
        print("\nKESIMPULAN: derau seed cukup besar dibanding jarak antar-konfigurasi.")
        print("Peringkat teratas mungkin bertahan, tetapi selisih kecil di tengah")
        print("tabel tidak bermakna. Sebaiknya tetap pakai beberapa seed.")
    else:
        print("\nKESIMPULAN: derau seed jauh lebih kecil daripada jarak antar-konfigurasi.")
        print("Peringkat dari satu seed dapat ditafsirkan; desain satu seed memadai.")


if __name__ == "__main__":
    main()

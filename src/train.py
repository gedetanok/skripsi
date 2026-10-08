"""Latih dan evaluasi satu konfigurasi (dataset x preprocessing x backbone).

Protokol mengikuti Bagian 3.6 proposal: AdamW, pemanasan linear tiga epoch lalu
peluruhan kosinus, maksimum 50 epoch dengan early stopping pada QWK validasi
(patience 10), dan checkpoint dengan QWK validasi terbaik yang dibawa ke
evaluasi test.

Keluaran per run, di bawah results/<dataset>/<teknik>__<backbone>/:

  metrics.json      akurasi, QWK, macro-F1, confusion matrix, biaya komputasi
  predictions.csv   prediksi per citra pada partisi test
  history.json      metrik tiap epoch

predictions.csv penting untuk tahap analisis: uji McNemar (Bagian 3.7)
membandingkan dua konfigurasi pada pasangan prediksi per citra, sehingga
prediksi mentahnya harus tersimpan, bukan hanya metrik agregatnya.

Jalankan:
    python -m src.train --dataset aptos2019 --technique clahe --backbone resnet50
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score)
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import DRDataset, build_sampler, class_weights
from src.models import build_model, count_parameters, inference_time_ms

MANIFEST_DIR = Path("manifests")


def run_name(technique: str, backbone: str, seed: int) -> str:
    """Nama folder satu run. Seed ikut masuk karena desainnya multi-seed:
    tanpa itu, run seed kedua menimpa hasil seed pertama."""
    return f"{technique}__{backbone}__seed{seed}"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_lambda(epoch: int, warmup: int, total: int, min_ratio: float) -> float:
    """Pemanasan linear selama `warmup` epoch, lalu kosinus sampai min_ratio."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_ratio + (1 - min_ratio) * cosine


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ids, trues, preds = [], [], []
    for x, y, image_id in loader:
        logits = model(x.to(device, non_blocking=True))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.numpy())
        ids.extend(image_id)
    return np.array(ids), np.concatenate(trues), np.concatenate(preds)


def score(trues: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(trues, preds)),
        "qwk": float(cohen_kappa_score(trues, preds, weights="quadratic")),
        "macro_f1": float(f1_score(trues, preds, average="macro", zero_division=0)),
    }


def run(dataset: str, technique: str, backbone: str, cache_dir: Path, out_dir: Path,
        epochs: int, batch_size: int, lr: float, weight_decay: float, warmup: int,
        patience: int, workers: int, seed: int, limit: int | None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    manifest = pd.read_csv(MANIFEST_DIR / f"{dataset}.csv")
    if limit:  # jalur uji cepat, bukan untuk hasil yang dilaporkan
        manifest = manifest.groupby("split", group_keys=False).head(limit)

    parts = {s: manifest[manifest.split == s] for s in ("train", "val", "test")}
    cache = Path(cache_dir) / dataset

    loaders = {}
    for split, rows in parts.items():
        ds = DRDataset(rows, cache, technique, augment=(split == "train"))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            sampler=build_sampler(rows) if split == "train" else None,
            shuffle=False, num_workers=workers, pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    model = build_model(backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(parts["train"]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: lr_lambda(e, warmup, epochs, min_ratio=1e-6 / lr)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_qwk, best_state, best_epoch, history = -np.inf, None, -1, []
    started = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y, _ in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
        scheduler.step()

        _, val_true, val_pred = predict(model, loaders["val"], device)
        val = score(val_true, val_pred)
        history.append({"epoch": epoch, "train_loss": total_loss / len(parts["train"]), **val})
        print(f"  epoch {epoch + 1:>2}/{epochs}  loss={history[-1]['train_loss']:.4f}  "
              f"val_qwk={val['qwk']:.4f}  acc={val['accuracy']:.4f}", flush=True)

        if val["qwk"] > best_qwk:
            best_qwk, best_epoch = val["qwk"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"  early stopping: QWK validasi tidak membaik selama {patience} epoch", flush=True)
            break

    model.load_state_dict(best_state)
    ids, test_true, test_pred = predict(model, loaders["test"], device)

    metrics = {
        "dataset": dataset, "technique": technique, "backbone": backbone,
        "test": score(test_true, test_pred),
        "val_qwk_best": float(best_qwk),
        "best_epoch": int(best_epoch),
        "epochs_ran": len(history),
        "confusion_matrix": confusion_matrix(test_true, test_pred, labels=range(5)).tolist(),
        "n_params": count_parameters(model),
        "inference_ms": inference_time_ms(model, device),
        "train_minutes": (time.time() - started) / 60,
        "device": device.type,
        "seed": seed,
    }

    run_dir = Path(out_dir) / dataset / run_name(technique, backbone, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    pd.DataFrame({"image_id": ids, "true": test_true, "pred": test_pred}).to_csv(
        run_dir / "predictions.csv", index=False)

    t = metrics["test"]
    print(f"SELESAI {dataset}/{technique}/{backbone}  "
          f"QWK={t['qwk']:.4f}  acc={t['accuracy']:.4f}  F1={t['macro_f1']:.4f}  "
          f"({metrics['train_minutes']:.1f} menit)", flush=True)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--technique", required=True)
    p.add_argument("--backbone", required=True, choices=["resnet50", "vit_b16"])
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="batasi jumlah citra per partisi (uji cepat)")
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()

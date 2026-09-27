"""Dataset PyTorch yang membaca cache citra hasil preprocessing.

Citra sudah di-crop, dikenakan teknik preprocessing, dan diubah ke 224x224 oleh
src/data/build_cache.py, sehingga di sini tinggal membaca berkas, mengenakan
augmentasi, dan menstandardisasi dengan statistik ImageNet.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD

NUM_CLASSES = 5


def build_transform(augment: bool) -> A.Compose:
    """Augmentasi Bagian 3.4.3; hanya dikenakan pada partisi train.

    Flip dan rotasi aman dipakai karena derajat ICDR tidak bergantung pada
    orientasi spasial; jitter kecerahan dan kontras meniru variasi pencahayaan
    antar lokasi pengambilan citra.
    """
    steps = []
    if augment:
        steps += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]
    steps += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(steps)


class DRDataset(Dataset):
    """Satu partisi dari satu dataset, dengan satu teknik preprocessing."""

    def __init__(self, manifest: pd.DataFrame, cache_dir: Path, technique: str,
                 augment: bool = False) -> None:
        self.rows = manifest.reset_index(drop=True)
        self.dir = Path(cache_dir) / technique
        self.transform = build_transform(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows.iloc[i]
        path = self.dir / f"{row.image_id}.jpg"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"citra cache tidak terbaca: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"], int(row.grade), row.image_id


def class_counts(manifest: pd.DataFrame) -> np.ndarray:
    return np.array([(manifest.grade == k).sum() for k in range(NUM_CLASSES)], dtype=np.float64)


def class_weights(manifest: pd.DataFrame) -> torch.Tensor:
    """Bobot kelas sebanding 1/sqrt(n_k), dinormalisasi agar berjumlah K.

    Akar kuadrat dipakai, bukan 1/n_k, agar kelas paling langka tidak
    mendominasi gradien; pada EyePACS rasio ketidakseimbangannya mencapai 34
    (Bagian 3.2.2).
    """
    counts = np.maximum(class_counts(manifest), 1.0)
    w = 1.0 / np.sqrt(counts)
    w = w / w.sum() * NUM_CLASSES
    return torch.tensor(w, dtype=torch.float32)


def build_sampler(manifest: pd.DataFrame) -> WeightedRandomSampler:
    """Weighted random sampler dengan bobot per citra sebanding 1/sqrt(n_k)."""
    counts = np.maximum(class_counts(manifest), 1.0)
    per_class = 1.0 / np.sqrt(counts)
    weights = per_class[manifest.grade.to_numpy()]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(manifest),
        replacement=True,
    )

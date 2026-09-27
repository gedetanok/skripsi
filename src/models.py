"""Kedua backbone yang dibandingkan, dengan kepala klasifikasi lima kelas.

Bagian 3.5 proposal: ResNet-50 dari torchvision dengan bobot ImageNet-1k, dan
ViT-B/16 dari timm dengan bobot ImageNet-21k. Kepala klasifikasi asli diganti
lapisan linear baru ke lima logit ICDR, dan seluruh parameter dilatih bersama
tanpa pembekuan lapisan.
"""

from __future__ import annotations

import timm
import torch
import torchvision
from torch import nn

NUM_CLASSES = 5


def build_model(backbone: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    if backbone == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone == "vit_b16":
        return timm.create_model(
            "vit_base_patch16_224.augreg_in21k",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    raise KeyError(f"backbone tidak dikenal: {backbone!r}")


def count_parameters(model: nn.Module) -> int:
    """Jumlah parameter terlatih; dilaporkan sebagai variabel biaya komputasi (Tabel 3.3)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def inference_time_ms(model: nn.Module, device: torch.device, runs: int = 50) -> float:
    """Waktu inferensi rata-rata per citra, variabel terikat ketiga di Tabel 3.3."""
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(10):  # pemanasan
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if start is not None:
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(runs):
            model(x)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / runs

    import time
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    return (time.perf_counter() - t0) / runs * 1000

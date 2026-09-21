"""Pipeline dasar dan keenam varian preprocessing untuk eksperimen DR grading.

Modul ini adalah sumber kebenaran tunggal untuk preprocessing. Algoritma kelima
teknik diambil persis dari gambar_ppt/generate_preprocessing_result.py, yaitu
kode yang memproduksi Gambar 2.1 di proposal, supaya citra di proposal, slide,
dan citra yang benar-benar dilatih berasal dari implementasi yang sama.

Varian (Bagian 3.4.2 proposal, ditambah baseline):

  baseline          tanpa enhancement, hanya pipeline dasar
  clahe             clip limit 2.0, tile 8x8, pada kanal L (LAB)
  ben_graham        alpha=4, gamma=128, sigma = diameter retina / 30
  adaptive_sigmoid  beta = rata-rata intensitas retina, alpha = 10
  lab_ace           CLAHE clip 3.0 pada L + normalisasi lokal, rekonstruksi RGB
  mcie              merge(green, CLAHE(L), Ben Graham) -> 3 kanal

Urutan pipeline mengikuti Bagian 3.4.1 dan 3.4.2: crop area retina -> teknik
preprocessing -> resize 224x224. Standardisasi ImageNet sengaja TIDAK dilakukan
di sini melainkan saat training, supaya cache bisa disimpan sebagai uint8 yang
ringan alih-alih float32.

Seluruh fungsi bekerja pada citra BGR uint8 mengikuti konvensi OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = 224

# Resolusi antara sebelum teknik preprocessing dijalankan. Teknik berbasis
# Gaussian blur (Ben Graham, adaptive sigmoid, LAB-ACE, MCIE) berbiaya sebanding
# dengan jumlah piksel, sehingga menjalankannya pada citra hasil crop yang utuh
# (sekitar 3500x3500) memakan puluhan detik per citra. Menurunkannya ke 1024
# lebih dulu memangkas biaya lebih dari sepuluh kali lipat, masih jauh di atas
# 224 sehingga lesi kecil belum hilang saat teknik dikenakan, dan merupakan
# resolusi yang sama dengan yang dipakai untuk memproduksi Gambar 2.1 proposal.
WORK_SIZE = 1024

# Dipakai saat training, bukan saat caching (lihat catatan di docstring modul).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Pipeline dasar                                                              #
# --------------------------------------------------------------------------- #
def crop_retina(img: np.ndarray, tol: int = 12) -> np.ndarray:
    """Buang bingkai hitam, lalu pad ke bujur sangkar.

    Bagian 3.4.1 menyebut crop ke kotak pembatas minimum lalu resize ke 224x224.
    Kotak pembatas itu dipad ke bujur sangkar lebih dulu karena sebagian dataset
    (EyePACS, Messidor-2) memuat citra yang terpotong di sisi atas dan bawah,
    sehingga resize langsung akan melonjongkan lingkaran retina dan mengubah
    bentuk lesi secara tidak seragam antar dataset.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > tol)
    if len(ys) == 0:  # citra gelap total, kembalikan apa adanya
        return img
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img
    return canvas


def retina_mask(img: np.ndarray, tol: int = 12) -> np.ndarray:
    """Mask area retina; tepinya dikikis agar artefak ring tidak ikut terbawa."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (gray > tol).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.erode(m, np.ones((9, 9), np.uint8))


def retina_diameter(mask: np.ndarray) -> float:
    """Diameter retina efektif, diturunkan dari luas mask."""
    return 2.0 * np.sqrt(mask.sum() / np.pi)


# --------------------------------------------------------------------------- #
# Teknik preprocessing                                                        #
# --------------------------------------------------------------------------- #
def baseline(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Arm pembanding: tanpa enhancement apa pun."""
    return img


def clahe_lab(img: np.ndarray, mask: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def ben_graham(img: np.ndarray, mask: np.ndarray, alpha: float = 4.0, gamma: float = 128.0) -> np.ndarray:
    sigma = retina_diameter(mask) / 30.0
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, alpha, blur, -alpha, gamma)


def adaptive_sigmoid(img: np.ndarray, mask: np.ndarray, k: float = 2.0,
                     std_floor: float = 1.0) -> np.ndarray:
    """beta = mean lokal, alpha = k / std lokal, dihitung per kanal.

    Sesuai Bagian 3.4.2 proposal, kedua parameter diturunkan dari mean dan
    simpangan baku intensitas patch lokal. Versi terdahulu memakai satu mean
    global lintas kanal; karena kanal merah fundus jauh lebih terang daripada
    biru dan hijau (mean 0,70 berbanding 0,12), seluruh kanal merah terdorong ke
    ujung saturasi dan kontras lesi justru hilang.

    Simpangan baku lokal dilantai pada simpangan baku global kanalnya, sebab
    pembagian dengan std lokal yang mendekati nol akan memperkuat derau di area
    retina yang datar.
    """
    f = img.astype(np.float32) / 255.0
    sigma = retina_diameter(mask) / 30.0

    mu = cv2.GaussianBlur(f, (0, 0), sigma)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur(f * f, (0, 0), sigma) - mu * mu, 1e-6))
    for i in range(3):
        channel = f[:, :, i][mask > 0]
        if channel.size:
            sd[:, :, i] = np.maximum(sd[:, :, i], std_floor * channel.std())

    return (1.0 / (1.0 + np.exp(-k * (f - mu) / sd)) * 255).astype(np.uint8)


def lab_ace(img: np.ndarray, mask: np.ndarray, clip: float = 3.0, tile: int = 8) -> np.ndarray:
    """CLAHE pada kanal L + normalisasi lokal; kanal A/B dibiarkan apa adanya."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)

    bg = cv2.GaussianBlur(cl, (0, 0), retina_diameter(mask) / 30.0)
    ln = cv2.normalize(cl.astype(np.float32) - bg.astype(np.float32) + 128,
                       None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([ln, a, b]), cv2.COLOR_LAB2BGR)


def mcie(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Kanal hijau + CLAHE(L) + Ben Graham digabung jadi masukan 3 kanal."""
    green = img[:, :, 1]
    cl = cv2.cvtColor(clahe_lab(img, mask), cv2.COLOR_BGR2LAB)[:, :, 0]
    bg = cv2.cvtColor(ben_graham(img, mask), cv2.COLOR_BGR2GRAY)
    return cv2.merge([green, cl, bg])


TECHNIQUES = {
    "baseline": baseline,
    "clahe": clahe_lab,
    "ben_graham": ben_graham,
    "adaptive_sigmoid": adaptive_sigmoid,
    "lab_ace": lab_ace,
    "mcie": mcie,
}


# --------------------------------------------------------------------------- #
def preprocess(img: np.ndarray, technique: str, size: int = IMAGE_SIZE,
               work_size: int = WORK_SIZE) -> np.ndarray:
    """Jalankan pipeline penuh pada satu citra BGR mentah.

    crop retina -> resize ke work_size -> teknik preprocessing -> mask ->
    resize size x size. Mengembalikan BGR uint8 yang siap disimpan ke cache.
    """
    if technique not in TECHNIQUES:
        raise KeyError(f"teknik tidak dikenal: {technique!r}; pilihan: {list(TECHNIQUES)}")

    cropped = crop_retina(img)
    if work_size and max(cropped.shape[:2]) > work_size:
        cropped = cv2.resize(cropped, (work_size, work_size), interpolation=cv2.INTER_AREA)
    mask = retina_mask(cropped)
    out = TECHNIQUES[technique](cropped, mask)
    out = out * mask[:, :, None]
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)

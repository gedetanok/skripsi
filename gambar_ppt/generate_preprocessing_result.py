"""Menghasilkan citra hasil kelima teknik preprocessing untuk slide PPT.

Algoritmanya diambil persis dari proposal_build/gambar/prep_demo_src.py, yaitu
skrip yang memproduksi Gambar 2.1 di proposal, supaya slide dan proposal memakai
implementasi yang sama:

  CLAHE            : clip limit 2.0, tile 8x8, pada kanal L (LAB)
  Ben Graham       : alpha=4, beta=-4, gamma=128, sigma = diameter/30
  Adaptive Sigmoid : beta = mean intensitas, alpha = 10, pada ketiga kanal
  LAB-ACE          : CLAHE clip 3.0 pada L + normalisasi lokal, rekonstruksi RGB
  MCIE             : merge(green, CLAHE(L), Ben Graham) -> 3 kanal

Beda dari skrip proposal hanya pada penyajian: citra sumber beresolusi lebih
tinggi, dipadkan ke bujur sangkar tanpa mendistorsi lingkaran retina, dimasking
ke area retina, dan disimpan satu berkas per teknik.

Keluaran: gambar_ppt/preprocessing_result/*.png
"""

import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "1. Original Images" / "a. Training Set" / "IDRiD_03.jpg"
OUT = ROOT / "gambar_ppt" / "preprocessing_result"
SIZE = 1024


# --------------------------------------------------------------------------- #
# Persiapan: crop lingkaran retina                                            #
# --------------------------------------------------------------------------- #
def crop_retina(img, tol=12):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    img = img[y0:y1, x0:x1]

    # padding ke bujur sangkar agar lingkaran retina tidak lonjong
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img
    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def retina_mask(img, tol=12):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (gray > tol).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    # kikis tepi sedikit supaya artefak ring tidak ikut terbawa
    m = cv2.erode(m, np.ones((9, 9), np.uint8))
    return m


def apply_mask(img, mask):
    return img * mask[:, :, None]


# --------------------------------------------------------------------------- #
# Teknik preprocessing                                                        #
# --------------------------------------------------------------------------- #
def clahe_lab(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def retina_diameter(mask):
    """Skrip proposal memakai sigma = lebar citra / 30, dan lebar hasil crop-nya
    kira-kira sama dengan diameter retina. Di sini citra dipad ke bujur sangkar,
    jadi diameternya dihitung dari mask agar nilainya setara."""
    return 2.0 * np.sqrt(mask.sum() / np.pi)


def ben_graham(img, mask, alpha=4.0, gamma=128.0):
    sigma = retina_diameter(mask) / 30.0
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, alpha, blur, -alpha, gamma)


def adaptive_sigmoid(img, mask, alpha=10.0):
    """beta = rata-rata intensitas, alpha tetap, dikenakan pada ketiga kanal."""
    f = img.astype(np.float32) / 255.0
    beta = float(f[mask > 0].mean())
    out = 1.0 / (1.0 + np.exp(-alpha * (f - beta)))
    return (out * 255).astype(np.uint8)


def lab_ace(img, mask, clip=3.0, tile=8):
    """CLAHE pada kanal L + normalisasi lokal, kanal A/B dibiarkan apa adanya."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)

    bg = cv2.GaussianBlur(cl, (0, 0), retina_diameter(mask) / 30.0)
    ln = cv2.normalize(cl.astype(np.float32) - bg.astype(np.float32) + 128,
                       None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([ln, a, b]), cv2.COLOR_LAB2BGR)


def mcie(img, mask):
    """Green channel + CLAHE(L) + Ben Graham digabung jadi satu masukan 3 kanal."""
    green = img[:, :, 1]
    cl = cv2.cvtColor(clahe_lab(img), cv2.COLOR_BGR2LAB)[:, :, 0]
    bg = cv2.cvtColor(ben_graham(img, mask), cv2.COLOR_BGR2GRAY)
    return cv2.merge([green, cl, bg])


# --------------------------------------------------------------------------- #
def main():
    raw = cv2.imread(str(SRC))
    if raw is None:
        raise SystemExit(f"Citra sumber tidak ditemukan: {SRC}")

    img = crop_retina(raw)
    mask = retina_mask(img)

    results = {
        "00_original": img,
        "01_clahe": clahe_lab(img),
        "02_ben_graham": ben_graham(img, mask),
        "03_adaptive_sigmoid": adaptive_sigmoid(img, mask),
        "04_lab_ace": lab_ace(img, mask),
        "05_mcie": mcie(img, mask),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        cv2.imwrite(str(OUT / f"{name}.png"), apply_mask(res, mask))

    # montase 2x3 untuk pembanding cepat
    thumbs = [cv2.resize(apply_mask(r, mask), (512, 512), interpolation=cv2.INTER_AREA)
              for r in results.values()]
    grid = np.vstack([np.hstack(thumbs[:3]), np.hstack(thumbs[3:])])
    cv2.imwrite(str(OUT / "99_montage.png"), grid)

    print(f"Sumber   : {SRC.name}")
    print(f"Keluaran : {OUT}")
    for name in results:
        print(f"  - {name}.png")


if __name__ == "__main__":
    main()

"""Bangun cache citra 224x224 untuk satu dataset, satu berkas per varian.

Preprocessing dijalankan sekali di muka dan hasilnya disimpan, bukan dihitung
ulang tiap epoch. Alasannya, teknik berbasis Gaussian blur berbiaya sekitar satu
detik per citra; pada 72 run dengan masing-masing sampai 50 epoch, menghitungnya
secara on-the-fly akan membuat GPU menunggu CPU dan menghabiskan jatah GPU
mingguan Kaggle untuk pekerjaan yang hasilnya selalu sama.

Cache juga memangkas ukuran data secara drastis: EyePACS menyusut dari puluhan
GB menjadi ratusan MB, sehingga muat diunggah sebagai Kaggle Dataset dan
di-attach ke notebook training.

Struktur keluaran:

    <out>/<dataset>/<teknik>/<image_id>.jpg

Pemakaian:

    # 1. periksa dulu apakah rel_path di manifest cocok dengan struktur folder
    python -m src.data.build_cache --dataset aptos2019 --root /kaggle/input/... --verify

    # 2. bangun cache-nya
    python -m src.data.build_cache --dataset aptos2019 --root /kaggle/input/... --out cache
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.data.preprocessing import IMAGE_SIZE, TECHNIQUES, WORK_SIZE, preprocess

MANIFEST_DIR = Path("manifests")

# Cache disimpan sebagai JPEG mutu 95, bukan PNG. Pada 224x224 selisih visualnya
# tidak kasat mata, sementara ukuran totalnya turun sekitar tujuh kali lipat
# (sekitar 3 GB berbanding 22 GB untuk seluruh grid), yang menentukan apakah
# cache masih praktis diunggah sebagai Kaggle Dataset.
JPEG_QUALITY = 95


def build_name_index(root: Path) -> dict[str, Path]:
    """Petakan nama berkas ke lokasinya, dengan menelusuri root sekali.

    Mirror pihak ketiga (umumnya di Kaggle) menyusun folder berbeda dari rilis
    resmi, padahal nama berkas citranya sama karena berasal dari sumber yang
    sama. Mencari berdasarkan nama berkas membuat manifest yang sama berlaku
    untuk susunan folder mana pun, sehingga rel_path tidak perlu ditulis ulang
    untuk setiap mirror.
    """
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)  # cocokkan juga tanpa ekstensi
    return index


def resolve(rel_path: str, root: Path, index: dict[str, Path] | None) -> Path | None:
    """Cari citra lewat rel_path; bila gagal dan index tersedia, lewat nama berkas."""
    direct = root / rel_path
    if direct.exists():
        return direct
    if index is None:
        return None
    name = Path(rel_path).name
    return index.get(name) or index.get(Path(name).stem)


def verify(manifest: pd.DataFrame, root: Path, limit: int = 200,
           index: dict[str, Path] | None = None) -> bool:
    """Periksa apakah rel_path benar-benar menunjuk berkas yang ada.

    Dijalankan sebelum pekerjaan berjam-jam dimulai, karena rel_path di manifest
    disusun dari konvensi penamaan tiap rilis dan belum tentu cocok dengan
    struktur folder hasil unduhan yang sebenarnya.
    """
    sample = manifest.sample(min(limit, len(manifest)), random_state=0)
    missing = [p for p in sample.rel_path if resolve(p, root, index) is None]

    print(f"diperiksa {len(sample)} berkas contoh di {root}"
          + (" (pencocokan nama berkas aktif)" if index is not None else ""))
    if not missing:
        print("  semua ditemukan")
        return True

    print(f"  TIDAK ditemukan: {len(missing)} dari {len(sample)}")
    for p in missing[:5]:
        print(f"    - {p}")
    print("\n  isi root yang sebenarnya:")
    for p in sorted(root.iterdir())[:10]:
        print(f"    {p.name}/" if p.is_dir() else f"    {p.name}")
    return False


def _process(args: tuple[str, str, str, str, int, int]) -> str | None:
    src, image_id, _root, out_dir, size, work_size = args
    img = cv2.imread(src)
    if img is None:
        return image_id  # dilaporkan sebagai gagal

    for technique in TECHNIQUES:
        dest = Path(out_dir) / technique / f"{image_id}.jpg"
        if dest.exists():
            continue
        result = preprocess(img, technique, size=size, work_size=work_size)
        cv2.imwrite(str(dest), result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[p.stem for p in MANIFEST_DIR.glob("*.csv")])
    parser.add_argument("--root", required=True, type=Path, help="root folder citra mentah dataset")
    parser.add_argument("--out", type=Path, default=Path("cache"))
    parser.add_argument("--verify", action="store_true", help="hanya periksa rel_path, tanpa membangun cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--by-name", action="store_true",
                        help="cocokkan citra berdasarkan nama berkas, bukan rel_path; "
                             "dipakai untuk mirror yang susunan foldernya berbeda")
    parser.add_argument("--work-size", type=int, default=WORK_SIZE)
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_DIR / f"{args.dataset}.csv")

    index = None
    if args.by_name:
        print(f"menelusuri {args.root} untuk mengindeks nama berkas...", flush=True)
        index = build_name_index(args.root)
        print(f"  {len(index):,} entri terindeks")

    if args.verify:
        raise SystemExit(0 if verify(manifest, args.root, index=index) else 1)

    if not verify(manifest, args.root, index=index):
        raise SystemExit("citra tidak ditemukan; coba --by-name bila memakai mirror "
                         "dengan susunan folder berbeda.")

    out_dir = args.out / args.dataset
    for technique in TECHNIQUES:
        (out_dir / technique).mkdir(parents=True, exist_ok=True)

    tasks, failed = [], []
    for r in manifest.itertuples():
        src = resolve(r.rel_path, args.root, index)
        if src is None:
            failed.append(r.image_id)
            continue
        tasks.append((str(src), r.image_id, str(args.root), str(out_dir),
                      args.size, args.work_size))

    if failed:
        print(f"peringatan: {len(failed)} citra tidak ditemukan dan dilewati")

    with ProcessPoolExecutor(args.workers) as pool:
        for result in tqdm(pool.map(_process, tasks, chunksize=16), total=len(tasks), desc=args.dataset):
            if result:
                failed.append(result)

    print(f"\nselesai: {len(tasks) - len(failed)}/{len(tasks)} citra x {len(TECHNIQUES)} varian -> {out_dir}")
    if failed:
        log = out_dir / "gagal.txt"
        log.write_text("\n".join(failed))
        print(f"GAGAL dibaca: {len(failed)} citra, daftarnya di {log}")


if __name__ == "__main__":
    main()

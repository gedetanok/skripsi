"""Bangun manifest terpadu untuk keenam dataset DR grading.

Keenam sumber label punya skema yang berbeda-beda. Skrip ini menormalkannya
menjadi satu skema tunggal yang dipakai oleh seluruh pipeline eksperimen
(cache preprocessing maupun training):

    dataset, image_id, rel_path, grade, split

  dataset   nama dataset (idrid, ddr, aptos2019, messidor2, eyepacs, deepdrid)
  image_id  identitas citra apa adanya dari file label sumber
  rel_path  path relatif terhadap root dataset, sehingga manifest tetap valid
            baik di lokal maupun saat di-mount di Kaggle Notebooks
  grade     derajat ICDR 0-4 (citra ungradable sudah dibuang)
  split     train / val / test

Partisi mengikuti Bagian 3.2.1 proposal: partisi resmi dipakai apa adanya bila
tersedia, sisanya dibagi stratified dengan seed tetap.

Jalankan:  python -m src.data.build_manifests
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABEL_DIR = Path("dataset_eda/labels")
OUT_DIR = Path("manifests")

# EyePACS terlalu besar untuk dilatih utuh (88.702 citra). Sesuai Bagian 3.2.1
# diambil subset class-stratified yang mempertahankan distribusi kelas aslinya.
EYEPACS_SUBSET = {"train": 10_000, "val": 2_000, "test": 5_000}


def stratified_split(df: pd.DataFrame, fracs: tuple[float, float, float]) -> pd.DataFrame:
    """Bagi df menjadi train/val/test secara stratified pada kolom grade."""
    f_train, f_val, _ = fracs
    train, rest = train_test_split(
        df, train_size=f_train, stratify=df.grade, random_state=SEED
    )
    val, test = train_test_split(
        rest, train_size=f_val / (1 - f_train), stratify=rest.grade, random_state=SEED
    )
    return pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    )


def stratified_subsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Ambil n baris sambil mempertahankan proporsi kelas."""
    if n >= len(df):
        return df
    keep, _ = train_test_split(df, train_size=n, stratify=df.grade, random_state=SEED)
    return keep


# --------------------------------------------------------------------------- #
# Satu fungsi per dataset. Masing-masing mengembalikan frame berkolom
# image_id, rel_path, grade, split.
# --------------------------------------------------------------------------- #

def build_idrid() -> pd.DataFrame:
    """Partisi resmi 413/103; 15% dari train disisihkan stratified sebagai val.

    Penamaan IDRiD dimulai ulang dari IDRiD_001 di tiap partisi resmi, sehingga
    103 nama bentrok antara training dan testing. Nama partisi asal karena itu
    dijadikan awalan agar image_id unik dan pasangan prediksi per citra pada uji
    McNemar tidak tertukar.
    """
    src = LABEL_DIR / "idrid"
    parts = []
    for fname, split, subdir in [
        ("a. IDRiD_Disease Grading_Training Labels.csv", "train", "a. Training Set"),
        ("b. IDRiD_Disease Grading_Testing Labels.csv", "test", "b. Testing Set"),
    ]:
        d = pd.read_csv(src / fname).rename(
            columns={"Image name": "image_id", "Retinopathy grade": "grade"}
        )
        d = d[["image_id", "grade"]].dropna()
        d["rel_path"] = subdir + "/" + d.image_id + ".jpg"
        d["image_id"] = f"{split}set_" + d.image_id
        parts.append(d.assign(split=split))

    df = pd.concat(parts)
    train = df[df.split == "train"]
    train, val = train_test_split(
        train, test_size=0.15, stratify=train.grade, random_state=SEED
    )
    return pd.concat([train, val.assign(split="val"), df[df.split == "test"]])


def build_ddr() -> pd.DataFrame:
    """Partisi resmi; grade 5 berarti ungradable dan dibuang."""
    src = LABEL_DIR / "ddr"
    parts = []
    for fname, split in [("train.txt", "train"), ("valid.txt", "val"), ("test.txt", "test")]:
        d = pd.read_csv(src / fname, sep=" ", header=None, names=["image_id", "grade"])
        d["rel_path"] = f"DR_grading/{fname.replace('.txt', '')}/" + d.image_id
        parts.append(d.assign(split=split))
    df = pd.concat(parts)
    return df[df.grade != 5]


def build_deepdrid() -> pd.DataFrame:
    """Partisi resmi; hanya field pertama tiap mata yang dipakai (994 citra).

    Label tersimpan per mata (left_eye_DR_Level / right_eye_DR_Level) sehingga
    derajat tiap citra diambil dari kolom yang sesuai sisi matanya. Skema file
    test berbeda: satu kolom DR_Levels yang sudah per citra.
    """
    src = LABEL_DIR / "deepdrid"
    parts = []
    for fname, split in [
        ("regular-fundus-training.csv", "train"),
        ("regular-fundus-validation.csv", "val"),
        ("regular-fundus-test.csv", "test"),
    ]:
        d = pd.read_csv(src / fname)
        d["eye"] = d.image_id.str.extract(r"_([lr])\d+$")[0]
        d["field"] = d.image_id.str.extract(r"_[lr](\d+)$")[0].astype(int)

        if "DR_Levels" in d.columns:
            d["grade"] = d.DR_Levels
        else:
            d["grade"] = d.apply(
                lambda r: r.left_eye_DR_Level if r.eye == "l" else r.right_eye_DR_Level,
                axis=1,
            )

        d = d[(d.field == 1) & d.grade.notna()].copy()
        d["patient"] = d.image_id.str.split("_").str[0]
        stem = fname.replace(".csv", "")
        d["rel_path"] = stem + "/" + d.patient + "/" + d.image_id + ".jpg"
        parts.append(d[["image_id", "rel_path", "grade"]].assign(split=split))

    return pd.concat(parts)


def build_aptos() -> pd.DataFrame:
    """Tidak ada partisi resmi (label test ditahan panitia): stratified 70/15/15."""
    d = pd.read_csv(LABEL_DIR / "aptos2019/train.csv").rename(
        columns={"id_code": "image_id", "diagnosis": "grade"}
    )
    d["rel_path"] = "train_images/" + d.image_id + ".png"
    return stratified_split(d, (0.70, 0.15, 0.15))


def build_messidor2() -> pd.DataFrame:
    """Tidak ada partisi resmi: stratified 70/15/15. Citra ungradable dibuang."""
    d = pd.read_csv(LABEL_DIR / "messidor2/messidor_data.csv")
    d = d[(d.adjudicated_gradable == 1) & d.adjudicated_dr_grade.notna()].copy()
    d["grade"] = d.adjudicated_dr_grade.astype(int)
    d["rel_path"] = "IMAGES/" + d.image_id
    return stratified_split(d[["image_id", "rel_path", "grade"]], (0.70, 0.15, 0.15))


def build_eyepacs() -> pd.DataFrame:
    """Partisi resmi train/test; val disisihkan dari train, lalu ketiganya disubsampel."""
    src = LABEL_DIR / "eyepacs"
    train = pd.read_csv(src / "trainLabels.csv").rename(
        columns={"image": "image_id", "level": "grade"}
    )
    test = pd.read_csv(src / "testLabels15.csv").rename(
        columns={"image": "image_id", "level": "grade"}
    )[["image_id", "grade"]]

    train, val = train_test_split(
        train, test_size=0.15, stratify=train.grade, random_state=SEED
    )

    parts = []
    for d, split, folder in [
        (train, "train", "train"),
        (val, "val", "train"),
        (test, "test", "test"),
    ]:
        d = stratified_subsample(d, EYEPACS_SUBSET[split]).copy()
        d["rel_path"] = f"{folder}/" + d.image_id + ".jpeg"
        parts.append(d.assign(split=split))

    return pd.concat(parts)


BUILDERS = {
    "idrid": build_idrid,
    "ddr": build_ddr,
    "deepdrid": build_deepdrid,
    "aptos2019": build_aptos,
    "messidor2": build_messidor2,
    "eyepacs": build_eyepacs,
}

COLUMNS = ["dataset", "image_id", "rel_path", "grade", "split"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=list(BUILDERS), help="bangun sebagian saja")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    targets = args.only or list(BUILDERS)

    for name in targets:
        df = BUILDERS[name]().assign(dataset=name)
        df["grade"] = df.grade.astype(int)
        df = df[COLUMNS].sort_values(["split", "image_id"]).reset_index(drop=True)

        dupes = df.image_id.duplicated().sum()
        if dupes:
            raise ValueError(
                f"{name}: {dupes} image_id ganda. image_id harus unik agar pasangan "
                "prediksi per citra pada uji McNemar tidak tertukar."
            )
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

        counts = df.grade.value_counts().sort_index()
        per_split = df.split.value_counts()
        print(
            f"{name:<10} n={len(df):>6}  "
            f"train/val/test={per_split.get('train', 0)}/{per_split.get('val', 0)}/{per_split.get('test', 0)}"
            f"  kelas={[counts.get(k, 0) for k in range(5)]}"
        )


if __name__ == "__main__":
    main()

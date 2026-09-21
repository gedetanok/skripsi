# Cara mengunduh FILE LABEL keenam dataset (untuk EDA distribusi kelas)

Untuk menghitung distribusi kelas per dataset, **kita hanya butuh file label (CSV/txt), bukan seluruh gambarnya.**
Ini menghemat puluhan GB (EyePACS saja ~88k gambar). Letakkan setiap file label ke folder yang ditunjuk di
bawah, lalu jalankan `eda_class_distribution.ipynb`.

Skala kelas: **ICDR 5 kelas** — 0 = No DR, 1 = Mild NPDR, 2 = Moderate NPDR, 3 = Severe NPDR, 4 = PDR.

---

## 1. IDRiD  →  `labels/idrid/`
- Sumber: IEEE DataPort, "Indian Diabetic Retinopathy Image Dataset (IDRiD)".
  <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid>
- Unduh bagian **"B. Disease Grading"**. Di dalamnya ada folder `2. Groundtruths/` berisi:
  - `a. IDRiD_Disease Grading_Training Labels.csv`
  - `b. IDRiD_Disease Grading_Testing Labels.csv`
- Salin kedua CSV itu ke `labels/idrid/`.
- Kolom: `Image name`, `Retinopathy grade` (0-4), `Risk of macular edema`.

## 2. DDR  →  `labels/ddr/`
- Sumber: GitHub `nkicsl/DDR-dataset`. <https://github.com/nkicsl/DDR-dataset>
- Ambil label **DR grading** (folder `DR_grading/`): `train.txt`, `valid.txt`, `test.txt`.
- Salin ketiga file ke `labels/ddr/`.
- Format tiap baris: `<namafile>.jpg <grade>` dengan grade 0-5, di mana **5 = ungradable** (akan dibuang).

## 3. APTOS 2019  →  `labels/aptos2019/`
- Sumber: Kaggle competition `aptos2019-blindness-detection`.
  <https://www.kaggle.com/c/aptos2019-blindness-detection/data>
- Hanya butuh `train.csv` (label uji ditahan panitia).
  - CLI: `kaggle competitions download -c aptos2019-blindness-detection -f train.csv`
- Salin `train.csv` ke `labels/aptos2019/`.
- Kolom: `id_code`, `diagnosis` (0-4).

## 4. Messidor-2  →  `labels/messidor2/`
- Sumber grade hasil adjudikasi (Krause et al., 2018): cari CSV **`messidor_data.csv`**
  (tersedia di beberapa mirror Kaggle, mis. dataset "Messidor-2 DR Grades").
- Salin ke `labels/messidor2/`.
- Kolom yang dipakai: `adjudicated_dr_grade` (0-4). Baris NaN (ungradable) dibuang.
  Kalau nama kolom berbeda, notebook akan menampilkan daftar kolom yang tersedia agar mudah disesuaikan.

## 5. EyePACS (Kaggle DR Detection 2015)  →  `labels/eyepacs/`
- Sumber: Kaggle competition `diabetic-retinopathy-detection`.
  <https://www.kaggle.com/c/diabetic-retinopathy-detection/data>
- Butuh `trainLabels.csv` (dan opsional `retinopathy_solution.csv` untuk label test).
  - CLI: `kaggle competitions download -c diabetic-retinopathy-detection -f trainLabels.csv.zip`
- Ekstrak `trainLabels.csv` ke `labels/eyepacs/`.
- Kolom: `image`, `level` (0-4).

## 6. DeepDRiD  →  `labels/deepdrid/`
- Sumber: GitHub `deepdrdoc/DeepDRiD`. <https://github.com/deepdrdoc/DeepDRiD>
- Ambil label **regular fundus** (mis. `regular-fundus-training.csv`, `regular-fundus-validation.csv`).
- Salin ke `labels/deepdrid/`.
- Kolom grade bisa bernama `patient_DR_Level` / `DR_level`. Notebook akan menampilkan kolom tersedia bila
  nama default tidak ditemukan, agar bisa disesuaikan di blok CONFIG.

---

### Catatan
- Untuk Kaggle CLI: butuh `pip install kaggle` + token `~/.kaggle/kaggle.json`, dan sekali klik **"Join
  competition / Accept rules"** di halaman kompetisi.
- Kalau satu dataset belum selesai diunduh, tidak apa-apa: notebook akan melewatinya dan tetap memproses
  yang sudah ada. Jalankan ulang setiap kali menambah file.

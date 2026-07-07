# Literatur Skripsi — Klasifikasi Diabetic Retinopathy dengan Uncertainty Quantification

**Topik:** Analisis Komparatif Metode Estimasi Ketidakpastian untuk Klasifikasi Tingkat Keparahan Diabetic Retinopathy pada Citra Fundus (IDRiD)

**Tanggal kompilasi:** 2026-04-20

---

## 00 — Legacy (Topik Sebelumnya, Tidak Terpakai)

Paper dari topik segmentasi SE/HE/OD sebelumnya. Disimpan sebagai arsip saja — **tidak dipakai untuk topik baru**.

---

## 01 — DR Grading Technical (Fondasi Teknis)

Paper tentang metode deep learning untuk DR Grading.

| File | Paper | Tahun | Venue | Kegunaan |
|---|---|---|---|---|
| `DRGNet_2022_JointSegGrading.pdf` | DRG-Net: Interactive Joint Learning | 2022 | arXiv | Contoh arsitektur joint seg+grading |
| `CrossFeatureFusion_2024_LesionMap.pdf` | Cross Feature Fusion Fundus+Lesion Map | 2024 | arXiv | Dual-stream design |
| `DivergentDomains_2024_DR_Generalization.pdf` | Divergent Domains, Convergent Grading | 2024 | arXiv | Cross-dataset generalization |
| `StageAware_2025_OrdinalRegression_DR.pdf` | Stage-Aware Ordinal Regression DR | 2025 | arXiv | Ordinal regression untuk DR |
| `FromRetinalPixelsToPatients_2025_DR_Evolution.pdf` | Evolution of DL Research in DR Screening | 2025 | arXiv | **Survey komprehensif terbaru** |
| `Zhou_2023_RETFound_Nature.pdf` | A foundation model for retinal images (RETFound) | 2023 | Nature | Foundation model baseline |

---

## 02 — Uncertainty Quantification Methods (Core Methods)

Paper fondasi metode uncertainty quantification. Ini yang akan kamu adopsi dan bandingkan.

| File | Paper | Tahun | Venue | Peran dalam Skripsi |
|---|---|---|---|---|
| `Gal_2016_MCDropout_Bayesian_Approximation.pdf` | Dropout as Bayesian Approximation | 2016 | ICML | **Metode 1: MC Dropout** |
| `Lakshminarayanan_2017_DeepEnsembles.pdf` | Simple and Scalable Uncertainty via Deep Ensembles | 2017 | NeurIPS | **Metode 2: Deep Ensemble** |
| `Sensoy_2018_EvidentialDeepLearning.pdf` | Evidential DL to Quantify Classification Uncertainty | 2018 | NeurIPS | **Metode 3: Evidential DL (Dirichlet)** |
| `Guo_2017_TemperatureScaling_Calibration.pdf` | On Calibration of Modern Neural Networks | 2017 | ICML | **Kalibrasi post-hoc + ECE metric** |
| `Geifman_2019_SelectiveNet.pdf` | SelectiveNet: Integrated Reject Option | 2019 | ICML | **Selective classification framework** |
| `Cao_2019_CORAL_OrdinalRegression.pdf` | CORAL: Rank-consistent Ordinal Regression | 2019 | arXiv | Baseline ordinal regression |
| `Shi_2021_CORN_OrdinalRegression.pdf` | CORN: Rank-Consistent Ordinal Regression | 2021 | Pattern Analysis | Ordinal regression (improved) |

---

## 03 — Uncertainty Quantification Specifically for DR

Paper yang sudah apply UQ ke DR — ini referensi paling penting untuk BAB II dan menunjukkan gap.

| File | Paper | Tahun | Venue | Catatan |
|---|---|---|---|---|
| `UncertaintyAware_Ordinal_CrossDataset_DR_2026.pdf` | Uncertainty-Aware Ordinal DL for cross-Dataset DR | 2026 | arXiv | **Paper paling dekat dgn skripsi** — evidential+ordinal+cross-dataset. Gap: tanpa lesion awareness. Tapi kita fokus UQ murni, jadi bisa jadi inspirasi, bukan replikasi. |
| `Nasir_2024_Bayesian_DR_Uncertainty.pdf` | Uncertainty-aware DR Detection with Bayesian | 2024/2025 | Nature Sci. Reports | Bayesian approaches untuk DR — referensi utama BAB II |

---

## 04 — Medical & Clinical DR

Paper fondasi klinis dan validasi AI untuk DR.

| File | Paper | Tahun | Venue | Peran |
|---|---|---|---|---|
| `Gulshan_2016_JAMA_Google_DRDetection.pdf` | Google Deep Learning DR Detection | 2016 | JAMA | **Paper klinis AI DR pertama** — must-cite |
| `Abramoff_2018_Pivotal_Trial_IDxDR.pdf` | Pivotal Trial IDx-DR (LumineticsCore) | 2018 | npj Digital Medicine | First FDA-approved autonomous AI |
| `IDxDR_2018_FDA_Review.pdf` | FDA De Novo Review IDx-DR | 2018 | FDA | Bukti regulatory — mendukung urgensi UQ untuk klinis |
| `Teo_2021_GlobalDR_Prevalence_2045.pdf` | Global DR Prevalence & Projection 2045 | 2021 | Ophthalmology | **Burden global** — latar belakang BAB I |
| `Wong_2023_DR_Pandemic_Friedenwald.pdf` | DR "Pandemic" — Friedenwald Lecture | 2023 | IOVS | State-of-urgency DR |
| `ADA_2024_Standards_Retinopathy.pdf` | ADA Standards of Medical Care: Retinopathy | 2024 | Diabetes Care | **Standar klinis** untuk screening |

---

## 05 — ICDR Guidelines (Severity Scale)

Standar klasifikasi yang jadi label target.

| File | Paper | Tahun | Venue | Peran |
|---|---|---|---|---|
| `Wilkinson_2003_ICDR_Severity_Scale.pdf` | Proposed International Clinical DR Severity Scales | 2003 | Ophthalmology | **Fondasi 5-class labeling** — must-cite di BAB II |
| `DRClassification_PastPresentFuture_Frontiers_2022.pdf` | Classification of DR: Past, Present and Future | 2022 | Frontiers | Evolusi sistem grading |

---

## 06 — Indonesia Context (Relevansi Lokal)

Data epidemiologi Indonesia untuk justifikasi urgensi skripsi.

| File | Paper | Tahun | Venue | Peran |
|---|---|---|---|---|
| `Saputra_2024_DR_Padang_Indonesia.pdf` | DR Prevalence in Padang | 2024 | Bioscientia Medicina | Data lokal Sumatera |
| `Sasongko_2025_Indonesia_DR_Cohort.pdf` | Indonesia DR Cohort Type 2 Diabetes | 2025 | Studi kohort | **Data terbaru Indonesia** |

---

## Paper yang Belum Bisa Di-download (Akses Institusi Diperlukan)

Beberapa paper diblokir dari automated download karena anti-bot publisher. Akses via perpustakaan Undiksha atau Google Scholar:

### Tier 1 (PENTING — harus didapat):
1. **Araujo T. et al. (2020). DR|GRADUATE: Uncertainty-aware DL-based DR grading in eye fundus images.** *Medical Image Analysis*, 63, 101715. [DOI: 10.1016/j.media.2020.101715](https://doi.org/10.1016/j.media.2020.101715)
   - **Ini paper paling penting untuk BAB II** — pioneering work uncertainty di DR grading
   - Coba: ResearchGate, Sci-Hub (legal di Indonesia), perpustakaan kampus

2. **Porwal P. et al. (2020). IDRiD: Diabetic Retinopathy – Segmentation and Grading Challenge.** *Medical Image Analysis*, 59, 101561. [DOI: 10.1016/j.media.2019.101561](https://doi.org/10.1016/j.media.2019.101561)
   - **Paper dataset yang harus kamu sitasi**
   - Alternatif: https://openreview.net/pdf/44911445969e2500122aebc971bd58ae95787649.pdf

3. **Nadeem M.W. et al. (2022). Deep Learning for DR Analysis: A Review.** *Sensors*, 22(18), 6780.
   - Survey komprehensif. [MDPI](https://www.mdpi.com/1424-8220/22/18/6780) — block otomasi, tapi bisa diakses manual via browser

### Tier 2 (Bagus untuk konteks Indonesia):
4. Irawati Y. et al. (2025). DR Incidence & Progression in Indonesia Cohort — PMC12396671
5. Soewondo P. et al. (2023). Indonesia DISCOVER Cohort — PMC10213161
6. Indonesian GPs DR Screening Jakarta (2023) — PMC10176940

Cara ambil: buka URL `https://pmc.ncbi.nlm.nih.gov/articles/PMCXXXXXXX/` di browser manual → klik "Download PDF".

---

## Peta Penggunaan untuk BAB

### BAB I (Latar Belakang)
- **Urgensi global:** Teo 2021, Wong 2023
- **Urgensi Indonesia:** Sasongko 2025, Saputra 2024
- **Gap deployment klinis:** Gulshan 2016 (akurasi tinggi tapi masih manual), IDxDR FDA 2018 (terbatas binary), Abramoff 2018
- **Kebutuhan uncertainty:** Nasir 2024, UncertaintyOrdinal 2026

### BAB II (Tinjauan Pustaka)
- **ICDR definition:** Wilkinson 2003, DRClassification 2022
- **DR grading methods evolution:** Gulshan 2016 → FromRetinalPixels 2025 → DivergentDomains 2024 → RETFound 2023
- **UQ methods fundamental:** Gal 2016 (MC Dropout), Lakshminarayanan 2017 (Ensembles), Sensoy 2018 (Evidential), Guo 2017 (Calibration)
- **Selective classification:** Geifman 2019
- **Ordinal regression:** Cao 2019 (CORAL), Shi 2021 (CORN), StageAware 2025
- **UQ in DR (state-of-the-art):** Araujo 2020 (DR|GRADUATE), Nasir 2024, UncertaintyOrdinal 2026

### BAB III (Metodologi)
- **Dataset:** Porwal 2020 (IDRiD)
- **Backbone:** EfficientNet dari tren benchmark (FromRetinalPixels 2025 survey)
- **UQ methods:** paper fondasi di kategori 02

### BAB IV (Hasil)
- **Comparison metrics:** Guo 2017 (ECE), Geifman 2019 (AURC), StageAware 2025 (QWK analysis)

### BAB V (Kesimpulan)
- Tempatkan kontribusi relative terhadap Araujo 2020 dan UncertaintyOrdinal 2026

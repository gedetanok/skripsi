#!/usr/bin/env bash
# Unduh FILE LABEL 5 dataset sisa dari Kaggle (IDRiD, DDR, APTOS, Messidor-2, EyePACS).
# DeepDRiD sudah diunduh terpisah dari GitHub.
#
# Prasyarat:
#   1) ~/.kaggle/kaggle.json sudah ada (chmod 600)
#   2) Sudah klik "Join Competition" untuk aptos2019 & diabetic-retinopathy-detection
#
# Jalankan:  bash download_kaggle_labels.sh
set -uo pipefail
cd "$(dirname "$0")"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
KG="python3 -m kaggle"

# Slug dataset (bisa diedit bila skrip memilih yang salah; lihat hasil 'search' di log)
IDRID_SLUG="${IDRID_SLUG:-}"
DDR_SLUG="${DDR_SLUG:-}"
MESSIDOR_SLUG="${MESSIDOR_SLUG:-}"

log() { printf '\n=== %s ===\n' "$1"; }

# --- 1. APTOS 2019 (competition) -> train.csv ---
log "APTOS 2019"
$KG competitions download -c aptos2019-blindness-detection -f train.csv -p labels/aptos2019 || \
  echo "  [!] gagal (sudah accept rules?)"
( cd labels/aptos2019 && [ -f train.csv.zip ] && unzip -o train.csv.zip && rm -f train.csv.zip; true )

# --- 2. EyePACS (competition) -> trainLabels.csv (+ test labels opsional) ---
log "EyePACS"
$KG competitions download -c diabetic-retinopathy-detection -f trainLabels.csv.zip -p labels/eyepacs || \
  echo "  [!] gagal (sudah accept rules?)"
( cd labels/eyepacs && [ -f trainLabels.csv.zip ] && unzip -o trainLabels.csv.zip && rm -f trainLabels.csv.zip; true )
$KG competitions download -c diabetic-retinopathy-detection -f retinopathy_solution.csv.zip -p labels/eyepacs 2>/dev/null && \
  ( cd labels/eyepacs && unzip -o retinopathy_solution.csv.zip && rm -f retinopathy_solution.csv.zip ) || true

# --- helper: resolve slug via search bila belum di-set, lalu unduh 1 file label ---
grab_dataset() { # $1=query  $2=dest_dir  $3=file_regex  $4=slug_var_value
  local query="$1" dest="$2" rx="$3" slug="$4"
  if [ -z "$slug" ]; then
    echo "  cari dataset untuk '$query':"
    $KG datasets list -s "$query" --csv 2>/dev/null | head -8
    slug=$($KG datasets list -s "$query" --csv 2>/dev/null | sed -n '2p' | cut -d, -f1)
  fi
  [ -z "$slug" ] && { echo "  [!] slug tak ditemukan untuk $query"; return; }
  echo "  pakai slug: $slug"
  echo "  daftar file:"; $KG datasets files -d "$slug" --csv 2>/dev/null | head -20
  local f
  f=$($KG datasets files -d "$slug" --csv 2>/dev/null | cut -d, -f1 | grep -iE "$rx" | head -1)
  if [ -n "$f" ]; then
    echo "  unduh file: $f"
    $KG datasets download -d "$slug" -f "$f" -p "$dest"
    ( cd "$dest" && for z in *.zip; do [ -f "$z" ] && unzip -o "$z" && rm -f "$z"; done; true )
  else
    echo "  [!] tak ada file cocok /$rx/ di $slug — cek daftar di atas, set *_SLUG manual."
  fi
}

# --- 3. IDRiD -> label grading (train+test CSV) ---
log "IDRiD"
grab_dataset "idrid diabetic retinopathy grading" labels/idrid "grad.*label.*\.csv|disease.*grad.*\.csv" "$IDRID_SLUG"

# --- 4. DDR -> DR grading labels (txt/csv) ---
log "DDR"
grab_dataset "ddr diabetic retinopathy grading" labels/ddr "train\.txt|grad|dr_grading" "$DDR_SLUG"

# --- 5. Messidor-2 -> adjudicated grades csv ---
log "Messidor-2"
grab_dataset "messidor-2 adjudicated dr grade" labels/messidor2 "messidor.*\.csv|adjudicat" "$MESSIDOR_SLUG"

log "SELESAI"
echo "Cek isi labels/*/ lalu jalankan ulang eda_class_distribution.ipynb."
find labels -type f | sort

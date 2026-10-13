"""Generate slide illustrations with the exact shared experiment preprocessing."""
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.preprocessing import preprocess, WORK_SIZE
SRC = ROOT / "1. Original Images/a. Training Set/IDRiD_03.jpg"
OUT = ROOT / "gambar_ppt/preprocessing_result"

def main():
    raw = cv2.imread(str(SRC))
    if raw is None:
        raise FileNotFoundError(SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, technique in [("00_original", "baseline"), ("01_clahe", "clahe"),
            ("02_ben_graham", "ben_graham"), ("03_adaptive_sigmoid", "adaptive_sigmoid"),
            ("04_lab_ace", "lab_ace"), ("05_mcie", "mcie")]:
        img = preprocess(raw, technique, size=WORK_SIZE)
        assert cv2.imwrite(str(OUT / f"{name}.png"), img)
        results.append(cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA))
    grid = np.vstack([np.hstack(results[:3]), np.hstack(results[3:])])
    assert cv2.imwrite(str(OUT / "99_montage.png"), grid)
    print(OUT)

if __name__ == "__main__":
    main()

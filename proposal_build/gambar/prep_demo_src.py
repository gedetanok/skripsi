"""Gambar 2.1: shared preprocessing, display resolution 512; work resolution 1024."""
from pathlib import Path
import sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.preprocessing import preprocess
SRC = ROOT / "ddb1_v02_01/images/diaretdb1_image003.png"
OUT = Path(__file__).resolve().parent / "prep_demo.png"

def main():
    raw = cv2.imread(str(SRC))
    if raw is None:
        raise FileNotFoundError(SRC)
    panels = [("Original / baseline", "baseline"), ("CLAHE", "clahe"),
              ("Ben Graham", "ben_graham"), ("Adaptive Sigmoid", "adaptive_sigmoid"),
              ("LAB-ACE", "lab_ace"), ("MCIE", "mcie")]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.8))
    for ax, (title, technique) in zip(axes.ravel(), panels):
        img = preprocess(raw, technique, size=512)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=12, fontfamily="serif")
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95,
                        wspace=0.04, hspace=0.18)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(OUT)

if __name__ == "__main__":
    main()

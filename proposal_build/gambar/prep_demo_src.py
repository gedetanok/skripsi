import cv2, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/Users/mac/Documents/Kuliah/Semester 6/Skripsi/ddb1_v02_01/images/diaretdb1_image003.png"

def retina_crop(img):
    g = img[:, :, 1]
    _, m = cv2.threshold(g, 12, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return img
    return img[ys.min():ys.max(), xs.min():xs.max()]

def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def clahe_L(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def ben_graham(bgr):
    s = bgr.shape[1] / 30.0
    out = cv2.addWeighted(bgr, 4, cv2.GaussianBlur(bgr, (0, 0), s), -4, 128)
    return out

def adaptive_sigmoid(bgr):
    f = bgr.astype(np.float32) / 255.0
    beta = f.mean()
    alpha = 10.0
    out = 1.0 / (1.0 + np.exp(-alpha * (f - beta)))
    return (out * 255).astype(np.uint8)

def lab_ace(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    bg = cv2.GaussianBlur(cl, (0, 0), cl.shape[1] / 30.0)
    ln = cv2.normalize(cl.astype(np.float32) - bg.astype(np.float32) + 128,
                       None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((ln, a, b)), cv2.COLOR_LAB2BGR)

def mcie(bgr):
    green = bgr[:, :, 1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    bg = cv2.cvtColor(ben_graham(bgr), cv2.COLOR_BGR2GRAY)
    return cv2.merge((green, cl, bg))  # 3-channel composite

img = retina_crop(cv2.imread(SRC))
img = cv2.resize(img, (512, 512))

panels = [
    ("Original", to_rgb(img)),
    ("CLAHE", to_rgb(clahe_L(img))),
    ("Ben Graham", to_rgb(ben_graham(img))),
    ("Adaptive Sigmoid", to_rgb(adaptive_sigmoid(img))),
    ("LAB-ACE", to_rgb(lab_ace(img))),
    ("MCIE", to_rgb(mcie(img))),
]

fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
for ax, (title, im) in zip(axes.ravel(), panels):
    ax.imshow(im)
    ax.set_title(title, fontsize=12, fontfamily="serif")
    ax.axis("off")
plt.tight_layout(pad=0.6)
plt.savefig("/tmp/prep_demo.png", dpi=170, bbox_inches="tight")
print("saved /tmp/prep_demo.png")

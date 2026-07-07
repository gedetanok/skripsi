import cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
BASE=Path(".")
img=cv2.cvtColor(cv2.imread(str(BASE/"1. Original Images/a. Training Set/IDRiD_13.jpg")),cv2.COLOR_BGR2RGB)
se=cv2.imread(str(BASE/"2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates/IDRiD_13_SE.tif"),cv2.IMREAD_GRAYSCALE)
he=cv2.imread(str(BASE/"2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates/IDRiD_13_EX.tif"),cv2.IMREAD_GRAYSCALE)
od=cv2.imread(str(BASE/"2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc/IDRiD_13_OD.tif"),cv2.IMREAD_GRAYSCALE)
H,W=img.shape[:2]
def rs(m):
    return cv2.resize(m,(W,H),interpolation=cv2.INTER_NEAREST) if m is not None else np.zeros((H,W),np.uint8)
se,he,od=rs(se),rs(he),rs(od)
ov=img.copy()
ov[se>0]=[255,0,0]      # soft exudate -> merah
ov[he>0]=[255,255,0]    # hard exudate -> kuning
ov[od>0]=[0,128,255]    # optic disc -> biru
blend=cv2.addWeighted(img,0.55,ov,0.45,0)
fig,ax=plt.subplots(1,2,figsize=(14,7))
ax[0].imshow(img); ax[0].set_title("(a) Citra fundus asli (IDRiD_13)",fontsize=13)
ax[1].imshow(blend); ax[1].set_title("(b) Anotasi lesi: soft exudate (merah),\nhard exudate (kuning), optic disc (biru)",fontsize=13)
for a in ax: a.axis("off")
plt.tight_layout()
plt.savefig("laporan_build/gambar/representative_se.png",dpi=110,bbox_inches="tight")
print("saved", Path("laporan_build/gambar/representative_se.png").exists())

import os
from pathlib import Path
import cv2
import random
from tqdm import tqdm

ROOT = Path("data/raw/deepfashion_category")
IMG_DIR = ROOT / "Img"
ANNO_DIR = ROOT / "Anno"

BBOX_FILE = ANNO_DIR / "list_bbox.txt"
CAT_FILE  = ANNO_DIR / "list_category_img.txt"


OUT = Path("data/raw/dataset_yolo")

for split in ["train", "val"]:
    (OUT / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (OUT / f"labels/{split}").mkdir(parents=True, exist_ok=True)


TOP = set([1,2,3,4,5,6,7,8,9])
BOTTOM = set([20,21,22,23])
SHOES = set([40,41])
ACCESSORY = set([50,51,52])

def map_category(cid):
    if cid in TOP: return 0
    if cid in BOTTOM: return 1
    if cid in SHOES: return 2
    if cid in ACCESSORY: return 3
    return None


bbox = {}
with open(BBOX_FILE) as f:
    for line in f.read().splitlines()[2:]:
        img, x1, y1, x2, y2 = line.split()
        bbox[img] = tuple(map(int, (x1, y1, x2, y2)))

cat = {}
with open(CAT_FILE) as f:
    for line in f.read().splitlines()[2:]:
        img, cid = line.split()
        cat[img] = int(cid)

# =====================================================
# 5) Randomly select 3000 samples
# =====================================================
all_imgs = [img for img in bbox.keys() if img in cat]
random.shuffle(all_imgs)

NUM_SAMPLES = 3000
chosen_imgs = all_imgs[:NUM_SAMPLES]

print(f"🚀 Using {len(chosen_imgs)} images for YOLO training")

# =====================================================
# 6) Convert DeepFashion annotations → YOLO format
# =====================================================
for img_name in tqdm(chosen_imgs):

    cid = cat[img_name]
    yolo_class = map_category(cid)
    if yolo_class is None:
        continue

    img_path = IMG_DIR / img_name
    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox[img_name]

    # Normalized YOLO bbox
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    # Random 80/20 split
    split = "train" if random.random() > 0.2 else "val"

    # =====================================================
    # SYMLINK instead of copying image (Saves space!)
    # =====================================================
    dst_img = OUT / f"images/{split}" / img_name
    dst_img.parent.mkdir(parents=True, exist_ok=True)

    if not dst_img.exists():  
        os.symlink(img_path.resolve(), dst_img)

    # Write YOLO .txt label
    dst_lbl = OUT / f"labels/{split}" / img_name.replace(".jpg", ".txt")
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_lbl, "w") as f:
        f.write(f"{yolo_class} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

print("🎉 Finished! YOLO dataset saved → data/raw/dataset_yolo/")

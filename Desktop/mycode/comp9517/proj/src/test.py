import random, shutil, os
from pathlib import Path

train_root = Path("../runs/dataset_cls/train")
for cls_dir in train_root.iterdir():
    imgs = list(cls_dir.glob("*.jpg"))
    keep = random.sample(imgs, 50)
    for img in imgs:
        if img not in keep:
            os.remove(img)
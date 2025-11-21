import os
import cv2
import time
import yaml
import shutil
import random
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_root", type=str, default="data/archive")
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--seed", type=int, default=random.randint(0, 999999))
    return ap.parse_args()


#Some helper functions
def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)



def yolo_box_to_xyxy(lbl_line, W, H):
    parts = lbl_line.strip().split()
    if len(parts) != 5:
        return None, None
    cid = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:])
    xmin = int((cx - w / 2.0) * W)
    ymin = int((cy - h / 2.0) * H)
    xmax = int((cx + w / 2.0) * W)
    ymax = int((cy + h / 2.0) * H)
    return cid, (xmin, ymin, xmax, ymax)


def clamp_xyxy(xyxy, W, H, margin_ratio=0.05):
    xmin, ymin, xmax, ymax = xyxy
    w = xmax - xmin
    h = ymax - ymin
    dx = int(w * margin_ratio)
    dy = int(h * margin_ratio)
    xmin = max(0, xmin - dx)
    ymin = max(0, ymin - dy)
    xmax = min(W - 1, xmax + dx)
    ymax = min(H - 1, ymax + dy)
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


#crop
def build_classification_dataset(yolo_root: Path, out_root: Path, class_names, margin_ratio=0.05, min_crop=10):
    ds_out = out_root / "dataset_cls"

    if ds_out.exists() and any((ds_out / "train").iterdir()):
        return ds_out

    ensure_clean_dir(ds_out)

    for split in ("train", "valid", "test"):
        for cname in class_names:
            ensure_clean_dir(ds_out / split / cname)

    for split in ("train", "valid", "test"):
        img_dir = yolo_root / split / "images"
        lbl_dir = yolo_root / split / "labels"
        img_paths = list(img_dir.glob("*.jpg"))

        for img_path in img_paths:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            with open(lbl_path, "r") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            for i, ln in enumerate(lines):
                cid, xyxy = yolo_box_to_xyxy(ln, W, H)
                if cid is None or xyxy is None:
                    continue
                xyxy = clamp_xyxy(xyxy, W, H, margin_ratio)
                if xyxy is None:
                    continue
                x1, y1, x2, y2 = xyxy
                if (x2 - x1) < min_crop or (y2 - y1) < min_crop:
                    continue
                crop = img[y1:y2, x1:x2].copy()
                cname = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
                out_path = ds_out / split / cname / f"{img_path.stem}_{i}.jpg"
                cv2.imwrite(str(out_path), crop)
    return ds_out


#find it
def get_dataloaders(ds_root: Path, img_size=224, batch_size=64):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    print("load check")
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    print("start check")
    train_ds = datasets.ImageFolder(str(ds_root / "train"), transform=tf_train)

    valid_ds = datasets.ImageFolder(str(ds_root / "valid"), transform=tf_eval)
    test_ds  = datasets.ImageFolder(str(ds_root / "test"), transform=tf_eval)

    print("checkpoint1")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print("checkpoint2")

    return train_dl, valid_dl, test_dl, train_ds.classes

def create_densenet(num_classes, freeze_backbone=False):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm, y_true, y_pred


def plot_confusion_matrix(cm, classes, out_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_and_eval(ds_root: Path, out_root: Path, num_classes, args):
    device = get_device()
    model = create_densenet(num_classes, args.freeze_backbone).to(device)
    train_dl, valid_dl, test_dl, classes = get_dataloaders(ds_root, batch_size=args.batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    print("checkpoint3")

    save_dir = out_root / f"densenet_run_{time.strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_path = -1, save_dir / "best_densenet.pth"

    train_losses, valid_f1s = [], []
    print("checkpoint4")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_dl:
            print("checkpoint7")
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        print("checkpoint6")
        scheduler.step()

        print("checkpoint5")

        train_loss = running_loss / len(train_dl.dataset)
        acc, f1, cm, _, _ = evaluate(model, valid_dl, device)
        print(f"[epoch {epoch:02d}] loss={train_loss:.4f} | valid_acc={acc:.4f} f1={f1:.4f}")

        train_losses.append(train_loss)
        valid_f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ saved best model to {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    acc_t, f1_t, cm_t, y_true_t, y_pred_t = evaluate(model, test_dl, device)
    plot_confusion_matrix(cm_t, classes, save_dir / "cm_test.png")
    report_path = save_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_true_t, y_pred_t, target_names=classes, digits=4))

    print(f"\n=== DONE ===\")



def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using fixed seed: {args.seed}")
    else:
        print("No seed fixed — results will be non-deterministic.")

    yolo_root = Path("data")
    out_root = Path("runs")
    config_file = yolo_root / "data.yaml"

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    class_names = cfg["names"]
    
    ds_cls_root = build_classification_dataset(
        yolo_root=yolo_root,
        out_root=out_root,
        class_names=class_names,
        margin_ratio=0.07,
        min_crop=10
    )

    train_and_eval(ds_cls_root, out_root, len(class_names), args)


if __name__ == "__main__":
    main()

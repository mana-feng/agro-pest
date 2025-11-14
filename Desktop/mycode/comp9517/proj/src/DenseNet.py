"""
从 YOLO 数据集构建 DenseNet 分类模型的完整流程
=====================================================
步骤：
1. 从 YOLO 数据集读取 (images + labels)
2. 按 YOLO 标签裁剪目标图片，构建分类数据集
3. 使用 DenseNet121 训练分类器
4. 输出指标、混淆矩阵、曲线、报告

用法：
    python train_from_yolo_densenet.py --yolo_root data/archive --epochs 30
"""

import os
import cv2
import time
import yaml
import shutil
import random
import argparse
import numpy as np
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





# -----------------------------
# 参数解析
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_root", type=str, default="data/archive", help="YOLO 数据集根目录")
    ap.add_argument("--out_root", type=str, default="runs", help="输出根目录")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--freeze_backbone", action="store_true", help="是否冻结特征提取层")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# -----------------------------
# 工具函数
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# -----------------------------
# YOLO 工具函数
# -----------------------------
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


# -----------------------------
# 步骤 1：从 YOLO 数据集裁剪分类数据
# -----------------------------
def build_classification_dataset(yolo_root: Path, out_root: Path, class_names, margin_ratio=0.05, min_crop=10):
    ds_out = out_root / "dataset_cls"

    if ds_out.exists() and any((ds_out / "train").iterdir()):
        print(f"[build] ⚡ Skipping rebuild — using existing dataset: {ds_out}")
        return ds_out

    ensure_clean_dir(ds_out)

    for split in ("train", "valid", "test"):
        for cname in class_names:
            ensure_clean_dir(ds_out / split / cname)

    for split in ("train", "valid", "test"):
        img_dir = yolo_root / split / "images"
        lbl_dir = yolo_root / split / "labels"
        img_paths = list(img_dir.glob("*.jpg"))

        print(f"[build] {split}: {len(img_paths)} images")
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
    print(f"[build] ✅ Cropped classification dataset at: {ds_out}")
    return ds_out


# -----------------------------
# 步骤 2：加载分类数据
# -----------------------------
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
    print("getdata loader train complete")
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    print("getdata loader eval complete")

    print("train_ds starting")
    train_ds = datasets.ImageFolder(str(ds_root / "train"), transform=tf_train)
    print("train_ds complete")

    valid_ds = datasets.ImageFolder(str(ds_root / "valid"), transform=tf_eval)
    test_ds  = datasets.ImageFolder(str(ds_root / "test"), transform=tf_eval)

    print("checkpoint1")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print("checkpoint2")

    return train_dl, valid_dl, test_dl, train_ds.classes


# -----------------------------
# 步骤 3：DenseNet 模型
# -----------------------------
def create_densenet(num_classes, freeze_backbone=False):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


# -----------------------------
# 步骤 4：训练与评估
# -----------------------------
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
    print(f"[device] {device}")



    print("[init] Creating DenseNet model ...")
    model = create_densenet(num_classes, args.freeze_backbone).to(device)
    print("[init] DenseNet model loaded ")



    train_dl, valid_dl, test_dl, classes = get_dataloaders(ds_root, batch_size=args.batch_size)

    #model = create_densenet(num_classes, args.freeze_backbone).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    print("checkpoint3")

    save_dir = out_root / f"densenet_run_{time.strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_path = -1, save_dir / "best_densenet.pth"

    train_losses, valid_f1s = [], []
    print("checkpoint4")

    print("Train dataset size:", len(train_dl.dataset))
    print("Train dataloader batches:", len(train_dl))

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

    print(f"\n=== DONE ===\nSaved in {save_dir}\nTest Acc={acc_t:.4f}, F1={f1_t:.4f}")


# -----------------------------
# 主函数
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    yolo_root = Path("data")
    out_root = Path("runs")
    config_file = yolo_root / "data.yaml"

    # 读取 YOLO 类别信息
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    class_names = cfg["names"]

    print(f"[YOLO] found {len(class_names)} classes: {class_names}")

    # 步骤 1：裁剪分类数据集
    ds_cls_root = build_classification_dataset(
        yolo_root=yolo_root,
        out_root=out_root,
        class_names=class_names,
        margin_ratio=0.07,
        min_crop=10
    )

    # 步骤 2-4：训练 DenseNet
    train_and_eval(ds_cls_root, out_root, len(class_names), args)


if __name__ == "__main__":
    main()

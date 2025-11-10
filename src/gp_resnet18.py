import os
from pathlib import Path
import random
import shutil
import time
import csv
import argparse
import yaml

import kagglehub
import cv2 as cv
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt

# Default CONFIG (can be overridden by config.yaml / CLI)
CONFIG = {
    "YOLO_ROOT": str(Path("data/archive")),
    "OUT_ROOT": str(Path("runs")),
    "CLASS_NAMES": [
        "Ants", "Bees", "Beetles", "Caterpillars", "Earthworms", "Earwigs",
        "Grasshoppers", "Moths", "Slugs", "Snails", "Wasps", "Weevils"
    ],
    # Cropping
    "CROP_MARGIN_RATIO": 0.07,
    "MIN_CROP_SIZE": 10,
    # Training
    "IMG_SIZE": 256,
    "BATCH_SIZE": 64,
    "NUM_WORKERS": 2,
    "EPOCHS": 30,
    "LR": 4e-4,
    "WEIGHT_DECAY": 1e-4,
    "SEED": 42,
    "SAVE_BEST_METRIC": "macro_f1",  # or "acc"
    "FREEZE_BACKBONE": False,
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml", help="path to config.yaml")
    # Optional CLI overrides:
    ap.add_argument("--yolo_root", type=str)
    ap.add_argument("--out_root", type=str)
    ap.add_argument("--img_size", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight_decay", type=float)
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--save_best_metric", type=str, choices=["acc", "macro_f1"])
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--seed", type=int)
    return ap.parse_args()


def load_config_with_overrides(default_cfg: dict):
    args = parse_args()
    cfg = default_cfg.copy()

    # 1) YAML
    if args.cfg and Path(args.cfg).exists():
        with open(args.cfg, "r") as f:
            y = yaml.safe_load(f) or {}
        keymap = {
            "yolo_root": "YOLO_ROOT",
            "out_root": "OUT_ROOT",
            "class_names": "CLASS_NAMES",
            "crop_margin_ratio": "CROP_MARGIN_RATIO",
            "min_crop_size": "MIN_CROP_SIZE",
            "img_size": "IMG_SIZE",
            "epochs": "EPOCHS",
            "batch": "BATCH_SIZE",
            "lr": "LR",
            "weight_decay": "WEIGHT_DECAY",
            "num_workers": "NUM_WORKERS",
            "save_best_metric": "SAVE_BEST_METRIC",
            "freeze_backbone": "FREEZE_BACKBONE",
            "seed": "SEED",
        }
        for k, v in y.items():
            k2 = keymap.get(k, k)
            if k2 in cfg and v is not None:
                cfg[k2] = v

    # 2) CLI overrides
    cli_map = {
        "yolo_root": "YOLO_ROOT",
        "out_root": "OUT_ROOT",
        "img_size": "IMG_SIZE",
        "epochs": "EPOCHS",
        "batch": "BATCH_SIZE",
        "lr": "LR",
        "weight_decay": "WEIGHT_DECAY",
        "num_workers": "NUM_WORKERS",
        "save_best_metric": "SAVE_BEST_METRIC",
        "seed": "SEED",
    }
    for arg_name, key in cli_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            cfg[key] = val

    if args.freeze_backbone:
        cfg["FREEZE_BACKBONE"] = True

    return cfg

# Utils
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def clamp_xyxy(xyxy, W, H, margin_ratio=0.0):
    xmin, ymin, xmax, ymax = xyxy
    if margin_ratio > 0:
        w = xmax - xmin
        h = ymax - ymin
        dx = int(w * margin_ratio)
        dy = int(h * margin_ratio)
        xmin -= dx; ymin -= dy; xmax += dx; ymax += dy
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(W - 1, xmax); ymax = min(H - 1, ymax)
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin, ymin, xmax, ymax)


def ensure_clean_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# Build cropped classification dataset
def build_classification_dataset(yolo_root: Path, out_root: Path, class_names, margin_ratio=0.05, min_crop=10):
    ds_out = out_root / "dataset_cls"
    if ds_out.exists():
        shutil.rmtree(ds_out)
    for split in ("train", "valid", "test"):
        for cname in class_names:
            ensure_clean_dir(ds_out / split / cname)

    for split in ("train", "valid", "test"):
        img_dir = yolo_root / split / "images"
        lbl_dir = yolo_root / split / "labels"
        img_paths = list(img_dir.glob("*.jpg"))
        print(f"[build] {split}: {len(img_paths)} images")
        for img_path in img_paths:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            img = cv.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            with open(lbl_path, "r") as f:
                lines = [ln for ln in f.read().strip().splitlines() if ln.strip()]
            for i, ln in enumerate(lines):
                cid, xyxy = yolo_box_to_xyxy(ln, W, H)
                if cid is None or xyxy is None:
                    continue
                xyxy = clamp_xyxy(xyxy, W, H, margin_ratio=margin_ratio)
                if xyxy is None:
                    continue
                x1, y1, x2, y2 = xyxy
                if (x2 - x1) < min_crop or (y2 - y1) < min_crop:
                    continue
                crop = img[y1:y2, x1:x2].copy()
                cname = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
                out_dir = ds_out / split / cname
                out_name = f"{img_path.stem}_{i}.jpg"
                cv.imwrite(str(out_dir / out_name), crop)
    print(f"[build] Cropped classification dataset at: {ds_out}")
    return ds_out

# DataLoaders
def get_dataloaders(ds_root: Path, img_size=224, batch=64, workers=2):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.95, 1.0)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    train_ds = datasets.ImageFolder(str(ds_root / "train"), transform=tf_train)
    valid_ds = datasets.ImageFolder(str(ds_root / "valid"), transform=tf_eval)
    test_ds = datasets.ImageFolder(str(ds_root / "test"), transform=tf_eval)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return train_dl, valid_dl, test_dl, train_ds.classes

# Model: ResNet18
def create_model(num_classes: int, freeze_backbone: bool = False):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# Eval helpers
@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in dl:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm, y_true, y_pred


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# Train & Eval pipeline
def train_and_eval(ds_root: Path, out_root: Path, num_classes: int, cfg: dict):
    device = get_device()
    print(f"[device] {device}")

    train_dl, valid_dl, test_dl, classes = get_dataloaders(
        ds_root, img_size=cfg["IMG_SIZE"], batch=cfg["BATCH_SIZE"], workers=cfg["NUM_WORKERS"]
    )
    print(f"[data] classes: {classes}")

    model = create_model(num_classes=num_classes, freeze_backbone=cfg["FREEZE_BACKBONE"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["EPOCHS"])
    criterion = nn.CrossEntropyLoss()

    best_metric = -1.0
    save_dir = Path(out_root / f"resnet18_run_{time.strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_w_path = save_dir / "best_resnet18.pth"

    train_losses, valid_accs, valid_f1s = [], [], []

    for epoch in range(1, cfg["EPOCHS"] + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_dl.dataset)
        acc, prec, rec, f1, cm, _, _ = evaluate(model, valid_dl, device)
        metric = f1 if cfg["SAVE_BEST_METRIC"] == "macro_f1" else acc

        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} | "
              f"valid_acc={acc:.4f} valid_prec={prec:.4f} valid_rec={rec:.4f} valid_f1={f1:.4f}")

        train_losses.append(train_loss)
        valid_accs.append(acc)
        valid_f1s.append(f1)

        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), best_w_path)
            print(f"  ↳ saved best to {best_w_path}")

    # Curves
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / 'curve_loss.png', dpi=200); plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(valid_accs, label='Valid Acc')
    plt.plot(valid_f1s, label='Valid F1')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / 'curve_metrics.png', dpi=200); plt.close()

    # Load best and evaluate
    model.load_state_dict(torch.load(best_w_path, map_location=device))
    acc_v, prec_v, rec_v, f1_v, cm_v, _, _ = evaluate(model, valid_dl, device)
    acc_t, prec_t, rec_t, f1_t, cm_t, y_true_t, y_pred_t = evaluate(model, test_dl, device)

    # Save confusion matrices
    plot_confusion_matrix(cm_v, classes, save_dir / "cm_valid.png")
    plot_confusion_matrix(cm_t, classes, save_dir / "cm_test.png")

    # Save csv
    metrics_csv = save_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "accuracy", "precision_macro", "recall_macro", "f1_macro"])
        w.writerow(["valid", f"{acc_v:.6f}", f"{prec_v:.6f}", f"{rec_v:.6f}", f"{f1_v:.6f}"])
        w.writerow(["test",  f"{acc_t:.6f}", f"{prec_t:.6f}", f"{rec_t:.6f}", f"{f1_t:.6f}"])

    # Classification report
    report_txt = save_dir / "classification_report_test.txt"
    with open(report_txt, "w") as f:
        f.write(classification_report(y_true_t, y_pred_t, target_names=classes, digits=4))

    print("\n=== DONE ===")
    print(f"Artifacts saved in: {save_dir}")
    print(f"  - Best weights: {best_w_path}")
    print(f"  - Valid CM: {save_dir / 'cm_valid.png'}")
    print(f"  - Test  CM: {save_dir / 'cm_test.png'}")
    print(f"  - Metrics : {metrics_csv}")
    print(f"  - Report  : {report_txt}")


def find_yolo_dataset_root(dataset_path: Path):
    """Find the root directory containing train/valid/test splits."""
    dataset_path = Path(dataset_path)
    
    # Check if dataset_path itself has train/valid/test
    if (dataset_path / "train" / "images").exists():
        return dataset_path
    
    # Search for common subdirectories
    for possible_dir in ["archive", "data", "dataset"]:
        candidate = dataset_path / possible_dir
        if (candidate / "train" / "images").exists():
            return candidate
    
    # Search in immediate subdirectories
    if dataset_path.exists() and dataset_path.is_dir():
        for subdir in dataset_path.iterdir():
            if subdir.is_dir() and (subdir / "train" / "images").exists():
                return subdir
    
    # If not found, return original path and let it fail with a clear error
    return dataset_path


def main():
    cfg = load_config_with_overrides(CONFIG)
    set_seed(cfg["SEED"])

    # Download dataset from Kaggle
    print("Downloading dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download("rupankarmajumdar/crop-pests-dataset")
        print(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have Kaggle credentials set up.")
        print("See README.md for instructions on setting up Kaggle authentication.")
        return
    
    # Find the correct YOLO dataset root
    yolo_root = find_yolo_dataset_root(Path(dataset_path))
    print(f"Using dataset root: {yolo_root}")
    
    # Verify dataset structure
    required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels", "test/images", "test/labels"]
    missing_dirs = []
    for req_dir in required_dirs:
        if not (yolo_root / req_dir).exists():
            missing_dirs.append(req_dir)
    
    if missing_dirs:
        print(f"Error: Missing required directories: {missing_dirs}")
        print(f"Expected structure: {yolo_root}/{{train,valid,test}}/{{images,labels}}/")
        return
    
    out_root = Path(cfg["OUT_ROOT"])

    # 1) Build cropped classification dataset
    ds_cls_root = build_classification_dataset(
        yolo_root=yolo_root,
        out_root=out_root,
        class_names=cfg["CLASS_NAMES"],
        margin_ratio=cfg["CROP_MARGIN_RATIO"],
        min_crop=cfg["MIN_CROP_SIZE"]
    )

    # 2) Train & Evaluate
    train_and_eval(
        ds_root=ds_cls_root,
        out_root=out_root,
        num_classes=len(cfg["CLASS_NAMES"]),
        cfg=cfg
    )


if __name__ == "__main__":
    main()
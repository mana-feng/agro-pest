# ---------------------------------------------------------------------------
# 农作物害虫分类实验 - 公共工具模块
# ---------------------------------------------------------------------------
# 将三个模型 notebook (SVM, ResNet18, Swin-Base) 中完全相同的逻辑集中到此处,
# 各 notebook 只保留自身模型特有的代码.
#
# 功能分块:
#   1. 路径解析       -- resolve_paths
#   2. 可复现性       -- set_seed
#   3. 设备选择       -- get_device
#   4. YOLO 标签解析  -- yolo_box_to_xyxy, clamp_xyxy
#   5. 目录工具       -- ensure_clean_dir
#   6. 数据集构建     -- build_classification_dataset
#   7. PyTorch 评估   -- evaluate
#   8. 可视化         -- plot_confusion_matrix, plot_training_curves
#   9. 结果保存       -- save_metrics_csv, save_classification_report
# ---------------------------------------------------------------------------

# --- 标准库 ---
import os
import random
import csv
import shutil
from pathlib import Path

# --- 第三方库 ---
import cv2 as cv
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===================================================================
# 1. 路径解析
# ===================================================================
# 在 Jupyter notebook 中运行时, 工作目录可能是 notebook 所在文件夹,
# 也可能是项目根目录 (取决于 IDE / 启动方式).
# resolve_paths() 依次尝试两种情况, 返回实际包含 "archive" 数据目录的那一个.
#
# 返回值:
#   project_root  -- 仓库根目录的绝对路径 (agro-pest/)
#   data_root     -- <project_root>/archive  (YOLO 格式数据集)
#   out_root      -- <project_root>/runs     (所有模型输出)
# ===================================================================

def resolve_paths():
    # 第一次尝试: 假设 cwd 在项目根目录下两层
    #   例: cwd = agro-pest/models/resnet18/  ->  ../.. = agro-pest/
    project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    if not os.path.isdir(os.path.join(project_root, "archive")):
        # 回退方案: 使用 Path.resolve() 获取真实文件系统路径,
        # 不受符号链接或 Jupyter 特殊行为影响
        project_root = os.path.abspath(os.path.join(str(Path().resolve()), "..", ".."))

    data_root = os.path.join(project_root, "archive")
    out_root = os.path.join(project_root, "runs")

    print(f"PROJECT_ROOT: {project_root}")
    print(f"DATA_ROOT:    {data_root}")
    print(f"OUT_ROOT:     {out_root}")
    print(f"Data exists:  {os.path.isdir(data_root)}")
    return project_root, data_root, out_root


# ===================================================================
# 2. 可复现性
# ===================================================================
# 固定所有随机源, 使重复运行产生完全一致的结果 (调试与公平对比所必需).
# ===================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===================================================================
# 3. 设备选择
# ===================================================================
# 按优先级选择最快的加速器:
#   Apple Silicon  ->  "mps"
#   NVIDIA GPU     ->  "cuda"
#   兜底           ->  "cpu"
# ===================================================================

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ===================================================================
# 4. YOLO 标签解析
# ===================================================================
# YOLO 格式将边界框存储为归一化的中心坐标:
#   <类别id> <cx> <cy> <宽> <高>
# 其中 cx/cy/w/h 均为 [0, 1] 范围, 相对于图像尺寸.
#
# yolo_box_to_xyxy 将一行标签转换为绝对像素坐标 (xmin, ymin, xmax, ymax).
#
# clamp_xyxy 在此基础上:
#   - 按比例向外扩展边界框 (让裁剪区域包含害虫周围的一点上下文);
#   - 将坐标裁剪到图像边界内;
#   - 若裁剪后框退化则返回 None.
# ===================================================================

def yolo_box_to_xyxy(lbl_line, W, H):
    parts = lbl_line.strip().split()
    if len(parts) != 5:
        return None, None
    cid = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:])
    # 将归一化的中心+宽高转换为绝对像素角点坐标
    xmin = int((cx - w / 2.0) * W)
    ymin = int((cy - h / 2.0) * H)
    xmax = int((cx + w / 2.0) * W)
    ymax = int((cy + h / 2.0) * H)
    return cid, (xmin, ymin, xmax, ymax)


def clamp_xyxy(xyxy, W, H, margin_ratio=0.0):
    xmin, ymin, xmax, ymax = xyxy
    # 按比例向外扩展边界框, 每侧扩展 margin_ratio * 框尺寸
    if margin_ratio > 0:
        w = xmax - xmin
        h = ymax - ymin
        dx = int(w * margin_ratio)
        dy = int(h * margin_ratio)
        xmin -= dx
        ymin -= dy
        xmax += dx
        ymax += dy
    # 裁剪到有效像素范围
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(W - 1, xmax)
    ymax = min(H - 1, ymax)
    # 防止退化框 (宽或高 <= 0)
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin, ymin, xmax, ymax)


# ===================================================================
# 5. 目录工具
# ===================================================================
# 简单封装: 创建目录 (含父目录), 若已存在也不报错.
# ===================================================================

def ensure_clean_dir(p):
    p.mkdir(parents=True, exist_ok=True)


# ===================================================================
# 6. 数据集构建
# ===================================================================
# 原始数据集为 YOLO 格式 (图片 + 每张图一个 .txt 标签文件).
# 图像分类模型 (ResNet18, Swin-Base) 需要 ImageFolder 兼容的目录结构:
#
#   dataset_cls/
#     train/
#       Ants/   <蚂蚁的裁剪图片>
#       Bees/   ...
#     valid/  ...
#     test/   ...
#
# build_classification_dataset 读取每张 YOLO 图片, 裁剪每个边界框
# (可加 margin), 将裁剪结果写入对应的类别文件夹.
# 生成的目录可直接传给 torchvision.datasets.ImageFolder.
#
# 参数:
#   yolo_root     -- YOLO 数据集根路径 (含 train/valid/test 子目录)
#   out_root      -- 分类数据集输出路径
#   class_names   -- 类别 id (int) -> 文件夹名称 的映射列表
#   margin_ratio  -- 边界框向外扩展比例 (0.05 = 5%)
#   min_crop      -- 丢弃小于此尺寸的裁剪 (任一边)
#
# 返回值:
#   ds_out  -- 生成的 dataset_cls/ 目录的 Path 对象
# ===================================================================

def build_classification_dataset(yolo_root, out_root, class_names, margin_ratio=0.05, min_crop=10):
    ds_out = Path(out_root) / "dataset_cls"

    # 清除上次构建, 避免残留过期裁剪
    if ds_out.exists():
        shutil.rmtree(ds_out)

    # 预先创建完整目录树: split / class_name
    for split in ("train", "valid", "test"):
        for cname in class_names:
            ensure_clean_dir(ds_out / split / cname)

    # 遍历每个 split, 裁剪并保存
    for split in ("train", "valid", "test"):
        img_dir = Path(yolo_root) / split / "images"
        lbl_dir = Path(yolo_root) / split / "labels"

        # 收集图片路径, 仅保留常见图片格式
        img_paths = list(img_dir.glob("*.*"))
        img_paths = [p for p in img_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        print(f"[build] {split}: {len(img_paths)} images")

        for img_path in tqdm(img_paths, desc=f"Build {split}"):
            # 通过文件名主干匹配标签文件 (不含扩展名)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            img = cv.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            # 读取该图片的所有边界框行
            with open(lbl_path, "r") as f:
                lines = [ln for ln in f.read().strip().splitlines() if ln.strip()]

            for i, ln in enumerate(lines):
                # 解析 YOLO 归一化坐标 -> 绝对像素坐标
                cid, xyxy = yolo_box_to_xyxy(ln, W, H)
                if cid is None or xyxy is None:
                    continue

                # 按 margin 扩展并裁剪到图像边界
                xyxy = clamp_xyxy(xyxy, W, H, margin_ratio=margin_ratio)
                if xyxy is None:
                    continue

                x1, y1, x2, y2 = xyxy

                # 跳过过小的无意义裁剪
                if (x2 - x1) < min_crop or (y2 - y1) < min_crop:
                    continue

                # 裁剪并复制 (防止原数组被释放后视图失效), 然后保存
                crop = img[y1:y2, x1:x2].copy()
                cname = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
                out_dir = ds_out / split / cname
                out_name = f"{img_path.stem}_{i}.jpg"
                cv.imwrite(str(out_dir / out_name), crop)

    print(f"[build] Cropped classification dataset at: {ds_out}")

    # 打印每个 split 的样本数, 用于快速验证
    for split in ("train", "valid", "test"):
        split_dir = ds_out / split
        total = sum(len(list((split_dir / c).glob("*"))) for c in class_names)
        print(f"  {split}: {total} samples")

    return ds_out


# ===================================================================
# 7. PyTorch 评估
# ===================================================================
# 在推理模式 (无梯度) 下对 DataLoader 运行模型, 计算标准分类指标.
#
# 返回值:
#   acc, prec, rec, f1  -- 标量指标 (宏平均)
#   cm                  -- 混淆矩阵 (numpy 数组)
#   y_true, y_pred      -- 原始标签列表 (用于进一步分析)
# ===================================================================

@torch.no_grad()
def evaluate(model, dl, device, desc="Eval"):
    model.eval()
    y_true, y_pred = [], []
    for x, y in tqdm(dl, desc=desc, leave=False):
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


# ===================================================================
# 8. 可视化
# ===================================================================

# --- 8a. 混淆矩阵热力图 ---
# 绘制方形热力图, 两个轴均为类别名称, 每个格子标注整数计数值
# (深色格用白色文字, 浅色格用黑色文字), 保存到 out_path (200 dpi).

def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每个格子中标注计数值, 根据背景深浅选择文字颜色以保证可读性
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)


# --- 8b. 训练曲线 ---
# 分别保存两张 PNG:
#   curve_loss.png    -- 训练损失 vs. epoch
#   curve_metrics.png -- 验证集准确率 & F1 vs. epoch

def plot_training_curves(train_losses, valid_accs, valid_f1s, save_dir):
    # 损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "curve_loss.png", dpi=200)
    plt.show()

    # 准确率 & F1 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(valid_accs, label="Valid Acc")
    plt.plot(valid_f1s, label="Valid F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "curve_metrics.png", dpi=200)
    plt.show()


# ===================================================================
# 9. 结果保存
# ===================================================================

# --- 9a. 指标 CSV ---
# 写入一个精简的 CSV, 包含两行 (valid / test) 和四列指标,
# 方便跨运行对比而无需打开完整分类报告.

def save_metrics_csv(save_dir, acc_v, prec_v, rec_v, f1_v, acc_t, prec_t, rec_t, f1_t):
    metrics_csv = save_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "accuracy", "precision_macro", "recall_macro", "f1_macro"])
        w.writerow(["valid", f"{acc_v:.6f}", f"{prec_v:.6f}", f"{rec_v:.6f}", f"{f1_v:.6f}"])
        w.writerow(["test", f"{acc_t:.6f}", f"{prec_t:.6f}", f"{rec_t:.6f}", f"{f1_t:.6f}"])
    return metrics_csv


# --- 9b. 分类报告 ---
# 将 sklearn 的 classification_report (每个类别的精确率/召回率/F1/支持数)
# 保存为纯文本文件, 用于归档.

def save_classification_report(save_dir, y_true, y_pred, class_names):
    report_txt = save_dir / "classification_report_test.txt"
    with open(report_txt, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    return report_txt

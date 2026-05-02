# 农作物害虫分类 -- 机器学习模型优化指南

基于当前项目状态: 13 类害虫, 2836 维传统特征, LightGBM/SVM 两个基线模型.

---

## 1. 特征工程优化

### 1.1 特征选择与降维

当前 2836 维特征中 HOG 占 1764 维 (62%), 存在明显冗余.

| 方法 | 说明 | 预期收益 |
|------|------|----------|
| 方差阈值过滤 | 删除方差接近 0 的特征 (如全零/近常量) | 去除无效维度, 加速训练 |
| 相关性去冗余 | 计算特征间 Pearson 相关系数, 阈值 >0.95 的只保留其一 | LBP 与多尺度 LBP 高度相关, 可削减 |
| PCA 降维 | 保留 95% 方差, 预计可从 2836 维降至 300-500 维 | 消除多重共线性, 加速 SVM |
| 基于特征重要性筛选 | 用 LightGBM 的 `feature_importances_` 取 Top-K | 保留对分类最有贡献的特征组 |
| 互信息法 | `sklearn.feature_selection.mutual_info_classif` | 捕捉非线性依赖, 比 F 检验更通用 |

推荐优先级: **方差阈值 -> 特征重要性筛选 -> PCA**

```python
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.decomposition import PCA

# 方差阈值
selector = VarianceThreshold(threshold=1e-5)
X_sel = selector.fit_transform(X_train)

# PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_sel)
print(f"PCA: {X_sel.shape[1]} -> {X_pca.shape[1]} dims, explained variance: {pca.explained_variance_ratio_.sum():.4f}")
```

### 1.2 特征组级别优化

| 特征组 | 当前维度 | 问题 | 建议 |
|--------|----------|------|------|
| HOG | 1764 | 占比过大, 与轮廓几何/Fourier 有信息重叠 | 降低 `pixels_per_cell` 分辨率或用 PCA 压缩 |
| HSV 直方图 | 512 | bins=512 过细, 稀疏 | 降至 bins=64 或 128 |
| LBP + 多尺度 LBP | 59+177=236 | 两者高度相关 | 保留多尺度 LBP, 去除单尺度 LBP |
| 颜色矩 + HSV 直方图 | 9+512=521 | 颜色矩是直方图的低阶统计量 | 可考虑去除颜色矩或直方图二选一 |
| Gabor | 96 | 计算开销大, 与 LBP 纹理信息重叠 | 通过特征重要性评估是否保留 |

### 1.3 增加 Bbox 元数据和背景特征

当前 `extract_features` 未包含 Bbox 元数据 (5 维) 和背景特征 (22 维). 这两组特征在 `EXPECTED_DIMS` 中定义但未加入 `EXTRACTED_DIMS`.

- **Bbox 元数据**: 害虫大小/宽高比/位置是强判别信号 (如 Earthworms 细长, Snails 偏圆)
- **背景特征**: 不同害虫栖息环境不同 (叶片/土壤/水面), 背景颜色和纹理可辅助分类

```python
# 在 extract_features() 中添加:
feat_parts["Bbox 元数据"] = extract_bbox_metadata(xyxy, W, H)
feat_parts["背景特征"] = extract_background_features(roi, xyxy, W, H)
```

---

## 2. 超参数调优

### 2.1 LightGBM

当前配置偏保守, 可通过系统搜索找到更优参数.

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    model = lgb.LGBMClassifier(**params, class_weight="balanced", random_state=42)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return accuracy_score(y_val, model.predict(X_val))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, timeout=3600)
print(f"Best score: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

关键调优方向:

| 参数 | 当前值 | 调优范围 | 说明 |
|------|--------|----------|------|
| `num_leaves` | 31 | 15-127 | 控制模型复杂度, 过大易过拟合 |
| `max_depth` | -1 (无限) | 3-15 | 限制深度可防止过拟合 |
| `learning_rate` | 0.05 | 0.01-0.3 | 小学习率 + 更多树通常更稳 |
| `n_estimators` | 500 | 配合 early_stopping 可设 1000+ | |
| `min_child_samples` | 20 | 5-50 | 叶子最小样本数, 防过拟合 |
| `reg_alpha/lambda` | 0.1 | 1e-3-10 | L1/L2 正则化 |

### 2.2 SVM

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "svc__C": [1, 10, 50, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1],
    "svc__kernel": ["rbf"],
}
grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)
```

SVM 在高维特征上对 `C` 和 `gamma` 非常敏感, GridSearch 通常能带来 3-5% 的提升.

---

## 3. 数据层面优化

### 3.1 数据增强

在 ROI 裁剪后、特征提取前对图像做增强, 扩充训练集.

```python
def augment_roi(roi):
    augmented = [roi]
    h, w = roi.shape[:2]
    center = (w // 2, h // 2)

    # 水平翻转
    augmented.append(cv.flip(roi, 1))

    # 小角度旋转
    for angle in [-15, 15]:
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(roi, M, (w, h), borderMode=cv.BORDER_REFLECT)
        augmented.append(rotated)

    # 亮度扰动
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV).astype(np.float32)
    for factor in [0.8, 1.2]:
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        augmented.append(cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR))

    return augmented
```

在 `extract_dataset_features` 中使用:

```python
for roi_aug in augment_roi(roi):
    roi_aug = cv.resize(roi_aug, resize_to)
    feature = extract_features(roi_aug, hog_desc)
    X.append(feature)
    y.append(cid)
```

### 3.2 类别不平衡处理

当前训练集类别分布不均 (Ants 2221 vs Weevils 972).

| 方法 | 说明 |
|------|------|
| `class_weight="balanced"` | 已在使用, 按类别频率反比加权 |
| SMOTE 过采样 | 对少数类合成新样本 |
| 欠采样 + 集成 | 对多数类下采样, 训练多个模型取平均 |

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 3.3 提高 ROI 裁剪质量

当前 `CROP_MARGIN_RATIO=0.05`, 裁剪框紧贴目标, 可能丢失边缘纹理.

- 尝试 `0.10` 或 `0.15`, 保留更多上下文
- `RESIZE_TO=(64, 64)` 可能丢失细节, 尝试 `128x128` (HOG 维度会变为 3240, 需评估开销)

---

## 4. 模型集成

### 4.1 异质模型融合

LightGBM 和 SVM 的决策边界不同, 融合可互补.

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[("lgbm", lgb_model), ("svm", svm_model)],
    voting="soft",
    weights=[2, 1],
)
```

### 4.2 Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stacking = StackingClassifier(
    estimators=[("lgbm", lgb_model), ("svm", svm_model)],
    final_estimator=LogisticRegression(max_iter=1000, multi_class="multinomial"),
    cv=5,
)
```

### 4.3 多特征子集 + 多模型

将 13 组特征分为颜色组 / 纹理组 / 形状组, 分别训练模型后融合:

| 分组 | 特征 | 维度 |
|------|------|------|
| 颜色组 | HSV 直方图 + 颜色矩 + 颜色相关图 | 629 |
| 纹理组 | LBP + 多尺度 LBP + GLCM + Gabor | 345 |
| 形状组 | Hu + HOG + 轮廓几何 + Zernike + Fourier + Canny | 1862 |

每组独立训练 LightGBM, 最后用 Softmax 融合概率输出.

---

## 5. 验证策略优化

### 5.1 交叉验证

当前仅用单次 train/test 划分, 结果方差大.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
print(f"5-Fold CV: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### 5.2 验证集调参, 测试集仅最终评估

当前直接在测试集上评估, 多次调参后会导致信息泄露. 建议从训练集划出 20% 作为验证集, 测试集留到最终.

---

## 6. 优化路线图

按投入产出比排序:

| 优先级 | 优化项 | 预期提升 | 实现难度 |
|--------|--------|----------|----------|
| P0 | 超参数搜索 (Optuna/GridSearch) | +3-5% | 低 |
| P0 | 增加 Bbox 元数据 + 背景特征 | +2-4% | 低 |
| P1 | 特征选择 (方差阈值 + 重要性筛选) | +1-3% | 低 |
| P1 | 数据增强 (翻转/旋转/亮度) | +2-4% | 低 |
| P1 | SMOTE 处理类别不平衡 | +1-2% | 低 |
| P2 | LightGBM + SVM 集成 | +2-3% | 中 |
| P2 | 提高 ROI 分辨率 (128x128) | +1-3% | 中 |
| P3 | PCA 降维 + SVM 重训 | +1-2% | 中 |
| P3 | 多特征子集独立建模融合 | +2-4% | 高 |
| P4 | 深度学习替代方案 (ResNet/EfficientNet 微调) | +10-20% | 高 |

> 注: 深度学习不在本文档范围内, 但对于图像分类任务, CNN 微调通常能将传统特征的准确率上限 (约 60-70%) 提升到 85%+.

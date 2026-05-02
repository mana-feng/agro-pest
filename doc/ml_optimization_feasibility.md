# 传统 ML 优化方案可行性分析

## 1. 当前基线性能

| 模型 | 特征维度 | 准确率 | Macro F1 |
|------|----------|--------|----------|
| SVM (RBF) | 2836 | 56.60% | 0.5599 |
| LightGBM | 2836 | 60.96% | 0.6068 |

## 2. 改进方案清单

| # | 改进项 | 预期收益 | 工作量 |
|---|--------|----------|--------|
| 1 | PCA 降维 | 2-5% | 小 |
| 2 | 特征选择 (RFECV/SelectKBest) | 3-8% | 中 |
| 3 | HOG 参数调整 | 1-3% | 小 |
| 4 | 数据增强 (旋转/翻转/颜色抖动) | 5-10% | 中 |
| 5 | Optuna 超参数搜索 (LightGBM) | 3-5% | 中 |
| 6 | GridSearchCV (SVM C/gamma) | 2-4% | 小 |
| 7 | Voting/Stacking 集成 | 5-10% | 大 |

## 3. 开源代码对比

针对 Agricultural Pests 12 类数据集的开源项目均未实现上述优化。

| 项目 | 方法 | 是否实现优化 |
|------|------|-------------|
| Felipe713/agro-pests | ResNet-18 迁移学习 | 否，仅深度学习 |
| jiyak12/Real-Time-Pest | Decision Tree / SVM | 否，仅基础预处理 + ImageDataGenerator (用于 DL) |
| evitanegara/Weed-Classification | HOG+LBP+颜色直方图 | 部分 (GridSearchCV + Voting/Stacking)，但为杂草分类非害虫 |
| Manya0407/SVM-based | HOG+SIFT+颜色直方图 | 否，仅 StandardScaler 标准化 |

**结论：** 针对该数据集的传统 ML 优化属于首创，无现成代码可参考。

## 4. 实施建议

### 阶段一：快速见效

- PCA 降维（保留 95% 方差）
- SelectKBest 特征选择（基于 F 值）
- HOG 参数网格搜索（cells_per_block, orientations）
- SVM GridSearchCV（C, gamma）

### 阶段二：核心优化

- Optuna 超参数搜索（LightGBM）
- ROI 数据增强（旋转/翻转/亮度扰动）

### 阶段三：集成融合

- VotingClassifier（SVM + LightGBM + RF）
- Stacking（多层模型融合）

## 5. 预期上限

| 阶段 | 预期 LightGBM 准确率 | 预期集成准确率 |
|------|---------------------|---------------|
| 当前基线 | 60.96% | - |
| 阶段一完成 | 63-66% | - |
| 阶段二完成 | 66-70% | - |
| 阶段三完成 | - | 70-75% |

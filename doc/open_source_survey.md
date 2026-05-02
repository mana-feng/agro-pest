# Agricultural Pests Image Dataset 开源代码调研

## 1. 数据集概述

本项目使用的数据集为 Kaggle 上的 **Agricultural Pests Image Dataset**，由 vencerlanz09 发布。

- 数据集地址: https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset
- 类别数: 12 类
- 类别名称: ants, bees, beetle, caterpillar, earwig, earthworms, grasshopper, moth, slug, snail, wasp, weevil
- 图像尺寸: 最大 300x300 px
- 标注格式: YOLO 格式 (本项目已转换为 VOC+YOLO)

---

## 2. 使用同一数据集的开源项目

### 2.1 Felipe713/agro-pests

- 仓库: https://github.com/Felipe713/agro-pests
- 方法: PyTorch + ResNet-18 迁移学习
- 数据集: 完全相同的 12 类 Agricultural Pests Image Dataset
- 结果: 3 epoch 验证准确率约 65%
- 特点:
  - 完整的 ML 流水线: 下载 -> 划分 -> 训练 -> 预测
  - 使用 DevContainer 配置开发环境
  - 包含 Jupyter Notebook 和独立脚本两种运行方式
- 局限: 仅使用深度学习方法，未涉及传统特征提取

### 2.2 jiyak12/Real-Time-Pest-Monitoring-System

- 仓库: https://github.com/jiyak12/Real-Time-Pest-Monitoring-System
- 方法: OpenCV 图像预处理 + Decision Tree / SVM
- 数据集: 相同的 12 类 Agricultural Pests Image Dataset
- 特点:
  - 包含图像预处理流水线 (灰度化、缩放、归一化)
  - EDA 分析害虫类型分布
  - 混淆矩阵和准确率可视化
  - 面向 IoT 部署场景
- 局限: 特征提取较为简单，未使用 HOG/LBP/GLCM 等高级特征

---

## 3. 使用传统特征提取 + ML 的相关项目

### 3.1 evitanegara/Weed-Classification-Using-Image-Processing

- 仓库: https://github.com/evitanegara/Weed-Classification-Using-Image-Processing
- 领域: 农业杂草分类 (非害虫)
- 特征: HOG + LBP + 颜色直方图
- 分类器: Logistic Regression + Random Forest + SVC + Voting + Stacking + CNN
- 数据集: Kaggle Plant Seedlings Classification (Charlock vs Cleavers)
- 特点:
  - **与本项目特征提取方法高度相似** (HOG, LBP, 颜色直方图)
  - 集成学习对比 (Voting vs Stacking)
  - GridSearchCV 超参数优化
  - CNN 作为 baseline 对比
- 参考价值: 高 -- 特征提取方案可直接对比
- 备注: 仓库仅含 README，代码通过 Kaggle Notebook 链接提供，未存储至本地 code 目录

### 3.2 Manya0407/SVM-based-image-classification-using-Feature-Descriptors

- 仓库: https://github.com/Manya0407/SVM-based-image-classification-using-Feature-Descriptors
- 领域: 通用图像分类
- 特征: HOG + SIFT + 颜色直方图
- 分类器: SVM (RBF 核)
- 特点:
  - SIFT 特征提取 (本项目未使用)
  - StandardScaler 标准化
  - 模型持久化 (joblib)
- 参考价值: 中 -- SIFT 特征可作为本项目扩展参考

### 3.3 yaremenko8/HOG_SVM

- 仓库: https://github.com/yaremenko8/HOG_SVM
- 领域: 交通标志分类
- 特征: HOG
- 分类器: SVM
- 特点:
  - 最简 HOG + SVM 实现
  - gamma/C 参数网格搜索
- 参考价值: 低 -- 过于简单

---

## 4. 大规模害虫识别项目

### 4.1 adhiiisetiawan/large-scale-pest-recognition

- 仓库: https://github.com/adhiiisetiawan/large-scale-pest-recognition
- 数据集: IP102 (102 类害虫, 75000+ 图像)
- 方法: CNN + 数据增强 + 正则化
- 特点:
  - 基于 Lightning-Hydra 模板
  - 大规模害虫识别 benchmark
  - 完整的训练/评估流水线

### 4.2 mathewGlenn/Corn-Pest-Dataset

- 仓库: https://github.com/mathewGlenn/Corn-Pest-Dataset
- 数据集: 玉米害虫 6 类 + 叶片病害 3 类
- 方法: Mobile CNN + 迁移学习
- 特点:
  - 面向移动端部署
  - 数据来源于 IP102 和 PlantVillage

---

## 5. 农业害虫数据集汇总

| 数据集 | 类别数 | 图像数 | 标注类型 | 来源 |
|--------|--------|--------|----------|------|
| Agricultural Pests (本项目) | 12 | ~26k | YOLO bbox | Kaggle |
| IP102 | 102 | 75000+ | 分类+检测 | CVPR 2019 |
| PlantVillage | 38 | 54000+ | 分类 | 学术 |
| Crop Pest-30 | 30 | ~15000 | CSV | Google Dataset Search |
| 大豆害虫 (firc-dataset) | 12 | 1799 | VOC+YOLO | 51CTO |
| Corn Pest | 9 | - | 分类 | GitHub |

---

## 6. 本项目与开源方案的对比

| 维度 | 本项目 | 典型开源方案 |
|------|--------|-------------|
| 特征数 | 13 种 (2836 维) | 2-3 种 (数百维) |
| 特征类型 | HSV直方图, Hu矩, LBP, HOG, 轮廓几何, GLCM, 颜色矩, 多尺度LBP, Gabor, Zernike, Fourier, Canny, 颜色相关图 | HOG, LBP, 颜色直方图 |
| 分类器 | LightGBM + SVM | ResNet-18 / SVM / RF |
| 数据集 | Agricultural Pests 12 类 | 同左 / IP102 / PlantVillage |
| 特征校验 | 有 (维度/NaN/Inf/全零检测) | 无 |
| 缓存机制 | 有 (维度校验自动失效) | 部分 |

**核心差异**: 本项目在传统特征提取的广度上远超现有开源方案，且具备完善的特征校验和调试机制。LightGBM 用于害虫分类在开源社区中属于较新颖的尝试。

---

## 7. 可参考的改进方向

1. **SIFT/SURF 特征**: Manya0407 项目使用了 SIFT 描述子，可作为额外特征补充
2. **集成学习**: evitanegara 项目对比了 Voting 和 Stacking，可考虑在 LightGBM/SVM 基础上做集成
3. **特征选择**: 2836 维特征中可能存在冗余，可使用 PCA 或基于特征重要性的方法降维
4. **深度特征融合**: 将传统特征与 CNN 提取的深度特征拼接，可能提升分类性能

---

## 8. 本地代码存储索引

| 项目 | 本地路径 | 文件列表 |
|------|----------|----------|
| Felipe713/agro-pests | `code/agro-pests/` | train.py, predict.py, download_data.py, split_data.py |
| Manya0407/SVM-based-image-classification | `code/SVM-based-image-classification/` | svm_classification.py, image_categorization.py, splitting_data.py, gui.py, Comparative Analysis/*.py |
| jiyak12/Real-Time-Pest-Monitoring-System | `code/Real-Time-Pest-Monitoring-System/` | optimising real time pest monitoring system.ipynb |
| evitanegara/Weed-Classification-Using-Image-Processing | - | 仓库仅含 README，代码在 Kaggle Notebook |

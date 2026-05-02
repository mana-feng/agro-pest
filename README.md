# 农业害虫分类

基于传统机器学习和深度学习的农业害虫图像分类系统。

## 项目结构

```
agro-pest/
├── config.yaml                 # 配置文件
├── requirements.txt            # Python 依赖
├── archive/                    # YOLO 格式数据集 (train/val/test)
│   ├── train/images/labels/
│   ├── val/images/labels/
│   └── test/images/labels/
├── models/
│   ├── feature_extraction.py   # 特征提取模块 (13 类特征, 2836 维)
│   ├── shared.py               # 共享工具函数
│   ├── ML/                     # 传统机器学习模型
│   │   ├── crop_pest_svm.ipynb
│   │   ├── crop_pest_lightgbm.ipynb
│   │   └── ML_ALGORITHMS_COMPARISON.md
│   ├── resnet18/gp_resnet18.ipynb
│   ├── densenet/gp_densenet.ipynb
│   └── swinbase/gp_swinbase.ipynb
├── doc/                        # 技术文档
│   ├── open_source_survey.md
│   └── ml_optimization_guide.md
└── result/                     # 评估结果
    ├── ML/
    ├── resnet18/
    ├── densenet/
    └── swinbase/
```

## 数据集

使用 YOLO 格式的边界框标注，共 12 类害虫：

| 类别 | 类别 | 类别 | 类别 |
|------|------|------|------|
| Ants (蚂蚁) | Bees (蜜蜂) | Beetles (甲虫) | Caterpillars (毛虫) |
| Earthworms (蚯蚓) | Earwigs (蠼螋) | Grasshoppers (蝗虫) | Moths (蛾) |
| Slugs (蛞蝓) | Snails (蜗牛) | Wasps (黄蜂) | Weevils (象鼻虫) |

## 特征提取

从裁剪的害虫 ROI (64x64) 中提取手工特征，共 13 类、2836 维：

| 特征 | 维度 | 描述 |
|------|------|------|
| HSV 颜色直方图 | 512 | 颜色分布 (8x8x8 bin) |
| Hu 不变矩 | 7 | 形状不变性 (平移/旋转/缩放不变) |
| LBP 纹理 | 59 | 局部纹理模式 (uniform, P=8, R=1) |
| HOG 方向梯度 | 1764 | 边缘方向梯度 (64x64 窗口, 9 bin) |
| 轮廓几何特征 | 8 | 形状属性 (矩形度、凸度、长宽比等) |
| GLCM Haralick | 13 | 区域纹理统计 (完整 Haralick 特征) |
| 颜色矩 | 9 | HSV 各通道均值/方差/偏度 |
| 多尺度 LBP | 177 | R=1,2,3 三个半径的 LBP 拼接 |
| Gabor 滤波 | 96 | 6 尺度 x 8 方向 x 2 (均值+标准差) |
| Zernike 矩 | 45 | n=0~8 阶正交矩 |
| Fourier 描述子 | 32 | 频域边界描述 |
| Canny 边缘统计 | 6 | 边缘密度/方向统计 |
| 颜色相关图 | 108 | HSV 三通道颜色对共现 |
| **总计** | **2836** | |

## 模型

### 传统机器学习

| 模型 | Notebook | 核/算法 | 关键参数 |
|------|----------|---------|---------|
| SVM | `crop_pest_svm.ipynb` | RBF 核 + StandardScaler | C=10, gamma=scale |
| LightGBM | `crop_pest_lightgbm.ipynb` | 梯度提升树 | lr=0.05, leaves=31, n_est=500 |

### 深度学习

| 模型 | Notebook | 输入尺寸 | 参数量 |
|------|----------|----------|--------|
| ResNet18 | `gp_resnet18.ipynb` | 256x256 | ~11.7M |
| DenseNet | `gp_densenet.ipynb` | 256x256 | ~7.0M |
| SwinBase | `gp_swinbase.ipynb` | 224x224 | ~88M |

## 结果对比

### 总体性能

| 模型 | 特征维度 | 准确率 | Macro P | Macro R | Macro F1 | Weighted F1 |
|------|----------|--------|---------|---------|----------|-------------|
| SVM (RBF) | 2836 | 56.60% | 0.5774 | 0.5621 | 0.5599 | 0.5592 |
| LightGBM | 2836 | 60.96% | 0.6207 | 0.6163 | 0.6068 | 0.5981 |
| ResNet18 | - | 86.36% | - | - | 0.8680 | 0.8630 |
| DenseNet | - | 88.10% | - | - | 0.8809 | 0.8811 |
| SwinBase | - | 90.28% | - | - | 0.9034 | 0.9036 |

### LightGBM vs SVM 逐类对比

| 类别 | SVM P | SVM R | SVM F1 | LGBM P | LGBM R | LGBM F1 | 胜出 |
|------|-------|-------|--------|--------|--------|---------|------|
| Ants | 0.5870 | 0.6207 | 0.6034 | 0.6395 | 0.6322 | 0.6358 | LGBM |
| Bees | 0.5556 | 0.7955 | 0.6542 | 0.5522 | 0.8409 | 0.6667 | LGBM |
| Beetles | 0.3333 | 0.2955 | 0.3133 | 0.4186 | 0.4091 | 0.4138 | LGBM |
| Caterpillars | 0.4891 | 0.7204 | 0.5826 | 0.4853 | 0.7097 | 0.5764 | SVM |
| Earthworms | 0.4884 | 0.5250 | 0.5060 | 0.6190 | 0.6500 | 0.6341 | LGBM |
| Earwigs | 0.5000 | 0.3151 | 0.3866 | 0.5135 | 0.2603 | 0.3455 | SVM |
| Grasshoppers | 0.4717 | 0.4545 | 0.4630 | 0.5417 | 0.4727 | 0.5049 | LGBM |
| Moths | 0.8095 | 0.7234 | 0.7640 | 0.8333 | 0.7447 | 0.7865 | LGBM |
| Slugs | 0.6071 | 0.3333 | 0.4304 | 0.5556 | 0.2941 | 0.3846 | SVM |
| Snails | 0.7660 | 0.7200 | 0.7423 | 0.7222 | 0.7800 | 0.7500 | LGBM |
| Wasps | 0.5769 | 0.6383 | 0.6061 | 0.7600 | 0.8085 | 0.7835 | LGBM |
| Weevils | 0.7447 | 0.6034 | 0.6667 | 0.8070 | 0.7931 | 0.8000 | LGBM |
| **Macro Avg** | **0.5774** | **0.5621** | **0.5599** | **0.6207** | **0.6163** | **0.6068** | **LGBM** |

LightGBM 在 10/12 类别上胜出，整体领先 SVM 约 4.4 个百分点。

### 全模型横向对比

| 维度 | SVM | LightGBM | ResNet18 | DenseNet | SwinBase |
|------|-----|----------|----------|----------|----------|
| 准确率 | 56.60% | 60.96% | 86.36% | 88.10% | 90.28% |
| 方法类型 | 传统 ML | 传统 ML | 深度学习 | 深度学习 | 深度学习 |
| 特征来源 | 手工 2836 维 | 手工 2836 维 | 自动学习 | 自动学习 | 自动学习 |
| 输入分辨率 | 64x64 | 64x64 | 256x256 | 256x256 | 224x224 |
| 参数量 | - | ~10^6 | ~11.7M | ~7.0M | ~88M |
| 训练时间 | 分钟级 | 分钟级 | 小时级 | 小时级 | 小时级 |
| 推理速度 | 快 | 快 | 中 | 中 | 慢 |
| 可解释性 | 低 | 高 (特征重要性) | 低 | 低 | 低 |
| 硬件需求 | CPU | CPU | GPU | GPU | GPU |
| 最佳类别 | Moths (0.76) | Moths (0.79) | - | - | - |
| 最差类别 | Beetles (0.31) | Beetles (0.41) | - | - | - |

### 模型选择建议

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 快速原型/基线 | LightGBM | CPU 可运行, 训练快, 特征重要性可解释 |
| 资源受限部署 | LightGBM | 无需 GPU, 模型文件小, 推理速度快 |
| 生产级精度 | SwinBase | 90.28% 最高准确率 |
| 精度与效率平衡 | DenseNet | 88.10% 准确率, 参数量仅 7M |
| 特征分析 | LightGBM | 可输出 2836 维特征重要性排序 |

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

编辑 `config.yaml` 调整路径、训练参数和裁剪设置：

```yaml
yolo_root: ./archive
out_root: runs
img_size: 256
epochs: 30
batch: 64
lr: 0.0003
crop_margin_ratio: 0.07
min_crop_size: 10
```

### 运行模型

打开对应的 Jupyter Notebook 并运行所有单元格：

- **传统 ML**: `models/ML/crop_pest_lightgbm.ipynb`
- **深度学习**: `models/{resnet18,densenet,swinbase}/gp_*.ipynb`

## 机器学习准确率低的原因分析

### 对比深度学习

| 方面 | 传统 ML | 深度学习 |
|------|---------|----------|
| 特征提取 | 手工设计（固定） | 自动学习（自适应） |
| 判别力 | 有限（57-61%） | 强（86-90%） |
| 鲁棒性 | 对预处理敏感 | 端到端优化 |
| 泛化能力 | 弱 | 强 |

### 核心问题

1. **手工特征判别力有限**
   - 颜色特征（HSV）：不同害虫颜色可能重叠
   - 纹理特征（LBP/GLCM）：对图像质量敏感
   - 形状特征（Hu 矩）：对姿态变化敏感
   - HOG 特征（1764 维）：占主导，但边缘信息对细粒度分类不足

2. **特征工程问题**
   - 特征冗余：HOG 占 62%，其他特征被淹没
   - 未进行特征选择或降维
   - 特征标准化不充分

3. **数据集限制**
   - 测试集仅 689 样本，统计显著性不足
   - 类别不平衡问题
   - 裁剪质量影响特征提取

4. **模型固有限制**
   - SVM：高维稀疏特征，RBF 核参数敏感
   - 多分类策略：OvR/OvO 效果受限



### 改进建议

详见 [doc/ml_optimization_guide.md](doc/ml_optimization_guide.md)，涵盖：

1. 特征优化（PCA 降维、特征选择、HOG 参数调整）
2. 数据增强（旋转、翻转、颜色抖动）
3. 模型调优（Optuna 超参数搜索、GridSearchCV）
4. 集成方法（Voting、Stacking、多特征子组融合）

## 关键结论

1. 深度学习模型显著优于传统 ML (90% vs 57-61%)
2. LightGBM 在传统 ML 中表现最佳 (60.96%)，比 SVM (56.60%) 高出 4.4 个百分点
3. Swin Transformer 达到最高准确率 90.28%
4. 手工特征在细粒度害虫分类中判别力有限，Beetles 类在两个模型中均为最差
5. LightGBM 提供特征重要性分析能力，支持快速迭代开发和特征工程优化
6. 传统 ML 适合作为基线模型和资源受限场景，但难以达到生产级精度

## 相关开源项目

详见 [doc/open_source_survey.md](doc/open_source_survey.md) 完整调研报告。

### 使用同一数据集的项目

| 项目 | 方法 | 结果 | 链接 |
|------|------|------|------|
| Felipe713/agro-pests | PyTorch + ResNet-18 | 3 epoch 验证准确率 ~65% | [GitHub](https://github.com/Felipe713/agro-pests) |
| jiyak12/Real-Time-Pest-Monitoring-System | OpenCV + Decision Tree / SVM | 面向 IoT 部署 | [GitHub](https://github.com/jiyak12/Real-Time-Pest-Monitoring-System) |

### 使用相似特征提取方法的项目

| 项目 | 领域 | 特征 | 分类器 | 链接 |
|------|------|------|--------|------|
| evitanegara/Weed-Classification | 农业杂草 | HOG + LBP + 颜色直方图 | LR + RF + SVC + Stacking | [GitHub](https://github.com/evitanegara/Weed-Classification-Using-Image-Processing) |
| Manya0407/SVM-based-classification | 通用图像 | HOG + SIFT + 颜色直方图 | SVM (RBF) | [GitHub](https://github.com/Manya0407/SVM-based-image-classification-using-Feature-Descriptors) |

### 大规模害虫识别项目

| 项目 | 数据集 | 方法 | 链接 |
|------|--------|------|------|
| adhiiisetiawan/large-scale-pest-recognition | IP102 (102 类, 75000+ 图像) | CNN + 数据增强 | [GitHub](https://github.com/adhiiisetiawan/large-scale-pest-recognition) |
| mathewGlenn/Corn-Pest-Dataset | 玉米害虫 6 类 | Mobile CNN | [GitHub](https://github.com/mathewGlenn/Corn-Pest-Dataset) |

### 本项目与开源方案的核心差异

- **特征广度**: 本项目 13 种特征 (2836 维)，远超典型开源方案的 2-3 种
- **特征校验**: 内置维度/NaN/Inf/全零检测，开源方案普遍缺失
- **缓存机制**: 维度校验自动失效，支持快速迭代
- **LightGBM 应用**: 在传统特征害虫分类中属于较新颖的尝试

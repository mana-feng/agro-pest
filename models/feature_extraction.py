# ---------------------------------------------------------------------------
# 传统图像特征提取模块
# ---------------------------------------------------------------------------
# 将 SVM 分类器中使用的所有特征提取函数集中到此处,
# 包括: HSV 直方图, Hu 不变矩, LBP, HOG, 轮廓几何特征, GLCM Haralick 特征.
#
# 特征维度汇总:
#   HSV 直方图      -- 512 维  (颜色分布)
#   Hu 不变矩       --   7 维  (粗略形状)
#   LBP 直方图      --  59 维  (微观纹理)
#   HOG             -- 1764 维 (边缘方向)
#   轮廓几何特征    --   8 维  (形状属性)
#   GLCM 纹理特征   --   6 维  (区域纹理)
#   总计            -- 2356 维
# ---------------------------------------------------------------------------

import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


# ===================================================================
# 1. HSV 颜色直方图 (512 维)
# ===================================================================
# 将 BGR 图像转到 HSV 空间, 对 H/S/V 三通道各分 8 个 bin 统计直方图,
# 归一化后展开为 8x8x8=512 维向量.
# HSV 空间相比 RGB 对光照变化更鲁棒, 适合捕捉害虫的颜色分布.

def extract_hsv_hist(roi):
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    hist = cv.calcHist(
        [hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
    )
    hist = cv.normalize(hist, hist).flatten()
    return hist


# ===================================================================
# 2. Hu 不变矩 (7 维)
# ===================================================================
# 计算灰度图的几何矩, 推导出 7 个 Hu 不变矩.
# 对平移/旋转/缩放具有不变性, 描述粗略的形状轮廓.
# 维度低, 判别力有限, 适合作为辅助特征.

def extract_hu_moments(roi):
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    hu = cv.HuMoments(cv.moments(gray)).flatten()
    return hu


# ===================================================================
# 3. LBP 局部二值模式 (59 维)
# ===================================================================
# 对灰度图每个像素取 P=8 个邻域 (半径 R=1), 用 uniform 模式编码.
# uniform 模式下跳变次数 <= 2 的有 58 种, 加 1 种非均匀模式, 共 59 bin.
# 统计整张图的 LBP 编码直方图并归一化.
# 捕捉微观纹理特征 (如体表粗糙度).

def extract_lbp(gray, P=8, R=1):
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    n_bins = P * (P - 1) + 3
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


# ===================================================================
# 4. HOG 方向梯度直方图 (1764 维)
# ===================================================================
# 计算每个像素的梯度方向和幅值, 在 8x8 的 cell 内统计 9 个方向的直方图,
# 每 2x2 个 cell (16x16 block) 做归一化, block 步长 8.
# 64x64 图像: 7x7=49 个 block, 每个 block 36 维, 共 1764 维.
# 捕捉边缘方向和轮廓梯度信息.

def create_hog_descriptor(win_size=(64, 64)):
    return cv.HOGDescriptor(
        _winSize=win_size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def extract_hog(roi, hog_desc):
    return hog_desc.compute(roi).flatten()


# ===================================================================
# 5. 轮廓几何特征 (8 维)
# ===================================================================
# 从二值化轮廓中提取形状属性:
#   [0] 面积 / 外接矩形面积 (矩形度, extent)
#   [1] 面积 / 凸包面积     (凸度, solidity)
#   [2] 外接矩形宽 / 高     (长宽比, aspect_ratio)
#   [3] 4 * pi * 面积 / 周长^2 (圆度, circularity)
#   [4] 凸包周长 / 轮廓周长 (凸包周长比)
#   [5] 等效直径             (与面积相同的圆的直径)
#   [6] 轮廓面积 (归一化)
#   [7] 轮廓周长 (归一化)
# 对害虫分类很有效: 毛虫细长, 蜗牛圆润, 甲虫椭圆.

def extract_contour_geometry(roi):
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(8)

    contour = max(contours, key=cv.contourArea)
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)

    if area < 1e-3 or perimeter < 1e-3:
        return np.zeros(8)

    x, y, w, h = cv.boundingRect(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    hull_perimeter = cv.arcLength(hull, True)

    rect_area = max(w * h, 1e-7)
    hull_area_safe = max(hull_area, 1e-7)
    h_safe = max(h, 1e-7)
    perimeter_safe = max(perimeter, 1e-7)

    extent = area / rect_area
    solidity = area / hull_area_safe
    aspect_ratio = w / h_safe
    circularity = 4.0 * np.pi * area / (perimeter_safe ** 2)
    hull_perim_ratio = hull_perimeter / perimeter_safe
    equiv_diameter = np.sqrt(4.0 * area / np.pi)
    norm_area = area / (roi.shape[0] * roi.shape[1])
    norm_perimeter = perimeter_safe / (2 * (roi.shape[0] + roi.shape[1]))

    return np.array([
        extent, solidity, aspect_ratio, circularity,
        hull_perim_ratio, equiv_diameter, norm_area, norm_perimeter,
    ])


# ===================================================================
# 6. GLCM 纹理特征 (6 维)
# ===================================================================
# 计算灰度共生矩阵 (4 个方向 0/45/90/135 度的平均),
# 提取 skimage 支持的 6 个统计量:
#   ASM (能量), contrast (对比度), dissimilarity (差异性),
#   homogeneity (同质性), energy (能量), correlation (相关性)
# 与 LBP 互补: LBP 看局部像素级关系, GLCM 看区域灰度共生关系.

def extract_glcm_haralick(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    gray_uint8 = (gray / gray.max() * 63).astype(np.uint8) if gray.max() > 0 else gray.astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)

    properties = ['ASM', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    features = []
    for prop in properties:
        val = graycoprops(glcm, prop).mean()
        features.append(val)

    return np.array(features)


# ===================================================================
# 7. 综合特征提取
# ===================================================================
# 将上述所有特征拼接为一个向量, 作为 SVM 的输入.
# 当前维度: 512 + 7 + 59 + 1764 + 8 + 6 = 2356

def extract_features(roi, hog_desc):
    hist = extract_hsv_hist(roi)
    hu = extract_hu_moments(roi)
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    lbp_feat = extract_lbp(gray)
    hog_feat = extract_hog(roi, hog_desc)
    contour_feat = extract_contour_geometry(roi)
    glcm_feat = extract_glcm_haralick(gray)
    return np.hstack([hist, hu, lbp_feat, hog_feat, contour_feat, glcm_feat])

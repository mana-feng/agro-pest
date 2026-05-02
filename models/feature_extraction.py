# ---------------------------------------------------------------------------
# 传统图像特征提取模块
# ---------------------------------------------------------------------------
# 将 SVM 分类器中使用的所有特征提取函数集中到此处,
# 包括: HSV 直方图, Hu 不变矩, LBP, HOG, 轮廓几何特征, GLCM Haralick 特征,
# 以及新增: 颜色矩, 多尺度 LBP, Gabor 滤波, Zernike 矩, Fourier 描述子,
#          Canny 边缘统计, 颜色相关图, Bbox 元数据, 背景特征.
#
# 特征维度汇总:
#   HSV 直方图      --  512 维  (颜色分布)
#   Hu 不变矩       --    7 维  (粗略形状)
#   LBP 直方图      --   59 维  (微观纹理)
#   HOG             -- 1764 维 (边缘方向)
#   轮廓几何特征    --    8 维  (形状属性)
#   GLCM 纹理特征   --   13 维  (区域纹理, 完整 Haralick)
#   颜色矩          --    9 维  (HSV 各通道均值/方差/偏度)
#   多尺度 LBP      --  177 维  (R=1,2,3 三个半径)
#   Gabor 滤波      --   96 维  (6 尺度 x 8 方向 x 2 (均值+标准差))
#   Zernike 矩      --   45 维  (n=0~8 阶正交矩)
#   Fourier 描述子  --   32 维  (频域边界描述)
#   Canny 边缘统计  --    6 维  (边缘密度/方向)
#   颜色相关图      --  108 维  (HSV 三通道颜色对共现)
#   Bbox 元数据     --    5 维  (面积/位置/宽高比/目标数)
#   背景特征        --   22 维  (背景颜色/对比度/背景纹理)
#   总计            -- 2836 维 (不含 Bbox 元数据和背景特征)
# ---------------------------------------------------------------------------

import cv2 as cv
import logging
import math
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import regionprops, label
from scipy.stats import skew

logger = logging.getLogger(__name__)

EXPECTED_DIMS = {
    "HSV 直方图": 512,
    "Hu 不变矩": 7,
    "LBP": 59,
    "HOG": 1764,
    "轮廓几何": 8,
    "GLCM Haralick": 13,
    "颜色矩": 9,
    "多尺度 LBP": 177,
    "Gabor": 96,
    "Zernike": 45,
    "Fourier": 32,
    "Canny": 6,
    "颜色相关图": 108,
    "Bbox 元数据": 5,
    "背景特征": 22,
}

EXTRACTED_DIMS = {k: EXPECTED_DIMS[k] for k in [
    "HSV 直方图", "Hu 不变矩", "LBP", "HOG", "轮廓几何",
    "GLCM Haralick", "颜色矩", "多尺度 LBP", "Gabor",
    "Zernike", "Fourier", "Canny", "颜色相关图",
]}

FEATURE_DIM = sum(EXTRACTED_DIMS.values())


def _validate_feature(name, feat, expected_dim=None):
    problems = []
    if feat is None:
        problems.append("返回值为 None")
        logger.warning("[特征校验] %s: %s", name, "; ".join(problems))
        return problems
    if not isinstance(feat, np.ndarray):
        problems.append(f"类型错误: 期望 np.ndarray, 实际 {type(feat).__name__}")
    else:
        if feat.ndim != 1:
            problems.append(f"维度错误: 期望 1D, 实际 {feat.ndim}D (shape={feat.shape})")
        if expected_dim is not None and feat.shape[0] != expected_dim:
            problems.append(f"长度错误: 期望 {expected_dim}, 实际 {feat.shape[0]}")
        nan_count = np.isnan(feat).sum()
        if nan_count > 0:
            problems.append(f"含 {nan_count} 个 NaN")
        inf_count = np.isinf(feat).sum()
        if inf_count > 0:
            problems.append(f"含 {inf_count} 个 Inf")
        all_zero = np.all(feat == 0)
        if all_zero:
            problems.append("全零向量 (可能提取失败)")
    if problems:
        logger.warning("[特征校验] %s: %s", name, "; ".join(problems))
    else:
        logger.debug("[特征校验] %s: OK, dim=%d", name, feat.shape[0])
    return problems


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
    _validate_feature("HSV 直方图", hist, EXPECTED_DIMS["HSV 直方图"])
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
    _validate_feature("Hu 不变矩", hu, EXPECTED_DIMS["Hu 不变矩"])
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
    _validate_feature("LBP", hist, EXPECTED_DIMS["LBP"])
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
    feat = hog_desc.compute(roi).flatten()
    _validate_feature("HOG", feat, EXPECTED_DIMS["HOG"])
    return feat


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
        logger.debug("[轮廓几何] 未检测到轮廓, 返回全零向量")
        feat = np.zeros(8)
        _validate_feature("轮廓几何", feat, EXPECTED_DIMS["轮廓几何"])
        return feat

    contour = max(contours, key=cv.contourArea)
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)

    if area < 1e-3 or perimeter < 1e-3:
        logger.debug("[轮廓几何] 面积或周长过小 (area=%.4f, perimeter=%.4f), 返回全零向量", area, perimeter)
        feat = np.zeros(8)
        _validate_feature("轮廓几何", feat, EXPECTED_DIMS["轮廓几何"])
        return feat

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

    feat = np.array([
        extent, solidity, aspect_ratio, circularity,
        hull_perim_ratio, equiv_diameter, norm_area, norm_perimeter,
    ])
    _validate_feature("轮廓几何", feat, EXPECTED_DIMS["轮廓几何"])
    return feat


# ===================================================================
# 6. GLCM 纹理特征 (13 维) -- 完整 Haralick 特征
# ===================================================================
# 计算灰度共生矩阵 (4 个方向 0/45/90/135 度的平均),
# 提取 skimage 支持的 6 个统计量 + 7 个额外 Haralick 特征:
#   基础: ASM, contrast, dissimilarity, homogeneity, energy, correlation
#   额外: sum_average, sum_variance, sum_entropy, diff_variance,
#         diff_entropy, imc1, imc2
# 与 LBP 互补: LBP 看局部像素级关系, GLCM 看区域灰度共生关系.

def extract_glcm_haralick(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    if gray.max() == 0:
        logger.debug("[GLCM] 输入灰度图全零, 返回全零向量")
        feat = np.zeros(13)
        _validate_feature("GLCM Haralick", feat, EXPECTED_DIMS["GLCM Haralick"])
        return feat

    gray_uint8 = (gray / gray.max() * 63).astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)

    properties = ['ASM', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    features = []
    for prop in properties:
        val = graycoprops(glcm, prop).mean()
        if np.isnan(val) or np.isinf(val):
            logger.warning("[GLCM] skimage 属性 %s 异常值: %.6f", prop, val)
        features.append(val)

    glcm_arr = glcm.mean(axis=(2, 3))
    p_ij = glcm_arr
    p_i = p_ij.sum(axis=1)
    p_j = p_ij.sum(axis=0)

    sum_avg = np.sum((np.arange(64)[:, None] + np.arange(64)[None, :]) * p_ij)
    sum_var = np.sum(((np.arange(64)[:, None] + np.arange(64)[None, :]) - sum_avg) ** 2 * p_ij)
    sum_entropy = -np.sum(p_ij * np.log(p_ij + 1e-10))

    diff_vals = np.abs(np.arange(64)[:, None] - np.arange(64)[None, :])
    diff_var = np.sum((diff_vals - np.sum(diff_vals * p_ij)) ** 2 * p_ij)
    p_diff = np.zeros(64)
    for k in range(64):
        p_diff[k] = np.sum(p_ij[diff_vals == k])
    diff_entropy = -np.sum(p_diff * np.log(p_diff + 1e-10))

    hx = -np.sum(p_i * np.log(p_i + 1e-10))
    hy = -np.sum(p_j * np.log(p_j + 1e-10))
    p_marginal = p_i[:, None] * p_j[None, :]
    valid_mask = (p_ij > 0) & (p_marginal > 0)
    q1 = np.sum(p_ij[valid_mask] * np.log(p_ij[valid_mask] / p_marginal[valid_mask]))
    q2 = np.sum((p_ij - p_marginal) ** 2 / (p_marginal + 1e-10))
    imc1 = q1 / max(hx, hy) if max(hx, hy) > 0 else 0
    imc2_val = 1 - np.exp(-2 * q2)
    imc2 = np.sqrt(max(imc2_val, 0))

    features.extend([sum_avg, sum_var, sum_entropy, diff_var, diff_entropy, imc1, imc2])
    feat = np.array(features)
    nan_mask = np.isnan(feat) | np.isinf(feat)
    if nan_mask.any():
        logger.warning("[GLCM] 特征含 NaN/Inf, 已替换为 0: %s", np.where(nan_mask)[0].tolist())
        feat[nan_mask] = 0
    _validate_feature("GLCM Haralick", feat, EXPECTED_DIMS["GLCM Haralick"])
    return feat


# ===================================================================
# 7. 颜色矩 (9 维)
# ===================================================================
# 对 HSV 三通道分别计算一阶矩 (均值), 二阶矩 (标准差), 三阶矩 (偏度).
# 相比直方图更紧凑 (9 维 vs 512 维), 对光照变化鲁棒.
# 一阶矩描述平均颜色, 二阶矩描述颜色分布范围, 三阶矩描述分布不对称性.

def extract_color_moments(roi):
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV).astype(np.float32)
    moments = []
    for ch in range(3):
        channel = hsv[:, :, ch].flatten()
        moments.append(np.mean(channel))
        std_val = np.std(channel)
        moments.append(std_val)
        if std_val > 0:
            moments.append(skew(channel))
        else:
            moments.append(0.0)
    feat = np.array(moments)
    _validate_feature("颜色矩", feat, EXPECTED_DIMS["颜色矩"])
    return feat


# ===================================================================
# 8. 多尺度 LBP (177 维)
# ===================================================================
# 在 R=1, 2, 3 三个半径下分别计算 uniform LBP 直方图 (各 59 维),
# 拼接得到 177 维特征.
# 不同半径捕捉不同粒度的纹理: R=1 看微观细节, R=3 看宏观纹理模式.

def extract_multiscale_lbp(gray, P=8, radii=[1, 2, 3]):
    features = []
    for r in radii:
        lbp = local_binary_pattern(gray, P=P, R=r, method="uniform")
        n_bins = P * (P - 1) + 3
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features.append(hist)
    feat = np.hstack(features)
    _validate_feature("多尺度 LBP", feat, EXPECTED_DIMS["多尺度 LBP"])
    return feat


# ===================================================================
# 9. Gabor 滤波响应 (96 维)
# ===================================================================
# 使用 6 个尺度 (频率) x 8 个方向 的 Gabor 滤波器组对灰度图进行卷积,
# 对每个滤波响应图计算均值和标准差, 得到 6x8x2=96 维特征.
# Gabor 滤波器对特定方向和频率的纹理敏感, 对甲虫鞘翅纹路等周期性纹理有效.

def extract_gabor_features(gray, num_scales=6, num_orientations=8):
    features = []
    for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
        for freq_idx in range(1, num_scales + 1):
            freq = 0.1 + 0.1 * freq_idx
            kernel = cv.getGaborKernel(
                ksize=(11, 11), sigma=2.0, theta=theta,
                lambd=1.0 / freq, gamma=0.5, psi=0,
            )
            filtered = cv.filter2D(gray, cv.CV_32F, kernel)
            features.append(np.mean(filtered))
            features.append(np.std(filtered))
    feat = np.array(features)
    _validate_feature("Gabor", feat, EXPECTED_DIMS["Gabor"])
    return feat


# ===================================================================
# 10. Zernike 矩 (45 维)
# ===================================================================
# 在单位圆内计算 Zernike 正交矩, 选取 n=0~8 阶 (共 45 个系数).
# 相比 Hu 矩, Zernike 矩具有更好的正交性和重建能力, 形状描述力更强.
# 对区分不同形状的害虫 (细长毛虫 vs 圆润蜗牛) 很有帮助.

def extract_zernike_moments(roi, max_order=8):
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    h, w = binary.shape
    center = (w / 2.0, h / 2.0)
    radius = min(w, h) / 2.0

    if radius < 1e-3:
        logger.debug("[Zernike] 图像尺寸过小 (%dx%d), 返回全零向量", w, h)
        feat = np.zeros(45)
        _validate_feature("Zernike", feat, EXPECTED_DIMS["Zernike"])
        return feat

    y, x = np.mgrid[0:h, 0:w]
    x_norm = (x - center[0]) / radius
    y_norm = (y - center[1]) / radius
    mask = (x_norm ** 2 + y_norm ** 2) <= 1.0

    binary_f = binary.astype(np.float64)
    moments = []
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            rho = np.sqrt(x_norm ** 2 + y_norm ** 2)
            theta = np.arctan2(y_norm, x_norm)
            radial = _zernike_radial(n, abs(m), rho)
            z_real = radial * np.cos(m * theta)
            z_imag = radial * np.sin(m * theta)
            zr = np.sum(binary_f * z_real * mask)
            zi = np.sum(binary_f * z_imag * mask)
            moments.append(np.sqrt(zr ** 2 + zi ** 2))
    feat = np.array(moments)
    _validate_feature("Zernike", feat, EXPECTED_DIMS["Zernike"])
    return feat


def _zernike_radial(n, m, rho):
    radial = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        coef = ((-1) ** s) * _nCr(n - s, s) * _nCr(n - 2 * s, (n - m) // 2 - s)
        radial += coef * rho ** (n - 2 * s)
    return radial


def _nCr(n, r):
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)


# ===================================================================
# 11. Fourier 描述子 (32 维)
# ===================================================================
# 对最大轮廓的边界点进行参数化, 做一维傅里叶变换,
# 取前 16 个低频系数的幅值 (归一化后), 得到 32 维实数特征 (实部+虚部).
# 低频系数描述轮廓的全局形状, 对噪声鲁棒.

def extract_fourier_descriptors(roi, num_coeffs=16):
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.debug("[Fourier] 未检测到轮廓, 返回全零向量")
        feat = np.zeros(num_coeffs * 2)
        _validate_feature("Fourier", feat, EXPECTED_DIMS["Fourier"])
        return feat

    contour = max(contours, key=cv.contourArea).squeeze()
    if len(contour) < 4:
        logger.debug("[Fourier] 轮廓点数不足 (%d < 4), 返回全零向量", len(contour))
        feat = np.zeros(num_coeffs * 2)
        _validate_feature("Fourier", feat, EXPECTED_DIMS["Fourier"])
        return feat

    complex_coords = contour[:, 0].astype(np.float64) + 1j * contour[:, 1].astype(np.float64)
    if len(complex_coords) <= num_coeffs:
        pad_len = num_coeffs + 1 - len(complex_coords)
        complex_coords = np.pad(complex_coords, (0, pad_len), mode='constant')
        logger.debug("[Fourier] 轮廓点数不足, 零填充至 %d", len(complex_coords))

    fft_coeffs = np.fft.fft(complex_coords)
    fft_coeffs = fft_coeffs[1:num_coeffs + 1]
    if np.abs(fft_coeffs[0]) > 0:
        fft_coeffs /= np.abs(fft_coeffs[0])
    else:
        logger.warning("[Fourier] 第一个 FFT 系数为零, 无法归一化")
    feat = np.hstack([fft_coeffs.real, fft_coeffs.imag])
    _validate_feature("Fourier", feat, EXPECTED_DIMS["Fourier"])
    return feat


# ===================================================================
# 12. Canny 边缘统计 (6 维)
# ===================================================================
# 对灰度图做 Canny 边缘检测, 提取:
#   [0] 边缘像素占比 (边缘密度)
#   [1] 边缘连通分量数
#   [2] 最大边缘连通分量面积占比
#   [3] 水平方向边缘占比
#   [4] 垂直方向边缘占比
#   [5] 平均边缘强度
# 简单但有效, 对区分不同边缘模式的害虫有帮助.

def extract_canny_edge_stats(roi):
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    # 自适应 Canny 阈值: 基于 Sobel 梯度中位数动态计算高低阈值
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    median_grad = np.median(grad_mag)
    low_thresh = int(max(0, 0.33 * median_grad))
    high_thresh = int(min(255, 0.66 * median_grad))
    edges = cv.Canny(gray, low_thresh, high_thresh)

    h, w = edges.shape
    total_pixels = h * w

    edge_pixels = np.sum(edges > 0)
    edge_density = edge_pixels / total_pixels

    labeled, num_components = label(edges > 0, return_num=True)
    if num_components > 0:
        regions = regionprops(labeled)
        max_region_area = max(r.area for r in regions)
        max_region_ratio = max_region_area / total_pixels
    else:
        max_region_ratio = 0

    edge_mask = edges > 0
    if edge_pixels > 0:
        gx = np.abs(sobelx[edge_mask])
        gy = np.abs(sobely[edge_mask])
        total_grad = gx.sum() + gy.sum() + 1e-10
        h_ratio = gx.sum() / total_grad
        v_ratio = gy.sum() / total_grad
        avg_strength = (gx.mean() + gy.mean()) / 2
    else:
        logger.debug("[Canny] 未检测到边缘像素 (median=%.2f, low=%d, high=%d)", median_grad, low_thresh, high_thresh)
        h_ratio = v_ratio = avg_strength = 0

    feat = np.array([edge_density, num_components, max_region_ratio, h_ratio, v_ratio, avg_strength])
    _validate_feature("Canny", feat, EXPECTED_DIMS["Canny"])
    return feat


# ===================================================================
# 13. 颜色相关图 (108 维)
# ===================================================================
# 将 HSV 三通道各量化到 6 个颜色级别, 分别计算距离为 1 像素的颜色对共现概率.
# 每通道 6x6=36 维, 三通道共 108 维.
# 相比直方图, 颜色相关图保留了颜色之间的空间位置关系.

def extract_color_correlogram(roi, quantize_bins=6, distance=1):
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    correlogram = []

    for ch in range(3):
        ch_max = 180 if ch == 0 else 256
        quantized = (hsv[:, :, ch].astype(np.int32) // (ch_max // quantize_bins)).clip(0, quantize_bins - 1)
        ch_corr = np.zeros((quantize_bins, quantize_bins), dtype=np.float64)

        for di in [-distance, 0, distance]:
            for dj in [-distance, 0, distance]:
                if di == 0 and dj == 0:
                    continue
                h_shift = h - abs(di)
                w_shift = w - abs(dj)
                if h_shift <= 0 or w_shift <= 0:
                    continue
                if di >= 0:
                    src_i, dst_i = slice(0, h_shift), slice(di, di + h_shift)
                else:
                    src_i, dst_i = slice(-di, -di + h_shift), slice(0, h_shift)
                if dj >= 0:
                    src_j, dst_j = slice(0, w_shift), slice(dj, dj + w_shift)
                else:
                    src_j, dst_j = slice(-dj, -dj + w_shift), slice(0, w_shift)
                c1 = quantized[src_i, src_j]
                c2 = quantized[dst_i, dst_j]
                for ci in range(quantize_bins):
                    mask = c1 == ci
                    if mask.sum() == 0:
                        continue
                    for cj in range(quantize_bins):
                        ch_corr[ci, cj] += np.sum(c2[mask] == cj)

        total = ch_corr.sum() + 1e-10
        ch_corr /= total
        correlogram.append(ch_corr.flatten())

    feat = np.hstack(correlogram)
    _validate_feature("颜色相关图", feat, EXPECTED_DIMS["颜色相关图"])
    return feat


# ===================================================================
# 14. Bbox 元数据特征 (5 维)
# ===================================================================
# 从 YOLO 标注的边界框中提取元数据:
#   [0] bbox 相对面积 (占整图比例)
#   [1] bbox 中心 x 坐标 (归一化)
#   [2] bbox 中心 y 坐标 (归一化)
#   [3] bbox 宽高比
#   [4] 单图目标数 (同一张图中有多少个标注框)
# 不同害虫体型差异大 (蚯蚓 vs 蚂蚁), 出现位置也有规律.

def extract_bbox_metadata(bbox_info, img_w, img_h, num_boxes):
    cx, cy, bw, bh = bbox_info
    area_ratio = (bw * bh) / (img_w * img_h + 1e-10)
    aspect = bw / (bh + 1e-10)
    feat = np.array([area_ratio, cx, cy, aspect, num_boxes])
    _validate_feature("Bbox 元数据", feat, EXPECTED_DIMS["Bbox 元数据"])
    return feat


# ===================================================================
# 15. 背景特征 (22 维)
# ===================================================================
# 利用 bbox 外的背景区域提取特征:
#   [0-2] 背景 HSV 均值 (3 维)
#   [3-5] 背景 HSV 标准差 (3 维)
#   [6-8] 前景-背景颜色差值 (3 维, 对比度)
#   [9-21] 背景 GLCM 纹理 (13 维, 完整 Haralick)
# 区分土壤上的害虫 vs 叶片上的害虫, 以及伪装性强的害虫.

def extract_background_features(full_img, bbox_xyxy):
    h, w = full_img.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    bg_pixels = full_img[mask == 0]

    if len(bg_pixels) < 10:
        logger.debug("[背景特征] 背景像素不足 (%d < 10), 返回全零向量", len(bg_pixels))
        feat = np.zeros(22)
        _validate_feature("背景特征", feat, EXPECTED_DIMS["背景特征"])
        return feat

    bg_hsv = cv.cvtColor(full_img, cv.COLOR_BGR2HSV)
    bg_hsv_pixels = bg_hsv[mask == 0]

    bg_mean = np.mean(bg_hsv_pixels, axis=0)
    bg_std = np.std(bg_hsv_pixels, axis=0)

    roi_hsv = cv.cvtColor(full_img[y1:y2, x1:x2], cv.COLOR_BGR2HSV)
    roi_mean = np.mean(roi_hsv, axis=(0, 1))
    contrast = np.abs(bg_mean - roi_mean)

    bg_gray = cv.cvtColor(full_img, cv.COLOR_BGR2GRAY)
    bg_gray[mask == 255] = 0
    bg_y1 = max(0, y1 - 64)
    bg_x1 = max(0, x1 - 64)
    bg_gray_roi_cropped = bg_gray[bg_y1:bg_y1 + 64, bg_x1:bg_x1 + 64]
    if bg_gray_roi_cropped.size > 0 and bg_gray_roi_cropped.max() > 0:
        bg_glcm = extract_glcm_haralick(bg_gray_roi_cropped)
    else:
        logger.debug("[背景特征] 背景灰度区域全零, GLCM 返回全零向量")
        bg_glcm = np.zeros(13)

    feat = np.hstack([bg_mean, bg_std, contrast[:3], bg_glcm])
    _validate_feature("背景特征", feat, EXPECTED_DIMS["背景特征"])
    return feat


# ===================================================================
# 16. 综合特征提取
# ===================================================================
# 将上述所有特征拼接为一个向量, 作为分类器的输入.
# 当前维度: 512 + 7 + 59 + 1764 + 8 + 13 + 9 + 177 + 96 + 45 + 32 + 6 + 108 = 2836
# (不含 Bbox 元数据和背景特征, 这两类需要额外传入参数)

def extract_features(roi, hog_desc):
    feat_parts = {}

    feat_parts["HSV 直方图"] = extract_hsv_hist(roi)
    feat_parts["Hu 不变矩"] = extract_hu_moments(roi)
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    feat_parts["LBP"] = extract_lbp(gray)
    feat_parts["HOG"] = extract_hog(roi, hog_desc)
    feat_parts["轮廓几何"] = extract_contour_geometry(roi)
    feat_parts["GLCM Haralick"] = extract_glcm_haralick(gray)
    feat_parts["颜色矩"] = extract_color_moments(roi)
    feat_parts["多尺度 LBP"] = extract_multiscale_lbp(gray)
    feat_parts["Gabor"] = extract_gabor_features(gray)
    feat_parts["Zernike"] = extract_zernike_moments(roi)
    feat_parts["Fourier"] = extract_fourier_descriptors(roi)
    feat_parts["Canny"] = extract_canny_edge_stats(roi)
    feat_parts["颜色相关图"] = extract_color_correlogram(roi)

    fixed_parts = []
    for name, feat in feat_parts.items():
        expected = EXPECTED_DIMS[name]
        if feat is None or not isinstance(feat, np.ndarray) or feat.ndim != 1:
            logger.error("[维度修复] %s: 无效特征, 替换为全零 %d 维", name, expected)
            fixed_parts.append(np.zeros(expected))
            continue
        actual = feat.shape[0]
        if actual == expected:
            fixed_parts.append(feat)
        elif actual < expected:
            logger.warning("[维度修复] %s: %d < %d, 右侧补零", name, actual, expected)
            fixed_parts.append(np.pad(feat, (0, expected - actual), mode='constant'))
        else:
            logger.warning("[维度修复] %s: %d > %d, 截断", name, actual, expected)
            fixed_parts.append(feat[:expected])

    result = np.hstack(fixed_parts)
    expected_total = sum(EXPECTED_DIMS[k] for k in feat_parts)

    nan_count = np.isnan(result).sum()
    inf_count = np.isinf(result).sum()
    if nan_count > 0 or inf_count > 0:
        logger.warning("[综合特征] NaN=%d, Inf=%d, 替换为 0", nan_count, inf_count)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    zero_count = np.sum(result == 0)
    logger.debug(
        "[综合特征] 总维度=%d (期望 %d), NaN=%d, Inf=%d, 零值=%d (%.1f%%)",
        result.shape[0], expected_total, nan_count, inf_count, zero_count,
        100.0 * zero_count / max(result.shape[0], 1),
    )

    return result

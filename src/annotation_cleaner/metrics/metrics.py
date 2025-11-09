#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metrics.py
-----------
이미지 복원 품질을 정량적으로 평가하기 위한 메트릭 함수 모음.

지원 메트릭:
- L1 Distance
- SSIM (Structural Similarity)
- Edge IoU (Canny 기반)
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


# ============================================================
# 🔹 Metric Functions
# ============================================================

def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    """픽셀 단위 L1 거리 (절대 오차 평균)"""
    if a.shape != b.shape:
        raise ValueError(f"L1 Error: 이미지 크기 불일치 {a.shape} vs {b.shape}")
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM (Structural Similarity Index)"""
    if a.shape != b.shape:
        raise ValueError(f"SSIM Error: 이미지 크기 불일치 {a.shape} vs {b.shape}")
    g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    data_range = float(g1.max() - g1.min()) or 255.0
    return ssim(g1, g2, data_range=data_range)


def edge_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Canny 엣지를 이용한 Edge IoU (경계 일치율)"""
    if a.shape != b.shape:
        raise ValueError(f"Edge IoU Error: 이미지 크기 불일치 {a.shape} vs {b.shape}")
    g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    e1, e2 = cv2.Canny(g1, 100, 200), cv2.Canny(g2, 100, 200)
    inter = np.logical_and(e1 > 0, e2 > 0).sum()
    union = np.logical_or(e1 > 0, e2 > 0).sum()
    return float(inter) / union if union > 0 else 0.0


# ============================================================
# 🔹 Wrapper
# ============================================================

def compute_all_metrics(img1: np.ndarray, img2: np.ndarray) -> dict:
    """모든 품질 지표를 한 번에 계산"""
    return {
        "L1": l1_distance(img1, img2),
        "SSIM": ssim_score(img1, img2),
        "Edge_IoU": edge_iou(img1, img2),
    }

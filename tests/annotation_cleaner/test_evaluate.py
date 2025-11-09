#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_evaluator.py
-----------------
Evaluator 클래스 단위 테스트

테스트 목표:
- config 주입 기반 초기화 검증
- metric 계산 로직 검증 (mock image)
- Full Image 평가 정상 수행 여부
- YOLO Crop 평가(mock YOLO 예측) 정상 수행 여부
"""

import os
import cv2
import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from unittest.mock import patch, MagicMock

from src.annotation_cleaner.evaluate import Evaluator


# ============================================================
# 🧱 테스트용 헬퍼 함수
# ============================================================
def create_dummy_image(path: Path, color=(128, 128, 128), size=(64, 64)):
    """단색 더미 이미지 생성"""
    img = np.full((*size, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


# ============================================================
# 🧩 PyTest Fixture
# ============================================================
@pytest.fixture
def temp_dataset(tmp_path):
    """
    더미 원본/생성 이미지 구조를 만드는 fixture.
    예시 구조:
    orig/
        repair/img1.jpg
        replace/img1.jpg
    gen/
        repair/img1.jpg
        replace/img1.jpg
    """
    orig_dir = tmp_path / "orig"
    gen_dir = tmp_path / "gen"
    metric_dir = tmp_path / "metrics"

    for root in [orig_dir, gen_dir]:
        for cat in ["repair", "replace"]:
            d = root / cat
            d.mkdir(parents=True, exist_ok=True)
            create_dummy_image(d / "img1.jpg", color=(100, 100, 100))
            create_dummy_image(d / "img2.jpg", color=(150, 150, 150))

    return {
        "orig_dir": orig_dir,
        "gen_dir": gen_dir,
        "metric_dir": metric_dir,
    }


# ============================================================
# 🧪 테스트 1: Evaluator 초기화
# ============================================================
def test_evaluator_initialization(temp_dataset):
    cfg = {
        "orig_dir": str(temp_dataset["orig_dir"]),
        "gen_dir": str(temp_dataset["gen_dir"]),
        "metric_dir": str(temp_dataset["metric_dir"]),
        "metrics": ["ssim", "l1", "edge_iou"],
        "yolo_model": "./dummy_yolo.pt",
        "imgsz": 416,
    }

    evaluator = Evaluator(**cfg)
    assert evaluator.orig_dir.exists()
    assert evaluator.gen_dir.exists()
    assert "ssim" in evaluator.metrics


# ============================================================
# 🧪 테스트 2: _compute_metrics 동작 검증
# ============================================================
def test_compute_metrics_returns_values(temp_dataset):
    evaluator = Evaluator(
        orig_dir=temp_dataset["orig_dir"],
        gen_dir=temp_dataset["gen_dir"],
        metric_dir=temp_dataset["metric_dir"],
        metrics=["ssim", "l1", "edge_iou"],
        yolo_model="./dummy.pt",
        imgsz=416,
    )

    img = np.full((32, 32, 3), 127, dtype=np.uint8)
    result = evaluator._compute_metrics(img, img)
    assert isinstance(result, dict)
    assert all(k in result for k in ["SSIM", "L1", "Edge_IoU"])
    assert all(isinstance(v, float) for v in result.values())


# ============================================================
# 🧪 테스트 3: Full Image 평가 (metrics.csv 생성 여부)
# ============================================================
def test_evaluate_full_images_creates_csv(temp_dataset):
    evaluator = Evaluator(
        orig_dir=temp_dataset["orig_dir"],
        gen_dir=temp_dataset["gen_dir"],
        metric_dir=temp_dataset["metric_dir"],
        metrics=["ssim", "l1"],
        yolo_model="./dummy.pt",
        imgsz=416,
    )

    save_path = temp_dataset["metric_dir"] / "metrics_full_image.csv"
    avg = evaluator.evaluate_full_images(save_path)

    assert save_path.exists(), "CSV 파일이 생성되지 않았습니다."
    assert isinstance(avg, dict)
    df = pd.read_csv(save_path)
    assert not df.empty


# ============================================================
# 🧪 테스트 4: YOLO Crop 평가 (Mocking 기반)
# ============================================================
@patch("src.annotation_cleaner.evaluate.YOLO")
def test_evaluate_with_yolo_crop_uses_tempdir(mock_yolo, temp_dataset):
    """YOLO 모델을 mock 처리하여 임시 폴더가 잘 동작하는지 검증"""
    # YOLO mock 설정
    mock_pred = MagicMock()
    mock_pred.boxes.xyxy = torch.tensor([[0, 0, 16, 16]])
    mock_yolo.return_value.predict.return_value = [mock_pred]

    evaluator = Evaluator(
        orig_dir=temp_dataset["orig_dir"],
        gen_dir=temp_dataset["gen_dir"],
        metric_dir=temp_dataset["metric_dir"],
        metrics=["ssim", "l1"],
        yolo_model="./dummy.pt",
        imgsz=416,
    )

    save_path = temp_dataset["metric_dir"] / "metrics_yolo_crop.csv"

    # 임시 디렉토리 내에서 YOLO Crop 평가 실행
    avg = evaluator.evaluate_with_yolo_crop(save_path)
    assert isinstance(avg, dict)
    assert save_path.exists()

    # 실제 gen_dir 아래에는 crop/bbox 폴더가 생기지 않아야 함
    assert not (evaluator.gen_dir / "crops").exists()
    assert not (evaluator.gen_dir / "bboxes").exists()

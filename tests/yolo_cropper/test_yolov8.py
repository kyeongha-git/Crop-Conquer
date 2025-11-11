#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_yolov8.py
-----------------------------
Lightweight smoke test for YOLOv8Pipeline (Config-driven)

✅ 목적:
- YOLOv8Pipeline이 내부 단계 주석 여부와 무관하게 정상 실행되는지 확인
- 실제 학습, 평가, 예측 등은 mock 처리 (실제 실행 없음)
"""

import pytest
from pathlib import Path
from unittest.mock import patch
from src.yolo_cropper.models.yolov8.yolov8 import YOLOv8Pipeline


# ==============================================================
# 🔹 Fixture: Mock Config
# ==============================================================
@pytest.fixture
def mock_yolov8_config(tmp_path):
    """Creates a minimal fake YOLOv8 config similar to config.yaml."""
    saved_model_dir = tmp_path / "saved_model" / "yolo_cropper"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov8s",
                "input_dir": str(tmp_path / "data" / "original")
            },
            "yolov8": {
                "yolov8_dir": str(tmp_path / "third_party" / "yolov8")
            },
            "dataset": {
                "saved_model_dir": str(saved_model_dir),
                "base_dir": str(tmp_path / "data" / "yolo_cropper"),
                "input_dir": str(tmp_path / "data" / "original"),
            }
        }
    }


# ==============================================================
# 🔹 Smoke Test (주석 여부 무관)
# ==============================================================
def test_yolov8_pipeline_runs_without_errors(tmp_path, mock_yolov8_config):
    """
    ✅ 목적: YOLOv8Pipeline이 예외 없이 실행되는지만 확인.
    내부 단계가 주석되어 있거나 반환값이 None이어도 PASS.
    """

    with patch("src.yolo_cropper.models.yolov8.yolov8.load_yaml_config", return_value=mock_yolov8_config), \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOv8Trainer"), \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOv8Evaluator"), \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOv8Predictor"), \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOPredictListGenerator"), \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOConverter"), \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOCropper"):

        # --- Run pipeline safely ---
        pipeline = YOLOv8Pipeline(config_path="dummy_config.yaml")

        result = None
        try:
            result = pipeline.run()
        except Exception as e:
            pytest.fail(f"YOLOv8Pipeline.run() raised an exception: {e}")

        # --- Assertions ---
        assert result is None or isinstance(result, dict), \
            "YOLOv8Pipeline should complete successfully (None or dict allowed)"

        print(f"[✓] YOLOv8Pipeline smoke test passed → result: {result}")

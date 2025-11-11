#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_yolov5.py
-----------------------------
Lightweight smoke test for YOLOv5Pipeline (Config-driven)
✅ 목적:
- YOLOv5Pipeline이 내부 단계 주석 여부와 무관하게 정상 실행되는지 확인
- 실제 학습, 평가, 예측 등은 모두 mock 처리
"""

import pytest
from pathlib import Path
from unittest.mock import patch
from src.yolo_cropper.models.yolov5.yolov5 import YOLOv5Pipeline


# ==============================================================
# 🔹 Fixture: Mock Config
# ==============================================================
@pytest.fixture
def mock_yolov5_config(tmp_path):
    """Creates a minimal fake YOLOv5 config similar to config.yaml"""
    saved_model_dir = tmp_path / "saved_model" / "yolo_cropper"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov5",
                "input_dir": str(tmp_path / "data" / "original")
            },
            "yolov5": {
                "yolov5_dir": str(tmp_path / "third_party" / "yolov5")
            },
            "dataset": {
                "saved_model_dir": str(saved_model_dir),
                "train_data_dir": str(tmp_path / "data" / "yolo_cropper"),
                "input_dir": str(tmp_path / "data" / "original"),
            }
        }
    }


# ==============================================================
# 🔹 Smoke Test (주석 여부 무관)
# ==============================================================
def test_yolov5_pipeline_runs_without_errors(tmp_path, mock_yolov5_config):
    """
    ✅ 목적: YOLOv5Pipeline이 예외 없이 실행되는지만 확인.
    내부 단계 주석 여부나 리턴값(None/dict)과 무관하게 pass.
    """

    # --- Patch all heavy submodules to lightweight mocks ---
    with patch("src.yolo_cropper.models.yolov5.yolov5.load_yaml_config", return_value=mock_yolov5_config), \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOv5Trainer"), \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOv5Evaluator"), \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOv5Predictor"), \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOPredictListGenerator"), \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOConverter"), \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOCropper"):

        # --- Run pipeline ---
        pipeline = YOLOv5Pipeline(config_path="dummy_config.yaml")

        result = None
        try:
            result = pipeline.run()
        except Exception as e:
            pytest.fail(f"YOLOv5Pipeline.run() raised an exception: {e}")

        # --- Assertions ---
        assert result is None or isinstance(result, dict), \
            "YOLOv5Pipeline should complete successfully (None or dict allowed)"

        print(f"[✓] YOLOv5Pipeline smoke test passed → result: {result}")

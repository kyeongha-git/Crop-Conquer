#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_yolov8_pipeline.py
-----------------------
Smoke test for YOLOv8Pipeline (Config-driven)
Ensures each step executes sequentially without real training/evaluation.
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
    """Creates a fake YOLOv8 config dictionary similar to config.yaml."""
    saved_model_dir = tmp_path / "saved_model" / "yolo_cropper"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    # weight 파일 생성하지 않음 (학습 호출이 일어나도록)
    # (saved_model_dir / "yolov8s.pt").write_text("dummy")

    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov8s",
                "input_dir": str(tmp_path / "data" / "original")
            },
            "yolov8": {
                "some_setting": "test"
            },
            "dataset": {
                "saved_model_dir": str(saved_model_dir),
                "base_dir": str(tmp_path / "data" / "yolo_cropper"),
                "input_dir": str(tmp_path / "data" / "original"),
            }
        }
    }


# ==============================================================
# 🔹 Core Smoke Test
# ==============================================================
def test_yolov8_pipeline_runs_all_steps(tmp_path, mock_yolov8_config):
    """
    ✅ YOLOv8Pipeline.run() should execute all steps without raising exceptions.
    Uses MagicMock for all external submodules.
    """

    with patch("src.yolo_cropper.models.yolov8.yolov8.load_yaml_config", return_value=mock_yolov8_config), \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOv8Trainer") as MockTrainer, \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOv8Evaluator") as MockEvaluator, \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOv8Predictor") as MockPredictor, \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOPredictListGenerator") as MockListGen, \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOConverter") as MockConverter, \
         patch("src.yolo_cropper.models.yolov8.yolov8.YOLOCropper") as MockCropper:

        # --- Mock behavior 설정 ---
        MockTrainer.return_value.run.return_value = None
        MockEvaluator.return_value.run.return_value = {"mAP@0.5": 0.91, "mAP@0.5:0.95": 0.87}
        MockPredictor.return_value.run.return_value = None
        MockListGen.return_value.run.return_value = None
        MockConverter.return_value.run.return_value = None
        MockCropper.return_value.crop_from_json.return_value = None

        # --- 실행 ---
        pipeline = YOLOv8Pipeline(config_path="dummy_config.yaml")
        metrics = pipeline.run()

        # --- Assertions ---
        assert isinstance(metrics, dict)
        assert "mAP@0.5" in metrics
        assert MockTrainer.called, "Trainer should be invoked"
        assert MockEvaluator.called, "Evaluator should be invoked"
        assert MockPredictor.called, "Predictor should be invoked"
        assert MockListGen.called, "PredictListGenerator should be invoked"
        assert MockConverter.called, "Converter should be invoked"
        assert MockCropper.called, "Cropper should be invoked"

        print(f"[✓] YOLOv8Pipeline test passed → metrics: {metrics}")

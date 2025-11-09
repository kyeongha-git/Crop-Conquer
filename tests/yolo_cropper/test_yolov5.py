#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_yolov5_pipeline.py
-----------------------
Smoke test for YOLOv5Pipeline (Config-driven)
Ensures that each step executes sequentially without real training/evaluation.
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
    """Creates a fake YOLOv5 config similar to config.yaml"""
    saved_model_dir = tmp_path / "saved_model" / "yolo_cropper"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    # weight 파일은 존재하지 않게 (학습 호출 테스트 위해)
    # (saved_model_dir / "yolov5.pt").write_text("dummy")  # ❌ intentionally not created

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
# 🔹 Core Smoke Test
# ==============================================================
def test_yolov5_pipeline_runs_all_steps(tmp_path, mock_yolov5_config):
    """
    ✅ YOLOv5Pipeline.run() should execute all steps without raising exceptions.
    Uses MagicMock for all external submodules.
    """

    with patch("src.yolo_cropper.models.yolov5.yolov5.load_yaml_config", return_value=mock_yolov5_config), \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOv5Trainer") as MockTrainer, \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOv5Evaluator") as MockEvaluator, \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOv5Predictor") as MockPredictor, \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOPredictListGenerator") as MockListGen, \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOConverter") as MockConverter, \
         patch("src.yolo_cropper.models.yolov5.yolov5.YOLOCropper") as MockCropper:

        # --- Configure mocks ---
        MockTrainer.return_value.run.return_value = None
        MockEvaluator.return_value.run.return_value = {"precision": 0.91, "recall": 0.88}
        MockPredictor.return_value.run.return_value = None
        MockListGen.return_value.run.return_value = None
        MockConverter.return_value.run.return_value = None
        MockCropper.return_value.crop_from_json.return_value = None

        # --- Instantiate and run ---
        pipeline = YOLOv5Pipeline(config_path="dummy_config.yaml")
        result = pipeline.run()

        # --- Assertions ---
        assert isinstance(result, dict)
        assert "precision" in result
        assert MockTrainer.called, "Trainer must be called"
        assert MockEvaluator.called, "Evaluator must be called"
        assert MockPredictor.called, "Predictor must be called"
        assert MockListGen.called, "Predict list generator must be called"
        assert MockConverter.called, "Converter must be called"
        assert MockCropper.called, "Cropper must be called"

        print(f"[✓] YOLOv5Pipeline test passed → metrics: {result}")

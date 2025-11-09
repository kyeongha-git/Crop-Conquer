#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_darknet_pipeline.py
------------------------
Smoke test for DarknetPipeline (Config-driven)
Ensures that pipeline steps execute in sequence without real Darknet build/train.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.yolo_cropper.models.darknet.darknet import DarknetPipeline


# ==============================================================
# 🔹 Fixture: Mock Config
# ==============================================================
@pytest.fixture
def mock_config(tmp_path):
    """Create a minimal fake config.yaml-like dict"""
    saved_model_dir = tmp_path / "saved_model" / "yolo_cropper"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    # ❌ weight 파일은 생성하지 않는다
    # (saved_model_dir / "yolov4.weights").write_text("dummy weights")

    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov4",
                "input_dir": str(tmp_path / "data" / "original")
            },
            "darknet": {
                "darknet_dir": str(tmp_path / "third_party" / "darknet")
            },
            "dataset": {
                "saved_model_dir": str(saved_model_dir),
                "train_data_dir": str(tmp_path / "data" / "yolo_cropper"),
            }
        }
    }

# ==============================================================
# 🔹 Core Smoke Test
# ==============================================================
def test_darknet_pipeline_runs_all_steps(tmp_path, mock_config, monkeypatch):
    """
    ✅ DarknetPipeline.run() should execute all pipeline steps
    without raising exceptions (mocking submodules).
    """

    # --- Patch submodules to prevent real file operations ---
    with patch("src.yolo_cropper.models.darknet.darknet.load_yaml_config", return_value=mock_config), \
         patch("src.yolo_cropper.models.darknet.darknet.CfgManager") as MockCfg, \
         patch("src.yolo_cropper.models.darknet.darknet.MakeManager") as MockMake, \
         patch("src.yolo_cropper.models.darknet.darknet.DarknetDataPreparer") as MockPrep, \
         patch("src.yolo_cropper.models.darknet.darknet.DarknetTrainer") as MockTrain, \
         patch("src.yolo_cropper.models.darknet.darknet.DarknetEvaluator") as MockEval, \
         patch("src.yolo_cropper.models.darknet.darknet.DarknetPredictor") as MockPred, \
         patch("src.yolo_cropper.models.darknet.darknet.YOLOCropper") as MockCrop:

        # --- Configure mocks ---
        MockCfg.return_value.generate.return_value = tmp_path / "cfg" / "dummy.cfg"
        MockMake.return_value.configure.return_value = None
        MockMake.return_value.rebuild.return_value = None
        MockMake.return_value.verify_darknet.return_value = True
        MockPrep.return_value.prepare.return_value = None
        MockTrain.return_value.verify_files.return_value = True
        MockTrain.return_value.run.return_value = None
        MockEval.return_value.run.return_value = {"mAP@0.5": 87.5}
        MockPred.return_value.run.return_value = ("outputs/json_results/yolov4/result.json", "outputs/json_results/predict.txt")
        MockCrop.return_value.crop_from_json.return_value = None

        # --- Instantiate and run pipeline ---
        pipeline = DarknetPipeline(config_path="dummy_config.yaml")
        result = pipeline.run()

        # --- Assertions ---
        assert isinstance(result, str), "Pipeline should return a result.json path"
        assert "result.json" in result
        MockCfg.assert_called_once()
        MockMake.assert_called_once()
        MockPrep.assert_called_once()
        MockTrain.assert_called_once()
        MockEval.assert_called_once()
        MockPred.assert_called_once()
        MockCrop.assert_called_once()

        # --- Check step order (informal) ---
        MockTrain.return_value.run.assert_called_once()
        MockEval.return_value.run.assert_called_once()
        MockPred.return_value.run.assert_called_once()

        print(f"[✓] DarknetPipeline test passed → result: {result}")

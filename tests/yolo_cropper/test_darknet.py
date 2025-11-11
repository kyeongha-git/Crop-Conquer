#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_darknet.py
------------------------
Lightweight smoke test for DarknetPipeline.
✅ 목적: 실행 시 예외가 발생하지 않고 result.json 경로 문자열을 반환하는지만 확인.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
from src.yolo_cropper.models.darknet.darknet import DarknetPipeline


# ==============================================================
# 🔹 Fixture: Mock Config
# ==============================================================
@pytest.fixture
def mock_config(tmp_path):
    """Create a minimal fake config.yaml-like dict"""
    saved_model_dir = tmp_path / "saved_model" / "yolo_cropper"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

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
# 🔹 Core Smoke Test (Simplified)
# ==============================================================
def test_darknet_pipeline_runs_without_errors(tmp_path, mock_config):
    """
    ✅ 목적: DarknetPipeline이 정상 실행되는지만 확인.
    - 내부 단계 호출 여부는 검증하지 않음
    - 모든 서브모듈은 mock 처리 (실제 파일/빌드 없음)
    """

    # --- Patch all heavy submodules to no-op mocks ---
    with patch("src.yolo_cropper.models.darknet.darknet.load_yaml_config", return_value=mock_config), \
         patch("src.yolo_cropper.models.darknet.darknet.CfgManager"), \
         patch("src.yolo_cropper.models.darknet.darknet.MakeManager"), \
         patch("src.yolo_cropper.models.darknet.darknet.DarknetDataPreparer"), \
         patch("src.yolo_cropper.models.darknet.darknet.DarknetTrainer"), \
         patch("src.yolo_cropper.models.darknet.darknet.DarknetEvaluator"), \
         patch("src.yolo_cropper.models.darknet.darknet.DarknetPredictor") as MockPred, \
         patch("src.yolo_cropper.models.darknet.darknet.YOLOCropper"):

        # Predictor mock return (result path only)
        MockPred.return_value.run.return_value = (
            "outputs/json_results/yolov4/result.json",
            "outputs/json_results/predict.txt",
        )

        # --- Run pipeline ---
        pipeline = DarknetPipeline(config_path="dummy_config.yaml")
        result = pipeline.run()

        # --- Assertions ---
        assert isinstance(result, str), "Pipeline should return a result.json path"
        assert result.endswith("result.json"), "Returned path should be a result.json file"

        print(f"[✓] DarknetPipeline smoke test passed → {result}")

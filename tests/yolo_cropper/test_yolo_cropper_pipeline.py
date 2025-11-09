#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_yolo_cropper_controller.py
-------------------------------
Unit test for YOLOCropperController (Config-driven unified YOLO pipeline dispatcher)
"""

import pytest
from unittest.mock import patch, MagicMock
from src.yolo_cropper.yolo_cropper import YOLOCropperController


# ==============================================================
# 🔹 Fixture: Base Config Template
# ==============================================================
@pytest.fixture
def base_config(tmp_path):
    """Creates a minimal fake config dictionary with variable model_name."""
    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov5",  # 기본값 (테스트별로 수정)
            }
        }
    }


# ==============================================================
# ✅ Case 1 — YOLOv2 / YOLOv4 → DarknetPipeline
# ==============================================================
@pytest.mark.parametrize("model_name", ["yolov2", "yolov4"])
def test_controller_dispatches_darknet_pipeline(model_name, base_config):
    """YOLOv2 / YOLOv4 should dispatch to DarknetPipeline"""
    base_config["yolo_cropper"]["main"]["model_name"] = model_name

    with patch("src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=base_config), \
         patch("src.yolo_cropper.yolo_cropper.importlib.import_module") as mock_import:

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.run.return_value = {"mAP@0.5": 0.75}
        mock_import.return_value = MagicMock(DarknetPipeline=mock_pipeline_cls)

        controller = YOLOCropperController(config_path="dummy.yaml")
        metrics = controller.run()

        mock_import.assert_called_once_with("src.yolo_cropper.models.darknet.darknet")
        mock_pipeline_cls.assert_called_once()
        assert "mAP@0.5" in metrics
        print(f"[✓] Darknet dispatch success ({model_name}) → {metrics}")


# ==============================================================
# ✅ Case 2 — YOLOv5 → YOLOv5Pipeline
# ==============================================================
def test_controller_dispatches_yolov5_pipeline(base_config):
    """YOLOv5 should dispatch to YOLOv5Pipeline"""
    base_config["yolo_cropper"]["main"]["model_name"] = "yolov5"

    with patch("src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=base_config), \
         patch("src.yolo_cropper.yolo_cropper.importlib.import_module") as mock_import:

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.run.return_value = {"precision": 0.9}
        mock_import.return_value = MagicMock(YOLOv5Pipeline=mock_pipeline_cls)

        controller = YOLOCropperController(config_path="dummy.yaml")
        metrics = controller.run()

        mock_import.assert_called_once_with("src.yolo_cropper.models.yolov5.yolov5")
        mock_pipeline_cls.assert_called_once()
        assert "precision" in metrics
        print(f"[✓] YOLOv5 dispatch success → {metrics}")


# ==============================================================
# ✅ Case 3 — YOLOv8(s/m/l/x) → YOLOv8Pipeline
# ==============================================================
@pytest.mark.parametrize("model_name", ["yolov8s", "yolov8m", "yolov8l", "yolov8x"])
def test_controller_dispatches_yolov8_pipeline(model_name, base_config):
    """YOLOv8 variants should dispatch to YOLOv8Pipeline"""
    base_config["yolo_cropper"]["main"]["model_name"] = model_name

    with patch("src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=base_config), \
         patch("src.yolo_cropper.yolo_cropper.importlib.import_module") as mock_import:

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.run.return_value = {"mAP@0.5": 0.88}
        mock_import.return_value = MagicMock(YOLOv8Pipeline=mock_pipeline_cls)

        controller = YOLOCropperController(config_path="dummy.yaml")
        metrics = controller.run()

        mock_import.assert_called_once_with("src.yolo_cropper.models.yolov8.yolov8")
        mock_pipeline_cls.assert_called_once()
        assert "mAP@0.5" in metrics
        print(f"[✓] YOLOv8 dispatch success ({model_name}) → {metrics}")


# ==============================================================
# ❌ Case 4 — Unsupported model_name
# ==============================================================
def test_controller_raises_for_invalid_model(base_config):
    """Unsupported model_name should raise ValueError"""
    base_config["yolo_cropper"]["main"]["model_name"] = "yoloX99"

    with patch("src.yolo_cropper.yolo_cropper.load_yaml_config", return_value=base_config):
        with pytest.raises(ValueError):
            controller = YOLOCropperController(config_path="dummy.yaml")
            controller.run()

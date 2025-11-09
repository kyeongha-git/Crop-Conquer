#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_cropper.py
----------------
Unified YOLO Pipeline Controller

- Reads model_name from config.yaml → yolo_cropper.main.model_name
- Automatically dispatches to corresponding pipeline:
    - YOLOv2 / YOLOv4 → DarknetPipeline
    - YOLOv5 → YOLOv5Pipeline
    - YOLOv8 (s/m/l/x) → YOLOv8Pipeline
"""

import sys
import importlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from utils.load_config import load_yaml_config
from utils.logging import setup_logging, get_logger


# --------------------------------------------------------
# 🔹 Unified YOLO Cropper Controller
# --------------------------------------------------------
class YOLOCropperController:
    def __init__(self, config_path: str = "utils/config.yaml"):
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.Controller")

        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})

        # config에서 model_name 읽기
        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()
        self.logger.info(f"Initialized YOLO Cropper Controller ({self.model_name.upper()})")

    # --------------------------------------------------------
    # 🔹 모델별 파이프라인 매핑 및 실행
    # --------------------------------------------------------
    def run(self):
        """Dispatch execution based on model_name"""

        # YOLOv8 계열 문자열 패턴 처리
        if self.model_name.startswith("yolov8"):
            module_path = "src.yolo_cropper.models.yolov8.yolov8"
            class_name = "YOLOv8Pipeline"
        elif self.model_name in ("yolov2", "yolov4"):
            module_path = "src.yolo_cropper.models.darknet.darknet"
            class_name = "DarknetPipeline"
        elif self.model_name == "yolov5":
            module_path = "src.yolo_cropper.models.yolov5.yolov5"
            class_name = "YOLOv5Pipeline"
        else:
            raise ValueError(f"❌ Unsupported model_name: {self.model_name}")

        self.logger.info(f"📦 Loading pipeline → {module_path}.{class_name}")

        # 모듈 로드
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ImportError(f"❌ Failed to import module {module_path}: {e}")

        # 클래스 확인
        if not hasattr(module, class_name):
            raise AttributeError(f"❌ {module_path} 내에 '{class_name}' 클래스가 없습니다.")

        # 파이프라인 클래스 로드 및 실행
        pipeline_class = getattr(module, class_name)
        pipeline = pipeline_class(config_path=str(self.config_path))

        self.logger.info(f"🚀 Running {self.model_name.upper()} pipeline...")
        metrics = pipeline.run()
        self.logger.info(f"✅ Pipeline complete ({self.model_name.upper()})")

        return metrics


# --------------------------------------------------------
# ✅ CLI Entry
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified YOLO Cropper Controller")
    parser.add_argument("--config", type=str, default="utils/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    controller = YOLOCropperController(config_path=args.config)
    controller.run()

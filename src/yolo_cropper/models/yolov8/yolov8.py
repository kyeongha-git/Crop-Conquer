#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
darknet.py
----------
DarknetPipeline (Config-driven)
Unified YOLOv2 / YOLOv4 workflow manager.

Steps:
1️⃣ cfg_manager
2️⃣ make_manager
3️⃣ data_prepare
4️⃣ train (skip if saved_model exists)
5️⃣ evaluate
6️⃣ predict
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging

# === Darknet Submodules ===
from src.yolo_cropper.models.yolov8.train import YOLOv8Trainer
from src.yolo_cropper.models.yolov8.evaluate import YOLOv8Evaluator
from src.yolo_cropper.models.yolov8.predict import YOLOv8Predictor
from src.yolo_cropper.core.make_predict import YOLOPredictListGenerator
from src.yolo_cropper.core.converter import YOLOConverter
from src.yolo_cropper.core.cropper import YOLOCropper


class YOLOv8Pipeline:
    """Unified YOLOv5 pipeline orchestrator (config-driven)."""

    def __init__(self, config_path: str = "utils/config.yaml"):
        setup_logging("logs/yolo_cropper")
        self.logger = get_logger("yolo_cropper.DarknetPipeline")

        # --------------------------------------------------------
        # Load Configuration
        # --------------------------------------------------------
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        # Shortcut configs
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = yolo_cropper_cfg.get("main", {})
        self.yolov8_cfg = yolo_cropper_cfg.get("yolov8", {})
        self.train_cfg = yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = yolo_cropper_cfg.get("dataset", {})

        # Paths
        self.model_name = self.main_cfg.get("model_name", "yolov8s").lower()
        self.saved_model_dir = Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")).resolve()
        self.base_dataset_dir = Path(
            f"{self.dataset_cfg.get('base_dir', 'data/yolo_cropper')}/{self.model_name}"
        ).resolve()

        self.input_dir = Path(self.dataset_cfg.get("input_dir", "data/original")).resolve()

        # Derived paths
        self.weight_path = self.saved_model_dir / f"{self.model_name}.pt"


        # Logging info
        self.logger.info(f"Initialized DarknetPipeline ({self.model_name.upper()})")
        self.logger.info(f" - Config path    : {self.config_path}")
        self.logger.info(f" - Training Dataset dir    : {self.base_dataset_dir}")
        self.logger.info(f" - Saved model dir: {self.saved_model_dir}")
        self.logger.info(f" - Input dir      : {self.input_dir}")

    # --------------------------------------------------------
    # Step 1️⃣ Train
    # --------------------------------------------------------
    def step_train(self):
        self.logger.info("[STEP 1] Starting YOLO v8 training...")
        if self.weight_path.exists():
            self.logger.info(f"[SKIP] Found existing trained model → {self.weight_path}")
            return
        trainer = YOLOv8Trainer(config=self.cfg)
        trainer.run()
        self.logger.info("[✓] Training step done")
        
    # --------------------------------------------------------
    # Step 2️⃣ Evaluate
    # --------------------------------------------------------
    def step_evaluate(self):
        self.logger.info("[STEP 2] Evaluation starts")
        evaluator  = YOLOv8Evaluator(config=self.cfg)
        metrics = evaluator.run()
        self.logger.info("[✓] Evaluation step done")
        return metrics


    # --------------------------------------------------------
    # Step 3️⃣ Make predict.txt
    # --------------------------------------------------------
    def step_make_predict(self):
        self.logger.info("[STEP 4] Generating predict.txt")
        maker = YOLOPredictListGenerator(config=self.cfg)  # config-driven
        maker.run()
        self.logger.info("[✓] predict.txt generated")


    # --------------------------------------------------------
    # Step 4️⃣ Predict (auto multi-folder)
    # --------------------------------------------------------
    def step_predict(self):
        self.logger.info("[STEP 3] Preparing dataset for Darknet...")
        predictor = YOLOv8Predictor(config=self.cfg)
        predictor.run()
        self.logger.info("[✓] Prediction step done")


    # --------------------------------------------------------
    # Step 5️⃣ Converter (YOLOv5 detect → unified result.json)
    # --------------------------------------------------------
    def step_converter(self):
        self.logger.info("[STEP 5] Converting YOLOv5 detects → result.json")
        conv = YOLOConverter(config=self.cfg)  # config-driven
        conv.run()
        self.logger.info("[✓] Conversion step done")
    
    # -------------------------------------------------
    # Step 6️⃣ Cropper (result.json 기반 ROI crop)
    # -------------------------------------------------
    def step_cropper(self):
        self.logger.info("[STEP 6] Cropping from result.json")
        cropper = YOLOCropper(config=self.cfg)  # config-driven
        cropper.crop_from_json()
        self.logger.info("[✓] Cropping step done")

    # --------------------------------------------------------
    # Unified Runner
    # --------------------------------------------------------
    def run(self):
        self.logger.info("🚀 Running YOLOv8 Pipeline")
        self.step_train()
        metrics = self.step_evaluate()
        self.step_make_predict()
        self.step_predict()
        self.step_converter()
        self.step_cropper()
        self.logger.info("=== ✅ YOLOv8 PIPELINE COMPLETE ===")
        return metrics

# --------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO v8 Unified Pipeline Runner")
    parser.add_argument("--config", type=str, default="utils/config.yaml",
                        help="Path to configuration YAML file")
    args = parser.parse_args()

    pipeline = YOLOv8Pipeline(config_path=args.config)
    pipeline.run()

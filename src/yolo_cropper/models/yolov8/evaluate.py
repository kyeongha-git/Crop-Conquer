#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluator.py
------------
YOLOv8 Evaluator (Config-driven)
- config.yaml 기반으로 학습된 모델 평가
- Precision / Recall / mAP@0.5 / mAP@0.5:0.95 계산
- 결과를 metrics/yolo_cropper/{model_name}_metrics.csv 에 저장
"""

from pathlib import Path
from datetime import datetime
import csv
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging


def safe_mean(value):
    """배열 또는 스칼라를 평균값으로 안전하게 변환"""
    if hasattr(value, "mean"):
        return float(np.mean(value))
    return float(value)


class YOLOv8Evaluator:
    """YOLOv8 Evaluation Class (config-driven)"""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.YOLOv8Evaluator")
        self.cfg = config

        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.yolov8_cfg = self.yolo_cropper_cfg.get("yolov8", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.train_cfg = self.yolo_cropper_cfg.get("train", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        self.model_name = self.main_cfg.get("model_name", "yolov8s")
        self.data_yaml = Path(self.yolov8_cfg.get("data_yaml", "data/yolo_cropper/yolov8/data.yaml")).resolve()
        self.imgsz = self.train_cfg.get("imgsz", 416)
        self.saved_model_dir = Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")).resolve()
        self.weights_path = (self.saved_model_dir / f"{self.model_name}.pt").resolve()

        self.metrics_dir = Path(self.dataset_cfg.get("metrics_dir", "metrics/yolo_cropper")).resolve()
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.metrics_dir / f"{self.model_name}_metrics.csv"

        self.logger.info(f"Initialized YOLOv8Evaluator ({self.model_name.upper()})")
        self.logger.debug(f" - Data YAML  : {self.data_yaml}")
        self.logger.debug(f" - Weights    : {self.weights_path}")
        self.logger.debug(f" - Metrics CSV: {self.csv_path}")

    # --------------------------------------------------------
    def run(self):
        """Evaluate YOLOv8 model and save metrics"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"❌ Model weights not found → {self.weights_path}")
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"❌ Dataset YAML not found → {self.data_yaml}")

        self.logger.info(f"🚀 Evaluating {self.model_name} on dataset: {self.data_yaml}")

        model = YOLO(self.weights_path)
        metrics = model.val(data=str(self.data_yaml), imgsz=self.imgsz, verbose=False)

        result_dict = {
            "model": self.model_name,
            "precision": safe_mean(metrics.box.p),
            "recall": safe_mean(metrics.box.r),
            "mAP@0.5": safe_mean(metrics.box.map50),
            "mAP@0.5:0.95": safe_mean(metrics.box.map),
        }

        # --- Save metrics to CSV ---
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[*result_dict.keys(), "timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow({**result_dict, "timestamp": timestamp})

        self.logger.info(f"[✓] Metrics saved → {self.csv_path}")
        self.logger.info(
            f"Precision: {result_dict['precision']:.4f} | Recall: {result_dict['recall']:.4f} | "
            f"mAP@0.5: {result_dict['mAP@0.5']:.4f} | mAP@0.5:0.95: {result_dict['mAP@0.5:0.95']:.4f}"
        )

        return result_dict

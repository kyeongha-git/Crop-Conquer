#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py
-----------
DarknetEvaluator (Config-driven)
- Supports YOLOv2 / YOLOv4 evaluation
- Receives config from main controller (no YAML read inside)
"""

import subprocess
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger
from src.yolo_cropper.metrics.metrics import get_metrics_parser


class DarknetEvaluator:
    """Runs Darknet mAP evaluation and parses results (YOLOv2 / YOLOv4)."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.DarknetEvaluator")

        # --------------------------------------------------------
        # Parse config
        # --------------------------------------------------------
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        # Directories
        self.darknet_dir = Path(self.darknet_cfg.get("darknet_dir", "third_party/darknet")).resolve()
        self.saved_model_dir = Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper"))
        self.metrics_dir = Path(self.dataset_cfg.get("metrics_dir", "metrics/yolo_cropper"))
        self.log_dir = self.darknet_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = self.main_cfg.get("model_name", "yolov2").lower()

        self.logger.info(f"Initialized DarknetEvaluator for {self.model_name.upper()}")
        self.logger.debug(f"Darknet dir : {self.darknet_dir}")
        self.logger.debug(f"Log dir     : {self.log_dir}")

    # --------------------------------------------------------
    # 🔹 Run evaluation
    # --------------------------------------------------------
    def run(self):
        """Run Darknet evaluation and parse results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"eval_{self.model_name}_{timestamp}.log"
        log_path = self.log_dir / log_filename
        relative_log_path = f"logs/{log_filename}"

        # Paths (Darknet 기준 상대 경로)
        obj_data = "data/obj.data"
        cfg_path = f"cfg/{self.model_name}-obj.cfg"
        weights_path = Path(f"{self.saved_model_dir}/{self.model_name}.weights").resolve()

        # ✅ Darknet 명령어 구성
        command = (
            f'./darknet detector map {obj_data} {cfg_path} {weights_path} '
            f'-dont_show -iou_thresh 0.5 -points 101 | tee {relative_log_path}'
        )

        self.logger.info(f"🚀 Starting Darknet evaluation ({self.model_name.upper()})...")
        self.logger.debug(f"[CMD] {command}")

        process = subprocess.run(
            ["bash", "-lc", command],
            cwd=self.darknet_dir,
            shell=False,
        )

        # ✅ Darknet의 반환 코드 처리
        if process.returncode not in (0, 1):
            raise RuntimeError(f"❌ Darknet evaluation failed (code: {process.returncode})")
        elif process.returncode == 1:
            self.logger.warning("Darknet exited with code 1 (non-fatal). Evaluation likely succeeded.")

        self.logger.info(f"[✓] Evaluation complete! Log saved → {log_path}")

        # ✅ 모델 이름 기반으로 적절한 metrics parser 선택
        parser = get_metrics_parser(self.model_name)
        metrics = parser(str(log_path))

        self._save_metrics_to_csv(metrics)
        return metrics

    # --------------------------------------------------------
    # 🔹 Save metrics to CSV
    # --------------------------------------------------------
    def _save_metrics_to_csv(self, metrics: dict):
        """Save parsed metrics to metrics/yolo_cropper/{model_name}_metrics.csv"""
        save_dir = self.metrics_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        csv_path = save_dir / f"{self.model_name}_metrics.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        row = {
            "model": self.model_name,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "mAP@0.5": metrics.get("mAP@0.5"),
            "timestamp": timestamp,
        }

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self.logger.info(f"[✓] Metrics saved → {csv_path}")

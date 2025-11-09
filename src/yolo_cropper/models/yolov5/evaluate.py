#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py
-----------
YOLOv5Evaluator (Config-driven)
- Uses Ultralytics val.py for evaluation
- Unified metrics parsing via get_metrics_parser()
"""

import subprocess
import csv
from datetime import datetime
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger
from src.yolo_cropper.metrics.metrics import get_metrics_parser


class YOLOv5Evaluator:
    """Runs YOLOv5 evaluation and parses metrics.csv results."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.YOLOv5Evaluator")

        # --------------------------------------------------------
        # Config Parsing
        # --------------------------------------------------------
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.yolov5_cfg = self.yolo_cropper_cfg.get("yolov5", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        # --------------------------------------------------------
        # Directories & Parameters
        # --------------------------------------------------------
        self.yolov5_dir = Path(self.yolov5_cfg.get("yolov5_dir", "third_party/yolov5")).resolve()
        self.data_yaml_path = Path(self.yolov5_cfg.get("data_yaml", "data/yolo_cropper/yolov5/data.yaml")).resolve()
        self.data_yaml = self._resolve_data_yaml(self.data_yaml_path)
        self.saved_model_dir = Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")).resolve()
        self.metrics_dir = Path(self.dataset_cfg.get("metrics_dir", "metrics/yolo_cropper")).resolve()
        self.log_dir = self.yolov5_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()
        self.weights_path = self.saved_model_dir / f"{self.model_name}.pt"

        self.logger.info(f"Initialized YOLOv5Evaluator for {self.model_name.upper()}")
        self.logger.debug(f"Repo Dir  : {self.yolov5_dir}")
        self.logger.debug(f"Data YAML : {self.data_yaml}")
        self.logger.debug(f"Weights   : {self.weights_path}")
    

    def _resolve_data_yaml(self, data_yaml_path: Path) -> Path:
        """
        Convert train/val/test paths in data.yaml to absolute paths.
        Example:
            train: ../train/images  → /abs/path/to/train/images
        """
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        base = data_yaml_path.parent.resolve()
        for key in ("train", "val", "test"):
            if key in data and isinstance(data[key], str):
                abs_path = (base / data[key]).resolve()
                if not abs_path.exists() and (abs_path / "images").exists():
                    abs_path = abs_path / "images"
                data[key] = str(abs_path)
                self.logger.info(f"  └─ Resolved {key}: {abs_path}")

        tmp_dir = Path(tempfile.mkdtemp(prefix="yolov5_datayaml_"))
        resolved_yaml = tmp_dir / "data_resolved.yaml"
        with open(resolved_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        self.logger.info(f"[✓] Temporary data.yaml created → {resolved_yaml}")
        return resolved_yaml

    # --------------------------------------------------------
    # 🔹 Run Evaluation
    # --------------------------------------------------------
    def run(self):
        """Run YOLOv5 evaluation using val.py and parse metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"val_{timestamp}.log"

        cmd = [
            "python", "val.py",
            "--data", str(self.data_yaml),
            "--weights", str(self.weights_path),
            "--task", "val",
            "--save-json"
        ]

        self.logger.info(f"🚀 Starting YOLOv5 evaluation ({self.model_name.upper()})...")
        self.logger.debug(f"[CMD] {' '.join(cmd)}")

        with open(log_path, "w", encoding="utf-8") as log_f:
            process = subprocess.run(cmd, cwd=self.yolov5_dir, stdout=log_f, stderr=subprocess.STDOUT)

        if process.returncode != 0:
            raise RuntimeError(f"❌ YOLOv5 evaluation failed (code={process.returncode}). See log: {log_path}")

        self.logger.info(f"[✓] YOLOv5 evaluation complete → {log_path}")

        # 1️⃣ Find latest results file (csv / json)
        results_dir = self.yolov5_dir / "runs" / "val"
        exp_dirs = sorted(results_dir.glob("exp*"))
        if not exp_dirs:
            raise FileNotFoundError(f"No val results found in {results_dir}")

        results_file = log_path  # 🔹 pass log path to parser

        parser = get_metrics_parser(self.model_name)
        metrics = parser(str(results_file))

        # 3️⃣ Save metrics to CSV
        self._save_metrics_to_csv(metrics)
        return metrics

    # --------------------------------------------------------
    # 🔹 Save Metrics
    # --------------------------------------------------------
    def _save_metrics_to_csv(self, metrics: dict):
        """Save parsed metrics to metrics/yolo_cropper/{model_name}_metrics.csv"""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.metrics_dir / f"{self.model_name}_metrics.csv"
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


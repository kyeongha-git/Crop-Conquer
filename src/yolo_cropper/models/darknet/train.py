#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py
--------
DarknetTrainer (Config-driven)

- Supports YOLOv2 / YOLOv4 training
- Receives full config object from YOLOCropper (no YAML read inside)
- Uses structured logging instead of print
"""

import os
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class DarknetTrainer:
    """Handles Darknet YOLOv2 / YOLOv4 training process."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.DarknetTrainer")

        # --------------------------------------------------------
        # Parse nested config
        # --------------------------------------------------------
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = self.darknet_cfg.get("dataset", {})

        self.darknet_dir = Path(self.darknet_cfg.get("darknet_dir", "third_party/darknet")).resolve()
        self.model_name = self.main_cfg.get("model_name", "yolov2")

        # Paths inside Darknet repo
        self.binary = self.darknet_dir / "darknet"
        self.data_file = "data/obj.data"
        self.cfg_file = f"cfg/{self.model_name}-obj.cfg"

        # Pretrained weights
        if self.model_name == "yolov2":
            self.weights_file = self.darknet_dir / "yolov2.weights"
        elif self.model_name == "yolov4":
            self.weights_file = self.darknet_dir / "yolov4.conv.137"
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        # Logs directory
        self.logs_dir = self.darknet_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized DarknetTrainer for {self.model_name.upper()}")
        self.logger.debug(f"  - Darknet dir : {self.darknet_dir}")
        self.logger.debug(f"  - Weight file : {self.weights_file}")
        self.logger.debug(f"  - Config file : {self.cfg_file}")

    # ------------------------------------------------------------
    # Verify all required files
    # ------------------------------------------------------------
    def verify_files(self) -> bool:
        """Check if all required files exist before training."""
        required = [
            self.binary,
            self.darknet_dir / self.data_file,
            self.darknet_dir / self.cfg_file,
            self.weights_file,
        ]
        missing = [str(f) for f in required if not f.exists()]

        if missing:
            self.logger.error("Missing required files:")
            for m in missing:
                self.logger.error(f"   - {m}")
            return False

        self.logger.info("[✓] All required files are present for training.")
        return True

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    def train(self, weights_init: str = None):
        """Run Darknet training (CPU/GPU agnostic)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"train_{self.model_name}_{timestamp}.log"

        init_weight = weights_init or os.path.basename(self.weights_file)
        clear_flag = "-clear" if self.model_name == "yolov2" else ""

        cmd = [
            "bash",
            "-lc",
            (
                f'./darknet detector train {self.data_file} {self.cfg_file} {init_weight} '
                f'{clear_flag} -dont_show -map | tee {log_path}'
            ),
        ]

        self.logger.info(f"🚀 Starting Darknet training ({self.model_name.upper()})")
        self.logger.debug(f"[CMD] {' '.join(cmd[2:])}")

        process = subprocess.run(cmd, cwd=self.darknet_dir, shell=False)

        if process.returncode != 0:
            raise RuntimeError(
                f"❌ Training failed (code: {process.returncode}). See log: {log_path}"
            )

        self.logger.info(f"[✓] Training complete! Log saved → {log_path}")

        # --------------------------------------------------------
        # ✅ Copy best checkpoint to saved_model
        # --------------------------------------------------------
        ckpt_dir = Path(self.dataset_cfg.get("checkpoints_dir", "checkpoints/yolo_cropper")/{self.model_name})
        saved_dir = Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper")/{self.model_name})

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        saved_dir.mkdir(parents=True, exist_ok=True)

        best_weight = next(ckpt_dir.glob("*_best.weights"), None)
        target_weight = saved_dir / f"{self.model_name}.weights"

        if best_weight:
            shutil.copy2(best_weight, target_weight)
            self.logger.info(f"[✓] Copied best weight → {target_weight}")
        else:
            self.logger.warning("[!] No '_best.weights' file found in checkpoints directory.")

        return str(target_weight)


if __name__ == "__main__":
    from utils.load_config import load_yaml_config
    from utils.logging import setup_logging
    import argparse

    parser = argparse.ArgumentParser(description="Debug Darknet training")
    parser.add_argument("--config", type=str, default="utils/config.yaml")
    args = parser.parse_args()

    setup_logging("logs/yolo_cropper")
    cfg = load_yaml_config(args.config)

    trainer = DarknetTrainer(config=cfg)
    if trainer.verify_files():
        trainer.train()

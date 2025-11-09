#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cfg_manager.py
--------------
Darknet (YOLOv2 / YOLOv4) Configuration Manager
"""

from pathlib import Path
import shutil
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging


class CfgManager:
    """
    Darknet cfg file generator.
    Receives entire YAML config from main controller (YOLOCropper).
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.CfgManager")

        # -----------------------------
        # Parse nested config
        # -----------------------------
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.cfg_overrides = self.darknet_cfg.get("cfg_overrides", {})
        self.model_name = self.main_cfg.get("model_name", "yolov2").lower().strip()
        
        if self.model_name not in ("yolov2", "yolov4"):
            raise ValueError("model_name must be either 'yolov2' or 'yolov4'")

        # -----------------------------
        # Define paths
        # -----------------------------
        darknet_root = Path(
            self.darknet_cfg.get("darknet_dir", "third_party/darknet")
        ).resolve()

        self.base_cfg = darknet_root / "cfg" / f"{self.model_name}.cfg"
        self.target_cfg = darknet_root / "cfg" / f"{self.model_name}-obj.cfg"

        if not self.base_cfg.exists():
            raise FileNotFoundError(f"Base cfg not found: {self.base_cfg}")

        self.target_cfg.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized CfgManager for {self.model_name.upper()}")
        self.logger.debug(f"Base cfg   : {self.base_cfg}")
        self.logger.debug(f"Target cfg : {self.target_cfg}")

    # ----------------------------------------------------------
    def _copy_base_cfg(self) -> None:
        shutil.copy(self.base_cfg, self.target_cfg)
        self.logger.info(f"Copied base {self.model_name}.cfg → {self.target_cfg}")

    # ----------------------------------------------------------
    def _apply_cfg_overrides(self) -> None:
        overrides = self.cfg_overrides
        if not overrides:
            self.logger.warning("No 'cfg_overrides' found in config.")
            return

        with open(self.target_cfg, "r", encoding="utf-8") as f:
            lines = f.readlines()

        def replace_line(lines, key, val):
            for i, line in enumerate(lines):
                if line.strip().replace(" ", "").startswith(f"{key}="):
                    lines[i] = f"{key}={val}\n"
            return lines

        # Inside [net]
        in_net = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[net]"):
                in_net = True
                continue
            elif stripped.startswith("[") and not stripped.startswith("[net]"):
                in_net = False
            if in_net:
                for k, v in overrides.items():
                    if stripped.replace(" ", "").startswith(f"{k}="):
                        lines[i] = f"{k}={v}\n"

        # Global replacements
        for k, v in overrides.items():
            lines = replace_line(lines, k, v)

        with open(self.target_cfg, "w", encoding="utf-8") as f:
            f.writelines(lines)

        self.logger.info("Applied cfg_overrides from config.yaml")

    # ----------------------------------------------------------
    def _update_filters_and_classes(self) -> None:
        overrides = self.cfg_overrides
        num_classes = int(overrides.get("classes", 2))

        if self.model_name == "yolov2":
            filters = (num_classes + 5) * 5
            marker = "[region]"
        else:
            filters = (num_classes + 5) * 3
            marker = "[yolo]"

        with open(self.target_cfg, "r", encoding="utf-8") as f:
            lines = f.readlines()

        updated = lines.copy()

        for i, line in enumerate(lines):
            if line.strip().startswith(marker):
                # Backtrack to [convolutional]
                j = i - 1
                while j >= 0 and not lines[j].strip().startswith("[convolutional]"):
                    j -= 1
                if j >= 0:
                    for k in range(j, i):
                        if lines[k].strip().startswith("filters="):
                            updated[k] = f"filters={filters}\n"
                            break
                # Update classes=
                for k in range(i, len(lines)):
                    if lines[k].strip().startswith("classes="):
                        updated[k] = f"classes={num_classes}\n"
                        break

        with open(self.target_cfg, "w", encoding="utf-8") as f:
            f.writelines(updated)

        self.logger.info(
            f"Updated filters={filters}, classes={num_classes} ({self.model_name.upper()})"
        )

    # ----------------------------------------------------------
    def generate(self) -> str:
        self.logger.info(f"🔧 Generating cfg for {self.model_name.upper()} ...")
        self._copy_base_cfg()
        self._apply_cfg_overrides()
        self._update_filters_and_classes()
        self.logger.info(f"✅ Config generated → {self.target_cfg}")
        return str(self.target_cfg)
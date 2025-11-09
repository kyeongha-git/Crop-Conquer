#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_predict_list.py
--------------------
YOLOv5PredictListGenerator (Config-driven)
- Generates predict.txt listing all image paths under dataset root.
- Receives config dict (no YAML read inside).
- Detects `input_root` automatically from config.
"""

from pathlib import Path
from typing import Dict, Any, List
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class YOLOPredictListGenerator:
    """Generates predict.txt listing image paths under input_root."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.YOLOvPredictListGenerator")

        # --------------------------------------------------------
        # Parse Config
        # --------------------------------------------------------
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.yolov5_cfg = self.yolo_cropper_cfg.get("yolov5", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        # --------------------------------------------------------
        # Directories
        # --------------------------------------------------------
        self.input_root = Path(self.main_cfg.get("input_dir", "data/yolo_cropper/original")).resolve()
        self.output_dir = Path(self.dataset_cfg.get("results_dir", "outputs/json_results")).resolve()
        self.output_path = self.output_dir / "predict.txt"

        # --------------------------------------------------------
        # Initialize
        # --------------------------------------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.input_root.exists():
            raise FileNotFoundError(f"❌ Source root not found: {self.input_root}")

        self.logger.info(f"YOLOv5PredictListGenerator initialized")
        self.logger.debug(f"Source root : {self.input_root}")
        self.logger.debug(f"Output path : {self.output_path}")

    # ==========================================================
    # 🔹 Collect image paths
    # ==========================================================
    def _collect_images(self) -> List[str]:
        """Collect all image paths under repair/replace folders."""
        exts = [".jpg", ".jpeg", ".png"]
        all_images = []

        for cls in ["repair", "replace"]:
            class_dir = self.input_root / cls
            if not class_dir.exists():
                self.logger.warning(f"[!] Missing class folder: {class_dir}")
                continue

            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in exts:
                    all_images.append(str(img_path.resolve()))

        if not all_images:
            raise FileNotFoundError(f"No images found under {self.input_root}")

        all_images.sort()
        return all_images

    # ==========================================================
    # 🔹 Write predict.txt
    # ==========================================================
    def _write_output(self, image_paths: List[str]):
        """Write all collected image paths to predict.txt."""
        self.output_path.write_text("\n".join(image_paths), encoding="utf-8")
        self.logger.info(f"[✓] Generated predict.txt → {self.output_path}")
        self.logger.info(f"   - Dataset root : {self.input_root}")
        self.logger.info(f"   - Total images : {len(image_paths)}")

    # ==========================================================
    # 🔹 Run full process
    # ==========================================================
    def run(self):
        """Generate predict.txt under outputs/json_results."""
        images = self._collect_images()
        self._write_output(images)


# ==========================================================
# ✅ Debug CLI Entry
# ==========================================================
if __name__ == "__main__":
    from utils.load_config import load_yaml_config
    from utils.logging import setup_logging
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv5 Predict List Generator (config-driven)")
    parser.add_argument("--config", type=str, default="utils/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    setup_logging("logs/yolo_cropper")
    cfg = load_yaml_config(args.config)

    generator = YOLOPredictListGenerator(config=cfg)
    generator.run()

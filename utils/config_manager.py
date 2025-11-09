#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config_manager.py
-----------------
Dynamic Config Manager (with CLI overrides)

- Reads base config.yaml
- Dynamically updates paths based on:
    annot_clean / yolo_crop / yolo_model
- Accepts CLI overrides directly
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Dynamic Config Manager that updates config.yaml paths."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.cfg = self._load_yaml()

        main_cfg = self.cfg.get("main", {})
        self.test_mode = self.cfg.get("annotation_cleaner", {}).get("annotation_clean", {}).get("test_mode", True)
        self.annot_clean = main_cfg.get("annot_clean", "on")
        self.yolo_crop = main_cfg.get("yolo_crop", "on")
        self.yolo_model = main_cfg.get("yolo_model", "yolov8s")

    # --------------------------------------------------------
    def _load_yaml(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # --------------------------------------------------------
    def update_paths(
        self,
        annot_clean: Optional[str] = None,
        yolo_crop: Optional[str] = None,
        yolo_model: Optional[str] = None,
        test_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update paths dynamically based on given overrides."""

        # CLI override 우선 반영
        if annot_clean is not None:
            self.annot_clean = annot_clean
        if yolo_crop is not None:
            self.yolo_crop = yolo_crop
        if yolo_model is not None:
            self.yolo_model = yolo_model
        if test_mode is not None:
            self.test_mode = test_mode

        # === Determine base dataset name ===
        if self.annot_clean == "on":
            base_dir = Path("data/generation")
        else:
            base_dir = Path("data/original")

        # === Determine cropped dataset dir ===
        if self.yolo_crop == "on":
            crop_dir = base_dir.parent / f"{base_dir.name}_crop" / self.yolo_model
        else:
            crop_dir = base_dir

        # === Update sections ===
        yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        data_augmentor_cfg = self.cfg.get("data_augmentor", {})
        classifier_cfg = self.cfg.get("Classifier", {})
        annotation_cfg = self.cfg.get("annotation_cleaner", {})

        # Annotation Cleaner test_mode    
        annotation_cfg.setdefault("annotation_clean", {})
        annotation_cfg["annotation_clean"]["test_mode"] = self.test_mode

        # YOLO Cropper input_dir
        yolo_cropper_cfg.setdefault("main", {})
        yolo_cropper_cfg["main"]["input_dir"] = str(base_dir)
        yolo_cropper_cfg["main"]["model_name"] = self.yolo_model

        # DataAugmentor input_dir
        data_augmentor_cfg.setdefault("data", {})
        data_augmentor_cfg["data"]["input_dir"] = str(crop_dir)
        data_augmentor_cfg["data"]["output_dir"] = str(crop_dir)

        # Classifier input_dir
        classifier_cfg.setdefault("data", {})
        classifier_cfg["data"]["input_dir"] = str(crop_dir)

        # Update main config
        self.cfg["main"]["annot_clean"] = self.annot_clean
        self.cfg["main"]["yolo_crop"] = self.yolo_crop
        self.cfg["main"]["yolo_model"] = self.yolo_model
        self.cfg["annotation_cleaner"] = annotation_cfg
        self.cfg["yolo_cropper"] = yolo_cropper_cfg
        self.cfg["data_augmentor"] = data_augmentor_cfg
        self.cfg["classifier"] = classifier_cfg

        return self.cfg

    # --------------------------------------------------------
    def save(self, output_path: Optional[str] = None):
        """Save updated config to YAML file."""
        target_path = Path(output_path or self.config_path)
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, allow_unicode=True)
        print(f"[✓] Updated config saved → {target_path}")


# --------------------------------------------------------
# ✅ CLI Debug Entry
# --------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Config Manager CLI")
    parser.add_argument("--config", type=str, default="utils/config.yaml")
    parser.add_argument("--annot_clean", type=str, choices=["on", "off"], default=None)
    parser.add_argument("--yolo_crop", type=str, choices=["on", "off"], default=None)
    parser.add_argument("--yolo_model", type=str, default=None)
    args = parser.parse_args()

    mgr = ConfigManager(args.config)
    updated = mgr.update_paths(
        annot_clean=args.annot_clean,
        yolo_crop=args.yolo_crop,
        yolo_model=args.yolo_model,
    )
    mgr.save()

    print("=== ✅ Updated Paths Summary ===")
    print(f"YOLO Cropper Input : {updated['yolo_cropper']['main']['input_dir']}")
    print(f"Augmentor Input    : {updated['data_augmentor']['data']['input_dir']}")
    print(f"Classifier Input   : {updated['classifier']['data']['input_dir']}")

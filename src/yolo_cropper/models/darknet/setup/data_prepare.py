#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_prepare.py
---------------
DarknetDataPreparer (Config-driven)

- Prepares dataset splits and metadata files (train.txt, valid.txt, test.txt, obj.data, obj.names)
- Works with injected config object (no direct YAML read)
- Compatible with third_party/darknet/data layout
"""

from pathlib import Path
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class DarknetDataPreparer:
    """
    Prepares dataset for Darknet (YOLOv2/v4).
    Supports Roboflow-like (train/images/*.jpg) and flat (train/*.jpg) structures.
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.DarknetDataPreparer")

        # ----------------------------------------------------
        # Parse nested config
        # ----------------------------------------------------
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        # Directories
        self.base_dir = Path(self.dataset_cfg.get("train_data_dir", "data/yolo_cropper"))
        self.darknet_data_dir = Path(self.darknet_cfg.get("darknet_data_dir", "third_party/darknet/data"))
        self.model_name = self.main_cfg.get("model_name", "yolov2").lower()

        self.darknet_data_dir.mkdir(parents=True, exist_ok=True)

        if not self.base_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.base_dir}")

        self.logger.info(
            f"Initialized DarknetDataPreparer for {self.model_name.upper()} "
            f"→ base_dir={self.base_dir}/{self.model_name}, darknet_data_dir={self.darknet_data_dir}"
        )

    # ----------------------------------------------------
    # 🔹 Public API
    # ----------------------------------------------------
    def prepare(self):
        self.logger.info(f"Preparing Darknet dataset for {self.model_name.upper()} → {self.darknet_data_dir}")
        self._generate_split_lists()
        class_names = self._get_class_names()
        self._generate_obj_files(class_names)
        self.logger.info("✅ Darknet dataset preparation complete.")
        return {
            "train_txt": str(self.darknet_data_dir / "train.txt"),
            "valid_txt": str(self.darknet_data_dir / "valid.txt"),
            "test_txt": str(self.darknet_data_dir / "test.txt"),
            "obj_data": str(self.darknet_data_dir / "obj.data"),
            "obj_names": str(self.darknet_data_dir / "obj.names"),
        }

    # ----------------------------------------------------
    # 🔹 Step 1: 이미지 리스트 생성
    # ----------------------------------------------------
    def _generate_split_lists(self):
        exts = (".jpg", ".jpeg", ".png", ".bmp")

        for split in ["train", "valid", "test"]:
            base_dir = self.base_dir / self.model_name / split
            if not base_dir.exists():
                self.logger.warning(f"⚠️ Split folder missing: {base_dir}")
                continue

            img_dir = base_dir / "images" if (base_dir / "images").exists() else base_dir
            output_file = self.darknet_data_dir / f"{split}.txt"

            images = [
                str(p.resolve()) for p in sorted(img_dir.glob("*"))
                if p.suffix.lower() in exts
            ]
            if not images:
                raise ValueError(f"No images found in {img_dir}")

            output_file.write_text("\n".join(images) + "\n", encoding="utf-8")
            self.logger.info(f"  └─ [{split}] {len(images)} images listed → {output_file.name}")

    # ----------------------------------------------------
    # 🔹 Step 2: 클래스명 추출
    # ----------------------------------------------------
    def _get_class_names(self):
        """Load class names from _darknet.labels or _classes.txt."""
        candidates = [
            self.base_dir / self.model_name / "train" / "_darknet.labels",
            self.base_dir / "_classes.txt",
        ]
        for path in candidates:
            if path.exists():
                class_names = [c.strip() for c in path.read_text(encoding="utf-8").splitlines() if c.strip()]
                self.logger.info(f"  └─ Loaded {len(class_names)} classes from {path.name}: {class_names}")
                return class_names
        raise FileNotFoundError("❌ No _darknet.labels or _classes.txt found.")

    # ----------------------------------------------------
    # 🔹 Step 3: obj.data / obj.names 생성
    # ----------------------------------------------------
    def _generate_obj_files(self, class_names):
        obj_data = self.darknet_data_dir / "obj.data"
        obj_names = self.darknet_data_dir / "obj.names"

        num_classes = len(class_names)
        backup_dir = Path(f"checkpoints/yolo_cropper/{self.model_name}")
        backup_dir.mkdir(parents=True, exist_ok=True)

        obj_data_content = (
            f"classes = {num_classes}\n"
            f"train = data/train.txt\n"
            f"valid = data/valid.txt\n"
            f"names = data/obj.names\n"
            f"backup = {backup_dir.resolve()}\n"
        )

        obj_data.write_text(obj_data_content, encoding="utf-8")
        obj_names.write_text("\n".join(class_names) + "\n", encoding="utf-8")

        self.logger.info(f"  └─ obj.data / obj.names created ({num_classes} classes)")
        self.logger.info(f"  └─ Backup path set to: {backup_dir.resolve()}")

    # ----------------------------------------------------
    # 🔹 Step 4: backup 경로 갱신
    # ----------------------------------------------------
    def update_backup_path(self, backup_dir: str):
        """Update the backup directory path in obj.data."""
        obj_data_path = self.darknet_data_dir / "obj.data"
        if not obj_data_path.exists():
            raise FileNotFoundError(f"obj.data not found: {obj_data_path}")

        lines = obj_data_path.read_text(encoding="utf-8").splitlines()
        new_lines = [
            f"backup = {backup_dir}" if line.strip().startswith("backup") else line
            for line in lines
        ]
        obj_data_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        self.logger.info(f"  └─ [updated] backup path → {backup_dir}")
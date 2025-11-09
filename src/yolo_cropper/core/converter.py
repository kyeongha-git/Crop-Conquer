#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
converter.py
------------
Unified YOLO result converter (Config-driven)
- Aggregates YOLOv5/YOLOv8 detection results (.txt) from multiple detect folders
  (e.g., repair + replace) into one unified result.json file.

Usage (debug):
    python src/yolo_cropper/core/converter.py --config utils/config.yaml
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging
from utils.load_config import load_yaml_config


# -------------------------------------------------------------
# 폴더 → 클래스 매핑
# -------------------------------------------------------------
FOLDER_TO_CLASS = {
    "repair": {"id": 0, "name": "repair"},
    "replace": {"id": 1, "name": "replace"},
}


def infer_class_from_folder(path: Path) -> Dict[str, Any]:
    """경로명에서 repair / replace 추론 (부분 문자열 기반)"""
    parts = [p.lower() for p in path.parts]
    for key, value in FOLDER_TO_CLASS.items():
        if any(key in part for part in parts):
            return value
    return {"id": -1, "name": "unknown"}


# =============================================================
# 🔹 Main Converter Class
# =============================================================
class YOLOConverter:
    """Aggregates YOLO detection (.txt) results into unified JSON (config-driven)."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.YOLOConverter")

        # --------------------------------------------------------
        # Config 해석
        # --------------------------------------------------------
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.main_cfg = self.yolo_cropper_cfg.get("main", {})
        self.dataset_cfg = self.yolo_cropper_cfg.get("dataset", {})

        # --------------------------------------------------------
        # 모델 및 경로
        # --------------------------------------------------------
        self.model_name = self.main_cfg.get("model_name", "yolov5").lower()
        self.detect_root = Path(self.dataset_cfg.get("detect_dir", "runs/detect")).resolve()
        self.output_json = Path(
            f"{self.dataset_cfg.get('results_dir', 'outputs/json_results')}/{self.model_name}/result.json"
        ).resolve()
        self.data_root = Path(self.main_cfg.get("input_dir", "data/original")).resolve()

        self.logger.info(f"Initialized YOLOConverter ({self.model_name.upper()})")
        self.logger.debug(f" - Detect Root : {self.detect_root}")
        self.logger.debug(f" - Output JSON : {self.output_json}")
        self.logger.debug(f" - Data Root   : {self.data_root}")

    # --------------------------------------------------------
    def _parse_detect_folder(self, detect_dir: Path, frame_start: int = 1) -> Tuple[List[Dict[str, Any]], int]:
        """하나의 detect 폴더에서 .txt detection 결과 파싱"""
        label_dir = detect_dir / "labels"
        if not label_dir.exists():
            self.logger.warning(f"[!] Skipping {detect_dir} (no labels/ folder)")
            return [], frame_start

        folder_class = infer_class_from_folder(detect_dir)
        class_id, class_name = folder_class["id"], folder_class["name"]

        results = []
        frame_id = frame_start

        for label_file in sorted(label_dir.glob("*.txt")):
            with open(label_file, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            base = label_file.stem
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = detect_dir / f"{base}{ext}"
                if cand.exists():
                    img_path = cand.resolve()
                    break

            # 원본 경로 복원 (데이터셋 기준)
            orig_path = self.data_root / class_name / f"{base}.png"
            if not orig_path.exists():
                for ext in [".jpg", ".jpeg"]:
                    cand = self.data_root / class_name / f"{base}{ext}"
                    if cand.exists():
                        orig_path = cand
                        break

            objects = []
            for ln in lines:
                parts = ln.split()
                if len(parts) not in (5, 6):
                    continue
                x, y, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) == 6 else None

                objects.append({
                    "class_id": class_id,
                    "name": class_name,
                    "relative_coordinates": {
                        "center_x": x,
                        "center_y": y,
                        "width": w,
                        "height": h
                    },
                    "confidence": conf
                })

            results.append({
                "frame_id": frame_id,
                "filename": str(orig_path.resolve()),
                "objects": objects
            })
            frame_id += 1

        self.logger.info(f"[+] Parsed {detect_dir.name} → {len(results)} frames")
        return results, frame_id

    # --------------------------------------------------------
    def run(self):
        """Aggregate all YOLO detection results to one JSON."""
        detect_dirs = sorted([
            p for p in self.detect_root.iterdir()
            if p.is_dir() and any(k in p.name for k in ("repair", "replace"))
        ])

        if not detect_dirs:
            raise FileNotFoundError(f"❌ No detect folders found in {self.detect_root}")

        all_results = []
        frame_id = 1

        for detect_dir in detect_dirs:
            results, frame_id = self._parse_detect_folder(detect_dir, frame_start=frame_id)
            all_results.extend(results)

        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        self.logger.info(f"[✓] Aggregated results saved → {self.output_json}")
        self.logger.info(f"   - Total frames: {len(all_results)}")


# -------------------------------------------------------------
# ✅ CLI Debug Entry
# -------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Config-driven YOLO Converter")
    parser.add_argument("--config", type=str, default="utils/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    setup_logging("logs/yolo_cropper")
    cfg = load_yaml_config(args.config)

    converter = YOLOConverter(config=cfg)
    converter.run()

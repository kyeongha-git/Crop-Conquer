#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict.py
----------
DarknetPredictor (Config-driven)

- Supports YOLOv2 / YOLOv4 inference
- Reads Darknet paths and options from injected config (not CLI)
- Uses structured logging instead of print
"""

import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class DarknetPredictor:
    """Handles YOLOv2 / YOLOv4 inference using Darknet binary."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.DarknetPredictor")

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
        self.data_dir = self.darknet_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.input_dir = Path(self.main_cfg.get("input_dir", "data/original"))
        self.output_dir = Path(self.dataset_cfg.get("results_dir", "outputs/json_results"))
        self.saved_model_dir = Path(self.dataset_cfg.get("saved_model_dir", "saved_model/yolo_cropper"))
        self.model_name = self.main_cfg.get("model_name", "yolov2").lower()

        self.logger.info(f"Initialized DarknetPredictor for {self.model_name.upper()}")
        self.logger.debug(f"Darknet dir : {self.darknet_dir}")
        self.logger.debug(f"Data dir    : {self.data_dir}")

    # --------------------------------------------------------
    # 🔹 List all images recursively
    # --------------------------------------------------------
    def _list_images(self, input_dir: Path):
        """repair/replace 등 하위 폴더까지 포함한 모든 이미지 탐색"""
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        image_paths = [str(p.resolve()) for p in input_dir.rglob("*") if p.suffix.lower() in exts]

        if not image_paths:
            raise FileNotFoundError(f"No images found recursively under {input_dir}")

        self.logger.info(f"Found {len(image_paths)} images under {input_dir}")
        return sorted(image_paths)

    # --------------------------------------------------------
    # 🔹 Main Prediction
    # --------------------------------------------------------
    def run(self):
        """Run YOLO detection using Darknet"""
        input_dir = self.input_dir
        predict_path = self.data_dir / "predict.txt"

        # 1️⃣ 이미지 리스트 생성
        images = self._list_images(input_dir)
        predict_path.write_text("\n".join(images) + "\n", encoding="utf-8")
        self.logger.info(f"[✓] predict.txt 생성 완료 ({len(images)}개 이미지) → {predict_path}")

        # 2️⃣ 경로 설정
        cfg_path = f"cfg/{self.model_name}-obj.cfg"
        obj_data = "data/obj.data"
        weights_path = Path(f"{self.saved_model_dir}/{self.model_name}.weights").resolve()
        internal_result = (self.data_dir / f"result_{self.model_name}.json").resolve()
        external_result = Path(f"{self.output_dir}/{self.model_name}/result.json")
        external_result.parent.mkdir(parents=True, exist_ok=True)

        # 3️⃣ YOLO 실행 명령어
        command = (
            f'./darknet detector test {obj_data} {cfg_path} {weights_path} '
            f'-thresh 0.25 -dont_show -ext_output -out {internal_result} < data/{predict_path.name}'
        )

        self.logger.info(f"🚀 YOLO Detection 시작 ({self.model_name.upper()})")
        self.logger.debug(f"[CMD] {command}")

        process = subprocess.run(["bash", "-lc", command], cwd=self.darknet_dir, shell=False)

        # 4️⃣ Darknet의 반환코드 처리
        if process.returncode not in (0, 1):
            raise RuntimeError(f"❌ Darknet detection failed (code: {process.returncode})")
        elif process.returncode == 1:
            self.logger.warning("Darknet exited with code 1 (non-fatal). Detection likely succeeded.")

        # 5️⃣ 결과 복사
        if internal_result.exists():
            shutil.copy2(internal_result, external_result)
            self.logger.info(f"[✓] Copied result → {external_result.resolve()}")
        else:
            self.logger.warning("[!] result.json not found in darknet/data folder!")

        # 6️⃣ predict.txt를 outputs/json_results에도 복사
        predict_copy_path = self.output_dir / "predict.txt"
        try:
            shutil.copy2(predict_path, predict_copy_path)
            self.logger.info(f"[✓] Copied predict.txt → {predict_copy_path.resolve()}")
        except Exception as e:
            self.logger.warning(f"[!] Failed to copy predict.txt → {e}")

        return str(external_result.resolve()), str(predict_path)
    

# --------------------------------------------------------
# ✅ Optional Debug Entry
# --------------------------------------------------------
if __name__ == "__main__":
    from utils.load_config import load_yaml_config
    from utils.logging import setup_logging
    import argparse

    parser = argparse.ArgumentParser(description="Debug Darknet Predictor")
    parser.add_argument("--config", type=str, default="utils/config.yaml")
    args = parser.parse_args()

    setup_logging("logs/yolo_cropper")
    cfg = load_yaml_config(args.config)

    predictor = DarknetPredictor(config=cfg)
    result_json, predict_txt = predictor.run()

    print("\n=== ✅ Prediction Result ===")
    print(f"Result JSON : {result_json}")
    print(f"Predict TXT : {predict_txt}")

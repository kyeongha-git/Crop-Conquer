#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
restore_crop.py
-------------------
패딩된 이미지를 기반으로 생성형 AI 결과(1024x1024)를 원본 크기로 복원하는 모듈.

구조적 특징:
- config는 main.py에서 읽어서 각 인자에 명시적으로 전달
- Logging 기반
- ImagePadder와 동일한 스타일의 초기화 구조 유지
"""

import os
import cv2
import json
import shutil
from pathlib import Path
from typing import List, Optional
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]  # Research/
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging


class RestoreCropper:
    """패딩 메타데이터를 기반으로 1024x1024 이미지를 원본 크기로 복원"""

    def __init__(
        self,
        input_dir: str,          # generated_image_padded
        output_dir: str,         # restored_image
        meta_dir: str,           # only_annotation_image_padded
        categories: Optional[List[str]] = None,
        metadata_name: str = "padding_info.json",
    ):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("RestoreCrop")

        # 경로 및 설정
        self.input_root = Path(input_dir)
        self.meta_root = Path(meta_dir)
        self.output_root = Path(output_dir)
        self.categories = categories or ["repair", "replace"]
        self.meta_name = metadata_name

        self.logger.info(f"📂 입력 폴더: {self.input_root}")
        self.logger.info(f"📜 메타데이터 폴더: {self.meta_root}")
        self.logger.info(f"💾 출력 폴더: {self.output_root}")

    # ============================================================
    # 🔧 내부 유틸
    # ============================================================
    def _load_metadata(self, meta_path: Path) -> Optional[dict]:
        """padding_info.json Load"""
        if not meta_path.exists():
            self.logger.warning(f"⚠️ 메타데이터 없음: {meta_path}")
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {os.path.splitext(k)[0]: v for k, v in data.items()}
        except Exception as e:
            self.logger.error(f"❌ 메타데이터 로드 실패 ({meta_path}): {e}")
            return None

    def _restore_single_image(self, img_path: Path, meta: dict, save_path: Path) -> bool:
        """단일 이미지를 원본 크기로 복원"""
        img = cv2.imread(str(img_path))
        if img is None:
            self.logger.warning(f"⚠️ 이미지 읽기 실패: {img_path.name}")
            return False

        h_orig, w_orig = meta["orig_size"]
        top, left = meta["pad_info"]["top"], meta["pad_info"]["left"]
        roi = img[top:top + h_orig, left:left + w_orig]

        success = cv2.imwrite(str(save_path), roi)
        if success:
            self.logger.info(f"✅ 복원 완료: {save_path.name}")
            return True
        else:
            self.logger.error(f"❌ 저장 실패: {save_path.name}")
            return False

    # ============================================================
    # 🚀 Public API
    # ============================================================
    def run(self):
        """카테고리별 복원 프로세스 실행"""
        if not self.input_root.exists():
            raise FileNotFoundError(f"❌ 입력 폴더를 찾을 수 없습니다: {self.input_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)
        total_restored = 0

        for category in self.categories:
            in_dir = self.input_root / category
            meta_path = self.meta_root / category / self.meta_name
            out_dir = self.output_root / category
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                self.logger.warning(f"⚠️ 입력 폴더 없음: {in_dir}")
                continue

            metadata = self._load_metadata(meta_path)
            if not metadata:
                continue

            restored_count = 0
            for file in sorted(os.listdir(in_dir)):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                name = os.path.splitext(file)[0]
                input_path = in_dir / file
                save_path = out_dir / file

                if name not in metadata:
                    shutil.copy(input_path, save_path)
                    self.logger.info(f"🔁 {file}: 패딩 생략 이미지 복사 완료")
                    continue

                success = self._restore_single_image(input_path, metadata[name], save_path)
                restored_count += int(success)

            self.logger.info(f"✅ {category}: {restored_count}개 복원 완료 → {out_dir}")
            total_restored += restored_count

        self.logger.info(f"🎉 전체 복원 완료 ({total_restored}개 파일)")

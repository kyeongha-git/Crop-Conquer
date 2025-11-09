#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
image_padding.py
----------------
입력 이미지를 target_size(기본 1024x1024)에 맞게 중앙 정렬 패딩하는 모듈.
- Logging 기반
- Class 구조화 (SRP)
- JSON 메타데이터 기록
- 음수 패딩 및 이미지 로드 실패 안전 처리
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


class ImagePadder:
    """이미지 크기를 중앙 기준으로 패딩하고 padding 정보 메타데이터를 기록"""

    DEFAULT_PADDING_COLOR = (0, 0, 0)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        categories: Optional[List[str]] = None,
        target_size: int = 1024,
        metadata_name: str = "padding_info.json",
    ):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("ImagePadder")

        self.input_root = Path(input_dir)
        self.output_root = Path(output_dir)
        self.categories = categories or ["repair", "replace"]
        self.target_size = target_size
        self.metadata_name = metadata_name
        self.padding_color = self.DEFAULT_PADDING_COLOR

        self.logger.info(f"📂 입력 경로: {self.input_root}")
        self.logger.info(f"💾 출력 경로: {self.output_root}")
        self.logger.info(f"🎨 타겟 해상도: {self.target_size}")

    # ============================================================
    # 🔧 내부 함수: 이미지 패딩
    # ============================================================
    def _pad_image(self, image_path: Path, save_path: Path) -> Optional[dict]:
        """이미지를 중앙 패딩하고 padding 정보를 반환"""
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.error(f"⚠️ {image_path.name}: 이미지 로드 실패 (경로 또는 형식 문제)")
            return None

        h, w = img.shape[:2]

        # target_size보다 크면 skip
        if h >= self.target_size and w >= self.target_size:
            self.logger.info(f"⏩ {image_path.name}: 이미 {self.target_size}px 이상 → 복사만 수행")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy(str(image_path), str(save_path))
            except Exception as e:
                self.logger.error(f"❌ {image_path.name}: 복사 실패 ({e})")
            return None

        # ✅ 음수 패딩 방지
        top = max(0, (self.target_size - h) // 2)
        bottom = max(0, self.target_size - h - top)
        left = max(0, (self.target_size - w) // 2)
        right = max(0, self.target_size - w - left)

        try:
            padded = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=self.padding_color
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(save_path), padded)

            if not success:
                self.logger.error(f"❌ {image_path.name}: 저장 실패")
                return None

            return {
                "orig_size": [h, w],
                "pad_info": {"top": top, "left": left, "bottom": bottom, "right": right},
            }

        except Exception as e:
            self.logger.error(f"⚠️ {image_path.name}: 패딩 중 오류 ({e})")
            return None

    # ============================================================
    # 🚀 Public API
    # ============================================================
    def run(self):
        """카테고리별 이미지 패딩 수행"""
        if not self.input_root.exists():
            raise FileNotFoundError(f"❌ 입력 폴더를 찾을 수 없습니다: {self.input_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)

        for category in self.categories:
            in_dir = self.input_root / category
            out_dir = self.output_root / category
            meta_path = out_dir / self.metadata_name

            if not in_dir.exists():
                self.logger.warning(f"⚠️ 폴더 없음: {in_dir}")
                continue

            self.logger.info(f"🧩 카테고리 처리 중: {category}")
            metadata = {}

            for file in sorted(os.listdir(in_dir)):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                input_path = in_dir / file
                save_path = out_dir / file
                info = self._pad_image(input_path, save_path)
                if info:
                    metadata[file] = info

            # ✅ 메타데이터 저장
            if metadata:
                try:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=4, ensure_ascii=False)
                    self.logger.info(f"✅ Padding 완료 → {out_dir}")
                    self.logger.info(f"🧾 메타데이터 저장 → {meta_path}")
                except Exception as e:
                    self.logger.error(f"❌ 메타데이터 저장 실패 ({meta_path}): {e}")
            else:
                self.logger.info(f"⚪ {category}: 새로 생성된 패딩 없음.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_annotation.py
-------------------
Gemini API를 사용해 차량 유리 이미지의 마킹/비본질적 요소를 제거하고
편향 없는(bias-free) 이미지를 생성하는 클래스 모듈.

리팩토링 포인트:
- config.yaml 직접 접근 ❌ → 상위 컨트롤러에서 주입받음
- Logging 기반 구조
- Class 단일 책임(SRP)
- 테스트 및 재사용 용이
"""

import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from PIL import Image
from google import genai
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]  # Research/
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger, setup_logging


# ============================================================
# 🔐 Gemini Client 초기화 함수
# ============================================================
def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """환경변수에서 Gemini API Key를 불러와 클라이언트 생성"""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("❌ GEMINI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
    try:
        return genai.Client(api_key=key)
    except Exception as e:
        raise RuntimeError(f"❌ Gemini 클라이언트 초기화 실패: {e}")


# ============================================================
# 🧩 CleanAnnotation 클래스
# ============================================================
class CleanAnnotation:
    """
    Gemini 기반 이미지 annotation 제거기
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model: str,
        prompt: str,
        categories: Optional[List[str]] = None,
        test_mode: bool = False,
        test_limit: int = 3,
        client: Optional[genai.Client] = None,
    ):
        setup_logging("logs/annotation_cleaner")
        self.logger = get_logger("CleanAnnotation")

        # 기본 설정
        self.input_root = Path(input_dir)
        self.output_root = Path(output_dir)
        self.categories = categories or ["repair", "replace"]
        self.model = model
        self.prompt = prompt
        self.test_mode = test_mode
        self.test_limit = test_limit

        # Gemini 클라이언트
        self.client = client or get_gemini_client()

        self.logger.info(f"📂 입력 경로: {self.input_root}")
        self.logger.info(f"💾 출력 경로: {self.output_root}")
        self.logger.info(f"🧩 모델: {self.model}")

    # ============================================================
    # 🔧 내부 유틸
    # ============================================================
    def _generate_clean_image(self, image_path: Path, output_path: Path) -> bool:
        """Gemini API를 사용해 단일 이미지에서 annotation 제거"""
        try:
            image = Image.open(image_path)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[self.prompt, image],
            )

            # 이미지 응답 처리
            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    gen_img = Image.open(BytesIO(part.inline_data.data))
                    gen_img.save(output_path)
                    self.logger.info(f"✅ 저장 완료: {output_path.name}")
                    return True
                elif getattr(part, "text", None):
                    self.logger.warning(f"📝 텍스트 응답 ({image_path.name}): {part.text}")
                    return False

        except Exception as e:
            self.logger.error(f"⚠️ {image_path.name} 처리 중 오류 발생: {e}")
            return False
        return False

    # ============================================================
    # 🚀 Public API
    # ============================================================
    def run(self):
        """카테고리별 폴더를 순회하며 annotation 제거 수행"""
        if not self.input_root.exists():
            raise FileNotFoundError(f"❌ 입력 폴더를 찾을 수 없습니다: {self.input_root}")

        self.output_root.mkdir(parents=True, exist_ok=True)
        processed_count = 0

        for category in self.categories:
            in_dir = self.input_root / category
            out_dir = self.output_root / category
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                self.logger.warning(f"⚠️ 폴더 없음: {in_dir}")
                continue

            image_files = [f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            for filename in image_files:
                input_path = in_dir / filename
                output_path = out_dir / filename

                if output_path.exists():
                    self.logger.info(f"⏩ {filename} 이미 존재 → 건너뜀")
                    continue

                success = self._generate_clean_image(input_path, output_path)
                processed_count += int(success)

                if self.test_mode and self.test_limit and processed_count >= self.test_limit:
                    self.logger.info(f"🧪 테스트 제한 도달 ({self.test_limit}장). 중단.")
                    return

        self.logger.info(f"🎉 모든 이미지 처리 완료. 총 {processed_count}개 파일.")

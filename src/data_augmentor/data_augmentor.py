#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_augmentor.py
-----------------
데이터 증강 전체 파이프라인 실행 스크립트.

Features:
- config.yaml 기반 Split + Augmentation 자동화
- CLI Entry Point (input_dir, output_dir, config_path)
- utils.load_config / utils.logging 통합
- split_dataset + augment_dataset 연동
"""

import argparse
import shutil
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from utils.load_config import load_yaml_config
from utils.logging import setup_logging, get_logger
from src.data_augmentor.core.split_dataset import split_dataset
from src.data_augmentor.core.augment_dataset import balance_augmentation


class DataAugmentor:
    """데이터셋 Split + Augmentation 통합 처리 클래스"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path (str): YAML 설정 파일 경로
            input_dir (str): 입력 데이터 경로 (예: data/original)
            output_dir (str, optional): 출력 경로 (기본값은 입력과 동일)
        """
        setup_logging("logs/data_augmentor")
        self.logger = get_logger("DataAugmentor")

        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)

        # 설정 섹션
        augmentor_cfg = self.cfg.get("data_augmentor", {})
        self.data_cfg = augmentor_cfg.get("data", {})
        self.split_cfg = augmentor_cfg.get("split", {})
        self.aug_cfg = augmentor_cfg.get("augmentation", {})

        # 경로 설정
        self.input_dir = Path(self.data_cfg.get("input_dir", "data/original"))
        self.output_dir = Path(self.data_cfg.get("output_dir", "data/original"))

        self.logger.info(f"✅ Config loaded from: {self.config_path}")
        self.logger.info(f"📂 Input dir : {self.input_dir}")
        self.logger.info(f"📁 Output dir: {self.output_dir}")

    # -----------------------------
    # Split 단계
    # -----------------------------
    def _run_split(self):
        self.logger.info("\n🧩 [1/2] Split 단계 실행 중...")
        split_dataset(
            data_dir=self.input_dir,
            output_dir=self.output_dir,
            split_cfg=self.split_cfg,
        )
        self.logger.info("✅ Split 완료!")

    def _cleanup_original_classes(self):
        """train/valid/test 분리 후 남은 원본 class 디렉토리 정리"""
        self.logger.info("\n🧹 [Cleanup] 원본 class 디렉토리 삭제 중...")
        for cls in ["repair", "replace"]:
            target = self.output_dir / cls
            if target.exists():
                try:
                    shutil.rmtree(target)
                    self.logger.info(f"🗑️  {target} 삭제 완료")
                except Exception as e:
                    self.logger.warning(f"⚠️  {target} 삭제 실패: {e}")
        self.logger.info("✅ Cleanup 완료!")

    # -----------------------------
    # Augmentation 단계
    # -----------------------------
    def _run_augment(self):
        if not self.aug_cfg.get("enable", False):
            self.logger.info("\n🚫 [2/2] 증강 비활성화됨 (config.yaml 설정에 따라 건너뜀)")
            return

        self.logger.info("\n🧠 [2/2] 클래스 불균형 증강 실행 중...")
        balance_augmentation(self.output_dir, self.aug_cfg)
        self.logger.info("✅ 증강 완료!")

    # -----------------------------
    # 전체 실행
    # -----------------------------
    def run(self):
        """Split + Augment 전체 파이프라인 실행"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"❌ 입력 데이터 경로가 존재하지 않습니다: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("\n🎯 [DataAugmentor] 파이프라인 시작")
        self.logger.info(f" - Split 비율: {self.split_cfg}")
        self.logger.info(f" - Augment 설정: {'활성화됨' if self.aug_cfg.get('enable', False) else '비활성화됨'}")

        self._run_split()
        self._cleanup_original_classes()
        self._run_augment()

        self.logger.info("\n🎉 전체 파이프라인 완료!")


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataAugmentor 파이프라인 실행")
    parser.add_argument("--config", default="./utils/config.yaml")
    args = parser.parse_args()

    augmentor = DataAugmentor(
        config_path=args.config
    )
    augmentor.run()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_dataset.py
-----------------
데이터셋을 train / valid / test로 분할하는 모듈.

Features:
- dataset_type / dataset_version 자동 경로 처리
- config.yaml 기반 비율 로드
- utils.load_config / utils.logging 통합
- reproducibility 보장 (seed 고정)
"""

import random
import shutil
from pathlib import Path
from typing import Dict, List

from utils.logging import get_logger


# -----------------------------
# Core Functions
# -----------------------------
def get_images(class_path: Path) -> List[Path]:
    """클래스 폴더 내 이미지 파일 목록 반환"""
    valid_ext = (".jpg", ".jpeg", ".png")
    return sorted([p for p in class_path.iterdir() if p.suffix.lower() in valid_ext])


def make_splits(
    images: List[Path],
    train_ratio: float,
    valid_ratio: float,
    seed: int = 42
) -> Dict[str, List[Path]]:
    """이미지 리스트를 train/valid/test로 분할"""
    random.seed(seed)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    return {
        "train": images[:train_end],
        "valid": images[train_end:valid_end],
        "test": images[valid_end:],
    }


def copy_images(
    class_name: str,
    class_path: Path,
    output_dir: Path,
    splits: Dict[str, List[Path]],
    logger
) -> None:
    """분할된 이미지를 각 split 폴더에 복사"""
    for split_name, files in splits.items():
        split_dir = output_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for src_path in files:
            dst_path = split_dir / src_path.name
            shutil.copy2(src_path, dst_path)
        logger.info(f"📦 Copied {len(files):>4} → {split_name}/{class_name}")


def split_dataset(
    data_dir: Path,
    output_dir: Path,
    split_cfg: Dict[str, float],
    seed: int = 42,
    logger=None
) -> None:
    """클래스별로 train/valid/test 분할 수행"""
    if logger is None:
        logger = get_logger("split_dataset")

    train_ratio = split_cfg.get("train_ratio", 0.8)
    valid_ratio = split_cfg.get("valid_ratio", 0.1)
    test_ratio = split_cfg.get("test_ratio", 0.1)
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "Train/Valid/Test 비율의 합이 1이어야 합니다."

    logger.info(f"📁 데이터 분할 시작: {data_dir}")
    logger.info(f" - 출력 경로: {output_dir}")
    logger.info(f" - 비율: train={train_ratio}, valid={valid_ratio}, test={test_ratio}")

    categories = [d.name for d in data_dir.iterdir() if d.is_dir()]
    if not categories:
        logger.warning(f"⚠️ No class folders found in {data_dir}")
        return

    for class_name in categories:
        class_path = data_dir / class_name
        images = get_images(class_path)
        if not images:
            logger.warning(f"[⚠️] {class_name} 폴더에 이미지가 없습니다. 건너뜀.")
            continue

        splits = make_splits(images, train_ratio, valid_ratio, seed)
        copy_images(class_name, class_path, output_dir, splits, logger)

        logger.info(
            f"[{class_name}] ✅ "
            f"train={len(splits['train'])}, "
            f"valid={len(splits['valid'])}, "
            f"test={len(splits['test'])}"
        )

    logger.info("✅ 데이터 분할 완료!")
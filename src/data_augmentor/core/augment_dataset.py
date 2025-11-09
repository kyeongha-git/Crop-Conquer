#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image, ImageEnhance
import numpy as np

from utils.logging import get_logger

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


"""
augment_dataset.py
------------------
train/repair 와 train/replace 의 클래스 불균형을 자동 감지하고
config.yaml 기반 파라미터로 증강을 수행하는 core 모듈.

Usage:
    from data_augmentor.core.augment_dataset import balance_augmentation
"""


# ============================================================
# 유틸리티
# ============================================================
def list_images(folder: Path):
    """폴더 내 이미지 파일 목록을 반환"""
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================
# 증강 Primitive
# ============================================================
def random_resized_crop(img: Image.Image, scale=(0.9, 1.0), ratio=(0.95, 1.05), trials=10):
    w, h = img.size
    area = w * h
    for _ in range(trials):
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        new_ratio = math.exp(random.uniform(*log_ratio))
        new_w = int(round(math.sqrt(target_area * new_ratio)))
        new_h = int(round(math.sqrt(target_area / new_ratio)))
        if 0 < new_w <= w and 0 < new_h <= h:
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            crop = img.crop((left, top, left + new_w, top + new_h))
            return crop.resize((w, h), Image.BICUBIC)

    # 실패 시 중앙 크롭
    s = clamp(scale[0], 0.0, 1.0)
    new_w, new_h = int(w * s), int(h * s)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    crop = img.crop((left, top, left + new_w, top + new_h))
    return crop.resize((w, h), Image.BICUBIC)


def random_hflip(img: Image.Image, p=0.5):
    return img.transpose(Image.FLIP_LEFT_RIGHT) if random.random() < p else img


def random_rotate(img: Image.Image, max_deg=10):
    deg = random.uniform(-max_deg, max_deg)
    return img.rotate(deg, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))


def random_translate(img: Image.Image, max_ratio=0.04):
    w, h = img.size
    tx = int(random.uniform(-max_ratio, max_ratio) * w)
    ty = int(random.uniform(-max_ratio, max_ratio) * h)
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, tx, 0, 1, ty),
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0),
    )


def color_jitter(
    img: Image.Image,
    b_range=(0.9, 1.1),
    c_range=(0.9, 1.15),
    s_range=(0.9, 1.15),
    hue_delta=0.03,
):
    if random.random() < 0.9:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*b_range))
    if random.random() < 0.9:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*c_range))
    if random.random() < 0.9:
        img = ImageEnhance.Color(img).enhance(random.uniform(*s_range))
    if hue_delta > 0 and random.random() < 0.5:
        img = img.convert("HSV")
        arr = np.array(img).astype(np.uint8)
        h_shift = int((random.uniform(-hue_delta, hue_delta)) * 255)
        arr[..., 0] = (arr[..., 0].astype(int) + h_shift) % 255
        img = Image.fromarray(arr, "HSV").convert("RGB")
    return img


# ============================================================
# 증강 파이프라인
# ============================================================
def augment_pipeline(img: Image.Image, aug_cfg: Dict) -> Image.Image:
    """config.yaml 기반 증강 파이프라인"""
    img = random_resized_crop(
        img,
        scale=tuple(aug_cfg.get("random_resized_crop", {}).get("scale", (0.9, 1.0))),
        ratio=tuple(aug_cfg.get("random_resized_crop", {}).get("ratio", (0.95, 1.05))),
    )
    img = random_hflip(img, p=aug_cfg.get("random_hflip_p", 0.5))
    img = random_rotate(img, max_deg=aug_cfg.get("random_rotate_deg", 10))
    img = random_translate(img, max_ratio=aug_cfg.get("random_translate_ratio", 0.04))
    img = color_jitter(
        img,
        b_range=tuple(aug_cfg.get("brightness_range", (0.9, 1.1))),
        c_range=tuple(aug_cfg.get("contrast_range", (0.9, 1.15))),
        s_range=tuple(aug_cfg.get("saturation_range", (0.9, 1.15))),
        hue_delta=aug_cfg.get("hue_delta", 0.03),
    )
    return img


# ============================================================
# 클래스 밸런싱 증강
# ============================================================
def _augment_until_equal(
    src_dir: Path, target_count: int, aug_cfg: Dict, seed: int = 42, logger=None
):
    """부족한 클래스 이미지를 target_count까지 증강"""
    if logger is None:
        logger = get_logger("augment_dataset")

    random.seed(seed)
    files = list_images(src_dir)
    cur = len(files)
    save_quality = 95
    i = 0

    logger.info(f"Augmenting {src_dir.name}: {cur} → {target_count}")

    while cur < target_count:
        src = random.choice(files)
        try:
            with Image.open(src).convert("RGB") as img:
                aug = augment_pipeline(img, aug_cfg)
                out_name = f"{src.stem}_aug_{i:05d}.jpg"
                out_path = src_dir / out_name
                if out_path.exists():
                    i += 1
                    continue
                aug.save(out_path, quality=save_quality)
                cur += 1
                i += 1
        except Exception as e:
            logger.warning(f"Skip {src.name}: {e}")

    logger.info(f"🎉 완료! {src_dir.name} = {cur}장 (균형 맞춤)")


def balance_augmentation(root_dir: Path, aug_cfg: Dict, seed: int = 42, logger=None):
    """train/repair vs train/replace 클래스 불균형 자동 감지 및 증강"""
    if logger is None:
        logger = get_logger("augment_dataset")

    train_dir = root_dir / "train"
    train_repair = train_dir / "repair"
    train_replace = train_dir / "replace"

    if not train_repair.exists() or not train_replace.exists():
        raise FileNotFoundError(f"경로 확인 필요: {train_repair} 또는 {train_replace}")

    repair_count = len(list_images(train_repair))
    replace_count = len(list_images(train_replace))

    logger.info(f"[Counts] train/repair={repair_count}, train/replace={replace_count}")

    if repair_count == replace_count:
        logger.info("✅ 두 클래스가 이미 균형 상태입니다.")
        return

    smaller_dir = train_repair if repair_count < replace_count else train_replace
    target_count = max(repair_count, replace_count)

    logger.info(
        f"📈 '{smaller_dir.name}' 클래스가 부족합니다. ({len(list_images(smaller_dir))} → {target_count})"
    )
    _augment_until_equal(smaller_dir, target_count, aug_cfg, seed=seed, logger=logger)

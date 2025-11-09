#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import tempfile
from PIL import Image, ImageChops
import numpy as np
import pytest
import random

from src.data_augmentor.core.augment_dataset import (
    list_images,
    clamp,
    random_resized_crop,
    random_hflip,
    random_rotate,
    random_translate,
    color_jitter,
    augment_pipeline,
    balance_augmentation,
)
from src.data_augmentor.core.split_dataset import split_dataset


"""
test_split_dataset.py
----------------------
Unit tests for `split_dataset.py`.

Test Goals:
- Verify dataset is split into train/valid/test with correct ratios
- Ensure folder structure (train/valid/test per class) is created
- Confirm image files are correctly copied
"""

@pytest.fixture
def dummy_dataset(tmp_path):
    """
    Create a dummy dataset for testing.
    Structure:
        tmp_path/
            repair/
                img_0.jpg ... img_9.jpg
            replace/
                img_0.jpg ... img_9.jpg
    """
    base = tmp_path
    for cls in ["repair", "replace"]:
        cls_dir = base / cls
        cls_dir.mkdir(parents=True)
        for i in range(10):
            (cls_dir / f"img_{i}.jpg").write_text(f"fake image {i}")
    return base


@pytest.fixture
def dummy_split_config():
    """Return a simple split configuration"""
    return {"train_ratio": 0.6, "valid_ratio": 0.2, "test_ratio": 0.2}


def test_split_dataset_creates_splits(dummy_dataset, dummy_split_config, tmp_path):
    """Check if split folders are correctly created and match expected counts"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    split_dataset(
        data_dir=dummy_dataset,
        output_dir=output_dir,
        split_cfg=dummy_split_config,
        seed=42,
    )

    # 각 split/class 폴더 존재 확인
    for split in ["train", "valid", "test"]:
        for cls in ["repair", "replace"]:
            target_dir = output_dir / split / cls
            assert target_dir.exists(), f"Missing folder: {target_dir}"

    # --- 실제 개수 계산 ---
    def count_images(split):
        return sum(len(list((output_dir / split / cls).glob("*.jpg"))) for cls in ["repair", "replace"])

    total_imgs = sum(len(list((dummy_dataset / cls).glob("*.jpg"))) for cls in ["repair", "replace"])
    train_imgs = count_images("train")
    valid_imgs = count_images("valid")
    test_imgs = count_images("test")

    # --- 기대값 계산 (비율 × 총합) ---
    expected_train = int(total_imgs * dummy_split_config["train_ratio"])
    expected_valid = int(total_imgs * dummy_split_config["valid_ratio"])
    expected_test = total_imgs - expected_train - expected_valid  # rounding 보정

    # --- 정량 비교 ---
    assert train_imgs == expected_train, f"Train split mismatch: {train_imgs} vs {expected_train}"
    assert valid_imgs == expected_valid, f"Valid split mismatch: {valid_imgs} vs {expected_valid}"
    assert test_imgs == expected_test, f"Test split mismatch: {test_imgs} vs {expected_test}"

    # 전체 합 검증
    assert total_imgs == train_imgs + valid_imgs + test_imgs, "Split total mismatch"



def test_split_dataset_reproducibility(dummy_dataset, dummy_split_config, tmp_path):
    """Ensure deterministic splitting with fixed seed"""
    output_1 = tmp_path / "out1"
    output_2 = tmp_path / "out2"

    split_dataset(dummy_dataset, output_1, dummy_split_config, seed=123)
    split_dataset(dummy_dataset, output_2, dummy_split_config, seed=123)

    train_1 = sorted((output_1 / "train" / "repair").iterdir())
    train_2 = sorted((output_2 / "train" / "repair").iterdir())

    # same seed → identical results
    assert [p.name for p in train_1] == [p.name for p in train_2]


def test_split_dataset_empty_class(tmp_path, dummy_split_config, caplog):
    """Handle empty class folder gracefully (warning, not crash)"""
    empty_class_dir = tmp_path / "repair"
    empty_class_dir.mkdir()
    (tmp_path / "replace").mkdir()

    split_dataset(tmp_path, tmp_path / "output", dummy_split_config, seed=1)
    assert "이미지가 없습니다" in caplog.text or "No class folders" in caplog.text


def test_split_dataset_invalid_ratios(dummy_dataset, tmp_path):
    """Invalid ratios should raise AssertionError"""
    bad_cfg = {"train_ratio": 0.5, "valid_ratio": 0.5, "test_ratio": 0.2}
    with pytest.raises(AssertionError):
        split_dataset(dummy_dataset, tmp_path / "out", bad_cfg)


"""
test_augment_dataset.py
-----------------------
Unit & Integration tests for `augment_dataset.py`.

Test Goals:
- ✅ Verify augmentation primitives (crop, flip, rotate, translate, color jitter) work as intended
- ✅ Ensure augment_pipeline() combines all transformations without error
- ✅ Validate balance_augmentation() correctly detects class imbalance and performs upsampling
- ✅ Confirm utility functions (list_images, clamp) behave correctly
- 🧩 Check reproducibility and output validity for randomized operations
"""

# ============================================================
# Fixtures & Utilities
# ============================================================

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir):
    """좌우 색상 그라디언트가 있는 테스트용 비대칭 이미지"""
    img_path = temp_dir / "sample.jpg"
    img = Image.new("RGB", (64, 64))
    for x in range(64):
        for y in range(64):
            color = (x * 4, y * 2, 128)
            img.putpixel((x, y), color)
    img.save(img_path)
    return img_path


def _img_to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.uint8).copy()


def _images_different(img1: Image.Image, img2: Image.Image) -> bool:
    diff = ImageChops.difference(img1, img2)
    return diff.getbbox() is not None


def count_images_in_dir(directory: Path) -> int:
    return len([f for f in directory.glob("*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])


# ============================================================
# Unit Tests: Utility Functions
# ============================================================

def test_list_images_filters_correctly(temp_dir):
    (temp_dir / "a.jpg").touch()
    (temp_dir / "b.png").touch()
    (temp_dir / "c.txt").touch()
    files = list_images(temp_dir)
    assert all(f.suffix.lower() in [".jpg", ".png"] for f in files)
    assert len(files) == 2


def test_clamp_behaves_correctly():
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(50, 0, 10) == 10


# ============================================================
# Unit Tests: Augmentation Primitives
# ============================================================

def test_random_resized_crop_changes_composition(sample_image):
    img = Image.open(sample_image)
    cropped = random_resized_crop(img)
    assert cropped.size == img.size
    assert _images_different(img, cropped)


def test_random_hflip_flips_horizontally(sample_image):
    img = Image.open(sample_image)
    arr = _img_to_array(img)
    arr[:, :32, :] = 0
    img = Image.fromarray(arr)
    flipped = random_hflip(img, p=1.0)
    assert np.array_equal(_img_to_array(flipped)[:, 0, :], arr[:, -1, :])


def test_random_rotate_rotates_pixels(sample_image):
    img = Image.open(sample_image)
    arr_before = _img_to_array(img)
    rotated = random_rotate(img, max_deg=45)
    arr_after = _img_to_array(rotated)
    assert np.mean(arr_before != arr_after) > 0.05


def test_random_translate_shifts_image_content(sample_image):
    img = Image.open(sample_image)
    arr_before = _img_to_array(img)
    translated = random_translate(img, max_ratio=0.2)
    arr_after = _img_to_array(translated)
    assert np.mean(arr_before != arr_after) > 0.05


def test_color_jitter_modifies_pixel_values(sample_image):
    random.seed(42)
    np.random.seed(42)

    img = Image.open(sample_image)
    arr_before = _img_to_array(img)
    out = color_jitter(img)
    arr_after = _img_to_array(out)

    mean_diff = abs(arr_before.mean() - arr_after.mean())
    assert mean_diff > 0.5, f"컬러 변화가 감지되지 않음 (Δmean={mean_diff:.3f})"


# ============================================================
# Integration: augment_pipeline
# ============================================================

def test_augment_pipeline_combines_all(sample_image):
    img = Image.open(sample_image)
    dummy_cfg = {
        "random_resized_crop": {"scale": [0.9, 1.0], "ratio": [0.95, 1.05]},
        "random_hflip_p": 0.5,
        "random_rotate_deg": 15,
        "random_translate_ratio": 0.05,
        "brightness_range": [0.9, 1.1],
        "contrast_range": [0.9, 1.15],
        "saturation_range": [0.9, 1.15],
        "hue_delta": 0.03,
    }
    result = augment_pipeline(img, dummy_cfg)
    assert isinstance(result, Image.Image)
    assert result.size == img.size


# ============================================================
# Functional: balance_augmentation
# ============================================================

def test_balance_augmentation_balances_classes(temp_dir):
    """balance_augmentation()이 더 적은 클래스를 감지해 균형 맞추는지 검증"""
    train_repair = temp_dir / "train" / "repair"
    train_replace = temp_dir / "train" / "replace"
    train_repair.mkdir(parents=True, exist_ok=True)
    train_replace.mkdir(parents=True, exist_ok=True)

    # repair: 5장, replace: 2장
    for i in range(5):
        Image.new("RGB", (32, 32), (128, 128, 128)).save(train_repair / f"repair_{i}.jpg")
    for i in range(2):
        Image.new("RGB", (32, 32), (128, 128, 128)).save(train_replace / f"replace_{i}.jpg")

    dummy_cfg = {"random_hflip_p": 1.0}
    balance_augmentation(root_dir=temp_dir, aug_cfg=dummy_cfg, seed=123)

    repair_count = count_images_in_dir(train_repair)
    replace_count = count_images_in_dir(train_replace)
    assert repair_count == replace_count


def test_balance_augmentation_handles_equal_case(temp_dir):
    """이미 균형 상태이면 추가 증강이 발생하지 않아야 함"""
    train_repair = temp_dir / "train" / "repair"
    train_replace = temp_dir / "train" / "replace"
    train_repair.mkdir(parents=True, exist_ok=True)
    train_replace.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        img.save(train_repair / f"repair_{i}.jpg")
        img.save(train_replace / f"replace_{i}.jpg")

    dummy_cfg = {"random_hflip_p": 1.0}
    before = count_images_in_dir(train_repair)
    balance_augmentation(root_dir=temp_dir, aug_cfg=dummy_cfg, seed=42)
    after = count_images_in_dir(train_repair)

    assert before == after

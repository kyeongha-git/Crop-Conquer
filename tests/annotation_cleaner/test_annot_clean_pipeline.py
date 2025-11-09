#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import io
import json
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock
import cv2
import numpy as np

from src.annotation_cleaner.core.image_padding import ImagePadder
from src.annotation_cleaner.core.clean_annotation import CleanAnnotation
from src.annotation_cleaner.core.restore_crop import RestoreCropper


"""
test_image_padder.py
--------------------
ImagePadder 클래스 통합 단위 테스트 (클린 코드 버전)
- 작은 이미지는 패딩 + 메타데이터 기록
- 큰 이미지는 복사만 수행 (메타데이터 미기록)
- 두 카테고리(repair, replace) 모두 검증
"""

# ============================================================
# 🧩 Helper Functions
# ============================================================
def create_test_image(path: Path, size=(512, 512), color=(128, 128, 128)):
    """테스트용 RGB 이미지 생성"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


def setup_test_environment(tmp_path: Path):
    """입출력 디렉토리 및 테스트용 이미지 구조 생성"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    (input_dir / "repair").mkdir(parents=True)
    (input_dir / "replace").mkdir(parents=True)

    # repair
    create_test_image(input_dir / "repair" / "small_repair.jpg", size=(512, 512))
    create_test_image(input_dir / "repair" / "large_repair.jpg", size=(1200, 1200))

    # replace
    create_test_image(input_dir / "replace" / "small_replace.jpg", size=(512, 512))
    create_test_image(input_dir / "replace" / "large_replace.jpg", size=(1400, 1400))

    return input_dir, output_dir


def load_metadata(meta_path: Path):
    """메타데이터(JSON) 파일 로드"""
    if not meta_path.exists():
        raise AssertionError(f"❌ 메타데이터 파일이 존재하지 않음: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def create_fake_padding_metadata(meta_path: Path, image_files):
    """테스트용 padding_info.json 파일 생성"""
    metadata = {}
    for file in image_files:
        metadata[file.name] = {
            "orig_size": [512, 512],
            "pad_info": {"top": 256, "left": 256, "bottom": 256, "right": 256},
        }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


# ============================================================
# ✅ Test Functions
# ============================================================
def test_image_padder_run_creates_expected_outputs(tmp_path):
    """ImagePadder가 작은 이미지는 패딩하고, 큰 이미지는 복사만 수행해야 함"""
    # --- 1️⃣ 테스트 환경 구성 ---
    input_dir, output_dir = setup_test_environment(tmp_path)

    # --- 2️⃣ 실행 ---
    padder = ImagePadder(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        categories=["repair", "replace"],
        target_size=1024,
    )
    padder.run()

    # --- 3️⃣ repair 카테고리 검증 ---
    verify_category_output(output_dir, "repair")

    # --- 4️⃣ replace 카테고리 검증 ---
    verify_category_output(output_dir, "replace")


def verify_category_output(output_dir: Path, category: str):
    """카테고리별 결과 검증 (클린 코드 스타일)"""
    out_dir = output_dir / category
    meta_path = out_dir / "padding_info.json"

    # 📁 기본 폴더 및 파일 존재 확인
    assert out_dir.exists(), f"{category} 출력 폴더가 없음"
    assert (out_dir / f"small_{category}.jpg").exists(), f"small_{category}.jpg 결과 이미지 없음"
    assert (out_dir / f"large_{category}.jpg").exists(), f"large_{category}.jpg 복사본 없음"
    assert meta_path.exists(), f"{category} 메타데이터 누락"

    # 🧾 메타데이터 로드 및 검증
    meta = load_metadata(meta_path)

    small_file = f"small_{category}.jpg"
    large_file = f"large_{category}.jpg"

    # 작은 이미지는 메타데이터에 포함되어야 함
    assert small_file in meta, f"{small_file} 메타데이터 누락"
    assert "pad_info" in meta[small_file], f"{small_file} pad_info 누락"

    # 큰 이미지는 메타데이터에 포함되면 안 됨
    assert large_file not in meta, f"{large_file} 메타데이터에 잘못 포함됨"


"""
test_clean_annotation.py
------------------------
CleanAnnotation 클래스 단위 테스트 (Mock 기반)
- Gemini API 호출 없이 이미지 생성 로직 검증
- _generate_clean_image() 및 run() 메서드 동작 확인
"""
# ============================================================
# 🧩 Mock Helper
# ============================================================
def mock_gemini_client(tmp_image: Path):
    """
    Gemini API 응답을 Mock 객체로 구성
    - response.candidates[0].content.parts[0].inline_data.data → 이미지 바이트
    """
    mock_client = MagicMock()

    # 가짜 이미지 데이터 준비
    with open(tmp_image, "rb") as f:
        fake_bytes = f.read()

    mock_inline_part = MagicMock()
    mock_inline_part.inline_data.data = fake_bytes
    mock_content = MagicMock()
    mock_content.parts = [mock_inline_part]
    mock_candidate = MagicMock()
    mock_candidate.content = mock_content
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]

    # client.models.generate_content() → mock_response 반환
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


# ============================================================
# ✅ Unit Test: _generate_clean_image
# ============================================================
def test_generate_clean_image_creates_output(tmp_path):
    """_generate_clean_image가 Gemini 응답을 통해 이미지 파일을 저장하는지 테스트"""
    input_img = create_test_image(tmp_path / "input.jpg")
    output_img = tmp_path / "output.jpg"

    fake_client = mock_gemini_client(input_img)

    cleaner = CleanAnnotation(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
        model="fake-model",
        prompt="Remove markings.",
        client=fake_client,
        test_mode=True,
    )

    success = cleaner._generate_clean_image(input_img, output_img)

    assert success is True, "_generate_clean_image가 False를 반환함"
    assert output_img.exists(), "출력 이미지가 저장되지 않음"


# ============================================================
# ✅ Integration Test: run()
# ============================================================
def test_clean_annotation_run_creates_outputs(tmp_path):
    """run()이 카테고리별 입력 폴더를 순회하며 결과 이미지를 생성해야 함"""
    # --- 1️⃣ 환경 구성 (공용 setup 함수 사용) ---
    input_dir, output_dir = setup_test_environment(tmp_path)

    # --- 2️⃣ Mock Client 준비 ---
    fake_client = mock_gemini_client(input_dir / "repair" / "small_repair.jpg")

    # --- 3️⃣ 실행 ---
    cleaner = CleanAnnotation(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        model="fake-model",
        prompt="Remove markings.",
        categories=["repair", "replace"],
        client=fake_client,
        test_mode=True,
        test_limit=10,
    )
    cleaner.run()

    # --- 4️⃣ 검증 ---
    for category in ["repair", "replace"]:
        for filename in ["small", "large"]:
            img_name = f"{filename}_{category}.jpg"
            output_path = output_dir / category / img_name
            assert output_path.exists(), f"{img_name} 결과 이미지가 생성되지 않음"

    # --- 5️⃣ Mock 호출 검증 ---
    fake_client.models.generate_content.assert_called()

    #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_restore_crop.py
--------------------
RestoreCropper 클래스 단위 테스트
- ImagePadder에서 생성된 메타데이터 기반 복원 테스트
"""


# ============================================================
# ✅ Unit Test: _restore_single_image
# ============================================================
def test_restore_single_image_restores_correct_roi(tmp_path):
    """_restore_single_image가 지정된 ROI를 잘라내어 복원하는지 테스트"""
    # --- 입력 1024x1024 이미지 생성 (흰 배경) ---
    padded_img_path = tmp_path / "padded.jpg"
    img = np.full((1024, 1024, 3), 255, np.uint8)
    cv2.imwrite(str(padded_img_path), img)

    # --- 메타데이터 생성 (중앙 512x512) ---
    meta = {
        "orig_size": [512, 512],
        "pad_info": {"top": 256, "left": 256, "bottom": 256, "right": 256},
    }

    # --- 복원 대상 ---
    output_path = tmp_path / "restored.jpg"

    # --- 실행 ---
    restorer = RestoreCropper(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
        meta_dir=str(tmp_path),
        metadata_name="padding_info.json",
    )
    success = restorer._restore_single_image(padded_img_path, meta, output_path)

    # --- 검증 ---
    assert success is True, "_restore_single_image가 실패함"
    assert output_path.exists(), "복원된 이미지가 저장되지 않음"
    restored_img = cv2.imread(str(output_path))
    assert restored_img.shape[:2] == (512, 512), "복원된 이미지 크기가 orig_size와 다름"


# ============================================================
# ✅ Integration Test: run()
# ============================================================
def test_restore_crop_run_restores_padded_images(tmp_path):
    """run()이 각 카테고리 폴더의 padded 이미지를 원본 크기로 복원해야 함"""
    # --- 1️⃣ 테스트 환경 구성 ---
    input_dir, output_dir = setup_test_environment(tmp_path)
    meta_dir = tmp_path / "meta"

    # generated_image_padded 폴더 구조 시뮬레이션
    padded_dir = tmp_path / "generated_image_padded"
    for category in ["repair", "replace"]:
        (padded_dir / category).mkdir(parents=True)
        create_test_image(padded_dir / category / f"small_{category}.jpg", size=(1024, 1024))
        create_test_image(padded_dir / category / f"large_{category}.jpg", size=(1024, 1024))

        meta_path = meta_dir / category / "padding_info.json"
        create_fake_padding_metadata(meta_path, [
            Path(f"small_{category}.jpg"),
        ])

    # --- 2️⃣ 실행 ---
    restorer = RestoreCropper(
        input_dir=str(padded_dir),
        output_dir=str(output_dir),
        meta_dir=str(meta_dir),
        categories=["repair", "replace"],
        metadata_name="padding_info.json",
    )
    restorer.run()

    # --- 3️⃣ 검증 ---
    for category in ["repair", "replace"]:
        out_dir = output_dir / category
        restored_small = out_dir / f"small_{category}.jpg"
        restored_large = out_dir / f"large_{category}.jpg"

        # 파일 존재 확인
        assert restored_small.exists(), f"{restored_small.name} 복원 실패"
        assert restored_large.exists(), f"{restored_large.name} 복사 실패"

        # 크기 검증
        img_small = cv2.imread(str(restored_small))
        img_large = cv2.imread(str(restored_large))
        assert img_small.shape[:2] == (512, 512), f"{category}: small 이미지 복원 크기 오류"
        assert img_large.shape[:2] == (1024, 1024), f"{category}: large 이미지는 복사이므로 원본 크기 유지"

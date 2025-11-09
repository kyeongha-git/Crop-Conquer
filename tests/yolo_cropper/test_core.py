#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
from pathlib import Path
import pytest
import copy
import json
import cv2
import numpy as np
import shutil

from src.yolo_cropper.core.converter import YOLOConverter, infer_class_from_folder
from src.yolo_cropper.core.cropper import YOLOCropper


# ==============================================================
# 🔹 Fixtures
# ==============================================================

@pytest.fixture
def tmp_dataset(tmp_path):
    """Creates a temporary dataset directory structure for testing."""
    data_root = tmp_path / "data" / "original"
    (data_root / "repair").mkdir(parents=True)
    (data_root / "replace").mkdir(parents=True)

    # Add valid dummy images (100x100 PNG)
    import numpy as np, cv2
    for cls in ["repair", "replace"]:
        for i in range(2):
            img = np.full((100, 100, 3), 127 if cls == "repair" else 200, dtype=np.uint8)
            cv2.imwrite(str(data_root / cls / f"img{i}.png"), img)

    return data_root


@pytest.fixture
def tmp_detect_root(tmp_path):
    """Creates a temporary detect directory structure mimicking YOLO outputs."""
    detect_root = tmp_path / "runs" / "detect"
    (detect_root / "yolov5_repair" / "labels").mkdir(parents=True)
    (detect_root / "yolov5_replace" / "labels").mkdir(parents=True)

    # repair labels
    (detect_root / "yolov5_repair" / "labels" / "img0.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n")
    (detect_root / "yolov5_repair" / "labels" / "img1.txt").write_text("0 0.4 0.4 0.3 0.3 0.8\n")

    # replace labels
    (detect_root / "yolov5_replace" / "labels" / "img0.txt").write_text("0 0.6 0.6 0.2 0.2 0.7\n")

    # Add mock images (as if YOLO detect outputs exist)
    for cls in ["yolov5_repair", "yolov5_replace"]:
        for i in range(2):
            (detect_root / cls / f"img{i}.jpg").write_text("mock_image_data")

    return detect_root


@pytest.fixture
def config_template(tmp_dataset, tmp_detect_root, tmp_path):
    """Creates a minimal config dictionary mimicking config.yaml"""
    return {
        "yolo_cropper": {
            "main": {
                "model_name": "yolov5",
                "input_dir": str(tmp_dataset)
            },
            "dataset": {
                "detect_dir": str(tmp_detect_root),
                "results_dir": str(tmp_path / "outputs" / "json_results"),
            },
        }
    }

@pytest.fixture
def tmp_results_json(tmp_path, tmp_dataset):
    """result.json + predict.txt를 생성하여 Cropper 테스트용으로 반환"""
    results_dir = tmp_path / "outputs" / "json_results" / "yolov5"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ result.json — 한 장에는 bbox가 있고, 한 장은 없음
    result_data = [
        {
            "filename": str(tmp_dataset / "repair" / "img0.png"),
            "objects": [
                {
                    "relative_coordinates": {
                        "center_x": 0.5, "center_y": 0.5,
                        "width": 0.4, "height": 0.4,
                    },
                    "confidence": 0.9,
                    "class_id": 0,
                    "name": "repair",
                }
            ],
        },
        {
            "filename": str(tmp_dataset / "replace" / "img1.png"),
            "objects": [],  # no detection
        },
    ]
    json_path = results_dir / "result.json"
    json_path.write_text(json.dumps(result_data, indent=4, ensure_ascii=False), encoding="utf-8")

    # 2️⃣ predict.txt — 전체 이미지 포함
    predict_txt = results_dir.parent / "predict.txt"
    all_imgs = sorted(str(p) for p in tmp_dataset.rglob("*.png"))
    predict_txt.write_text("\n".join(all_imgs), encoding="utf-8")

    return {"json": json_path, "predict": predict_txt, "root": results_dir.parent}

# ==============================================================
# 🔹 Unit Tests
# ==============================================================


"""
test_converter.py
-----------------
Unit tests for YOLOConverter and helper functions in converter.py

Test Focus:
- infer_class_from_folder()
- _parse_detect_folder()
- run() end-to-end JSON aggregation
"""


def test_infer_class_from_folder():
    """Test folder name class inference."""
    repair_path = Path("/some/path/repair/images")
    replace_path = Path("/another/replace/set")
    unknown_path = Path("/no/class/here")

    assert infer_class_from_folder(repair_path)["name"] == "repair"
    assert infer_class_from_folder(replace_path)["name"] == "replace"
    assert infer_class_from_folder(unknown_path)["name"] == "unknown"


def test_parse_detect_folder_parses_valid_labels(tmp_dataset, tmp_detect_root, config_template):
    """Ensure _parse_detect_folder() correctly parses YOLO label files."""
    converter = YOLOConverter(config_template)
    detect_dir = tmp_detect_root / "yolov5_repair"

    results, next_frame = converter._parse_detect_folder(detect_dir)
    assert isinstance(results, list)
    assert len(results) == 2
    assert next_frame == 3  # starts at 1, increments twice
    assert all("objects" in r for r in results)
    assert "filename" in results[0]
    assert "relative_coordinates" in results[0]["objects"][0]


def test_run_creates_result_json(tmp_dataset, tmp_detect_root, config_template):
    """Test end-to-end run() creates aggregated result.json correctly."""
    converter = YOLOConverter(config_template)
    converter.run()

    # result.json 경로 확인
    output_json = converter.output_json
    assert output_json.exists(), "result.json should be created"
    data = json.loads(output_json.read_text(encoding="utf-8"))

    # 데이터 유효성 검사
    assert isinstance(data, list)
    assert len(data) > 0
    first_item = data[0]
    assert "filename" in first_item
    assert "objects" in first_item
    assert isinstance(first_item["objects"], list)
    assert "relative_coordinates" in first_item["objects"][0]

def test_run_raises_if_no_detect_folder(tmp_path, config_template):
    """When no valid detect folders exist, run() should raise FileNotFoundError."""
    # 🔹 독립된 config 복사본 사용 (기존 fixture 변조 방지)
    cfg = copy.deepcopy(config_template)

    # 🔹 빈 detect 디렉토리 생성
    empty_detect = tmp_path / "runs" / "detect_empty"
    empty_detect.mkdir(parents=True, exist_ok=True)
    cfg["yolo_cropper"]["dataset"]["detect_dir"] = str(empty_detect)

    converter = YOLOConverter(cfg)

    # 🔹 실제로 detect_root 내부가 비었는지 double-check
    assert list(converter.detect_root.iterdir()) == [], f"Detect root not empty: {list(converter.detect_root.iterdir())}"

    # 🔹 FileNotFoundError 발생 확인
    with pytest.raises(FileNotFoundError) as excinfo:
        converter.run()

    assert "No detect folders" in str(excinfo.value)


"""
test_cropper.py
---------------
Unit tests for YOLOCropper (Config-driven).
Focus: JSON-based cropping, missing images handling, and original copy behavior.
"""


def test_cropper_creates_crops_and_originals(tmp_path, config_template, tmp_results_json):
    """✅ YOLOCropper가 bbox 기반 크롭 및 No-detection 복사를 정상 수행"""
    # config 업데이트 (results_dir 적용)
    config_template["yolo_cropper"]["dataset"]["results_dir"] = str(tmp_results_json["root"])

    cropper = YOLOCropper(config_template)
    cropper.crop_from_json()

    out_dir = cropper.output_dir
    assert out_dir.exists(), "Output directory must exist after cropping."

    repair_dir = out_dir / "repair"
    replace_dir = out_dir / "replace"
    repair_files = list(repair_dir.glob("*.jpg"))
    replace_files = list(replace_dir.glob("*.png"))

    # repair/img0 → crop된 결과 존재
    assert any("_1" in f.name for f in repair_files), "Cropped file must contain '_1' suffix"
    # replace/img1 → 원본 복사됨
    assert any("img1.png" in f.name for f in replace_files), "No-detection file must be copied"

    # crop된 이미지가 실제 크롭된 영역인지 간단히 확인
    img = cv2.imread(str(repair_files[0]))
    assert img is not None and img.size > 0, "Cropped image should be valid."


def test_cropper_raises_if_json_missing(tmp_path, config_template, tmp_results_json):
    """❌ result.json이 없으면 FileNotFoundError"""
    config_template["yolo_cropper"]["dataset"]["results_dir"] = str(tmp_results_json["root"])
    cropper = YOLOCropper(config_template)
    cropper.json_path.unlink()  # 삭제

    with pytest.raises(FileNotFoundError):
        cropper.crop_from_json()


def test_cropper_raises_if_predict_missing(tmp_path, config_template, tmp_results_json):
    """❌ predict.txt가 없으면 FileNotFoundError"""
    config_template["yolo_cropper"]["dataset"]["results_dir"] = str(tmp_results_json["root"])
    cropper = YOLOCropper(config_template)
    cropper.predict_list.unlink()  # 삭제

    with pytest.raises(FileNotFoundError):
        cropper.crop_from_json()


def test_cropper_copies_missing_images(tmp_path, config_template, tmp_results_json):
    """✅ JSON에 없는 이미지라도 predict.txt에 있으면 원본 복사"""
    config_template["yolo_cropper"]["dataset"]["results_dir"] = str(tmp_results_json["root"])
    cropper = YOLOCropper(config_template)

    # result.json에서 하나의 항목만 남겨 일부 이미지 누락시킴
    partial_data = json.loads(cropper.json_path.read_text(encoding="utf-8"))[:1]
    cropper.json_path.write_text(json.dumps(partial_data, indent=4, ensure_ascii=False), encoding="utf-8")

    cropper.crop_from_json()

    out_dir = cropper.output_dir
    copied = list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.jpg"))
    assert len(copied) > 0, "Missing images should be copied to output folder."
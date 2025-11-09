import os
import json
import tempfile
from unittest.mock import patch, MagicMock
import pytest
import torch
import matplotlib

# ✅ Headless 환경 설정 (서버/CI에서 plt 사용 가능하게)
matplotlib.use("Agg")

from src.classifier.evaluate import Evaluator


# ======================================================
# ✅ 0️⃣ 테스트용 Dummy Config 생성
# ======================================================
@pytest.fixture
def dummy_cfg(tmp_path):
    """Evaluator에서 요구하는 최소 config 구조"""
    cfg = {
        "train": {
            "save_dir": str(tmp_path / "saved_models"),
            "metric_dir": str(tmp_path / "metrics"),
        },
        "wandb": {
            "enabled": False  # 테스트에서는 wandb 비활성화
        },
    }
    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["train"]["metric_dir"], exist_ok=True)
    return cfg


@pytest.fixture
def evaluator(tmp_path, dummy_cfg):
    """Evaluator 인스턴스 (모든 저장 경로를 tmp_path 기반으로 강제)"""
    input_dir = str(tmp_path / "data" / "original")
    os.makedirs(input_dir, exist_ok=True)
    ev = Evaluator(
        input_dir=input_dir,
        model="vgg16",
        cfg=dummy_cfg,  # ✅ 변경된 부분: config_path 대신 dict 주입
    )
    return ev


# ======================================================
# ✅ 1️⃣ _get_transform() 테스트
# ======================================================
def test_get_transform_vgg(evaluator):
    transform = evaluator._get_transform()
    assert transform is not None
    ops = [t.__class__.__name__ for t in transform.transforms]
    has_aug = any("Affine" in o or "Random" in o for o in ops)
    assert not has_aug
    print(f"✅ _get_transform() 테스트 통과 — 포함 연산: {ops}")


# ======================================================
# ✅ 2️⃣ _load_model() 테스트
# ======================================================
def test_load_model_file_exists(evaluator):
    """_load_model()이 정상적으로 모델을 로드하는지"""
    dummy_model_path = os.path.join(evaluator.save_root, "vgg16.pt")
    os.makedirs(os.path.dirname(dummy_model_path), exist_ok=True)
    torch.save({}, dummy_model_path)

    with patch("torch.nn.Module.load_state_dict", return_value=None):
        model = evaluator._load_model()

    assert model is not None
    print("✅ _load_model() 테스트 통과")


# ======================================================
# ✅ 3️⃣ _load_data() 테스트
# ======================================================
def test_load_data_returns_loader(evaluator):
    """_load_data()가 DataLoader를 반환하는지"""
    transform = evaluator._get_transform()

    with patch(
        "src.classifier.evaluate.ClassificationDataset",
        return_value=[(torch.rand(3, 360, 360), torch.tensor(1))],
    ):
        loader = evaluator._load_data(transform)

    assert loader is not None and hasattr(loader, "__iter__")
    print(f"✅ _load_data() 테스트 통과 — 배치 수: {len(loader)}")


# ======================================================
# ✅ 4️⃣ _save_results() 테스트
# ======================================================
def test_save_results_creates_files(evaluator, tmp_path):
    """_save_results()가 metrics.json 및 confusion matrix 파일을 tmp_path 내에 저장하는지"""
    temp_metrics_dir = tmp_path / "metrics"
    evaluator.metric_root = temp_metrics_dir
    os.makedirs(temp_metrics_dir, exist_ok=True)

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    acc, f1 = 0.85, 0.8

    evaluator._save_results(y_true, y_pred, acc, f1)

    # ✅ 새로운 구조: metrics/classifier/vgg16/metrics.json
    metrics_path = os.path.join(temp_metrics_dir, "classifier", "vgg16", "metrics.json")
    cm_path = os.path.join(temp_metrics_dir, "classifier", "vgg16", "cm.png")

    assert os.path.exists(metrics_path), f"❌ metrics.json 파일 없음: {metrics_path}"
    assert os.path.exists(cm_path), f"❌ confusion matrix 이미지 없음: {cm_path}"

    with open(metrics_path, "r") as f:
        data = json.load(f)
    assert "accuracy" in data and "f1_score" in data
    print("✅ _save_results() 테스트 통과 (tmp_path 내부에 저장됨)")


# ======================================================
# ✅ 5️⃣ run() 통합(Mock) 테스트
# ======================================================
def test_run_integration_mock(evaluator, tmp_path):
    """run() 전체 파이프라인이 정상적으로 연결되는지(Mock 기반)"""
    evaluator.metric_root = tmp_path / "metrics"
    os.makedirs(evaluator.metric_root, exist_ok=True)

    with patch.object(Evaluator, "_get_transform", return_value=MagicMock()):
        with patch.object(
            Evaluator, "_load_data", return_value=[(torch.rand(1, 3, 360, 360), torch.tensor([1]))]
        ):
            with patch.object(Evaluator, "_load_model", return_value=MagicMock()):
                with patch("torch.sigmoid", return_value=torch.tensor([[0.9]])):
                    acc, f1 = evaluator.run()

    assert isinstance(acc, float) and isinstance(f1, float)

    # ✅ 수정된 경로 반영
    metrics_json = os.path.join(evaluator.metric_root, "classifier", "vgg16", "metrics.json")
    assert os.path.exists(metrics_json), f"❌ metrics.json 파일 없음: {metrics_json}"
    print(f"✅ run() 통합(Mock) 테스트 통과 — ACC={acc:.4f}, F1={f1:.4f}")

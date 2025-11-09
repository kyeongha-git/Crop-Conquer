"""
test_models.py

Unit & integration tests for all classification models.
- VGG16 / ResNet152 / MobileNetV2 / MobileNetV3
- Verifies forward pass, parameter freezing, dropout usage, and end-to-end data flow.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.classifier.data.cnn_data_loader import ClassificationDataset
from src.classifier.data.data_preprocessing import DataPreprocessor
from src.classifier.models.factory import get_model


# ==============================================================
# 🧩 Helper Functions
# ==============================================================

def run_forward_pass(model_name: str, num_classes: int = 1, input_size=(1, 3, 360, 360)):
    """모델을 로드하고 더미 입력으로 forward pass 수행"""
    model = get_model(model_name, num_classes=num_classes)
    model.eval()
    x = torch.randn(*input_size)
    with torch.no_grad():
        y = model(x)
    return model, y


def compute_loss(output: torch.Tensor):
    """BCEWithLogitsLoss 계산 (NaN 방지 확인 포함)"""
    criterion = nn.BCEWithLogitsLoss()
    target = torch.ones_like(output)
    loss = criterion(output, target)
    assert not torch.isnan(loss), "❌ 손실 계산 중 NaN 발생"
    return loss.item()


def count_parameters(model: nn.Module):
    """전체 및 학습 가능한 파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ==============================================================
# ① Unit Test: Model Structure & Forward
# ==============================================================

@pytest.mark.parametrize(
    "model_name, expect_dropout",
    [
        ("vgg16", True),
        ("resnet152", False),   # ResNet에는 Dropout 없음
        ("mobilenet_v2", True),
        ("mobilenet_v3", True),
    ],
)
def test_model_forward_and_structure(model_name, expect_dropout):
    """모델 forward 및 Dropout 존재 여부 테스트"""
    print(f"\n🧠 [TEST] {model_name.upper()} forward pass & structure 검증")

    model, output = run_forward_pass(model_name)

    # ✅ 출력 차원 검증
    assert output.ndim == 2 and output.shape[1] == 1, f"{model_name} 출력 shape 오류: {output.shape}"

    # ✅ Dropout 존재 여부 확인
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    assert has_dropout == expect_dropout, (
        f"{model_name}: Dropout 존재 여부 불일치 "
        f"(expected={expect_dropout}, found={has_dropout})"
    )

    # ✅ BCE 손실 계산
    loss_val = compute_loss(output)
    total_params, trainable_params = count_parameters(model)
    print(f" - BCE Loss: {loss_val:.4f}")
    print(f" - Params: total={total_params:,}, trainable={trainable_params:,}")
    print(f"✅ {model_name.upper()} 구조 및 Forward Test 통과")


# ==============================================================
# ② Unit Test: Backbone Freeze 동작 검증
# ==============================================================

@pytest.mark.parametrize("model_name", ["resnet152", "mobilenet_v2", "mobilenet_v3"])
def test_freeze_backbone_option(model_name):
    """freeze_backbone 옵션이 실제로 파라미터에 반영되는지 테스트"""
    model_frozen = get_model(model_name, freeze_backbone=True)
    model_trainable = get_model(model_name, freeze_backbone=False)

    frozen_params = [p.requires_grad for p in model_frozen.parameters()]
    trainable_params = [p.requires_grad for p in model_trainable.parameters()]

    assert any(trainable_params), f"{model_name}: freeze_backbone=False인데 모두 freeze됨"
    assert not all(frozen_params), f"{model_name}: freeze_backbone=True인데 일부 trainable"


# ==============================================================
# ③ Integration Test: Data → Transform → Model Pipeline
# ==============================================================

@pytest.mark.parametrize(
    "model_name",
    ["vgg16", "resnet152", "mobilenet_v2", "mobilenet_v3"],
)
def test_real_end_to_end_pipeline(tmp_path, model_name):
    """
    실제 파이프라인 테스트:
    1️⃣ Dataset → 2️⃣ Transform → 3️⃣ Model Forward
    """

    print(f"\n🔗 [REAL TEST] {model_name.upper()} 실제 데이터 파이프라인 테스트")

    # ---------------------------------
    # 1️⃣ 테스트용 가짜 데이터셋 생성
    # ---------------------------------
    data_dir = tmp_path / "data" / "original_crop" / "yolov2" / "train" / "repair"
    data_dir.mkdir(parents=True, exist_ok=True)
    dummy_path = data_dir / "dummy.jpg"

    # (랜덤 픽셀 이미지 생성)
    img = Image.fromarray((torch.rand(3, 360, 360).permute(1, 2, 0).numpy() * 255).astype("uint8"))
    img.save(dummy_path)

    # ---------------------------------
    # 2️⃣ Dataset + Transform 로드
    # ---------------------------------
    dp = DataPreprocessor(img_size=(360, 360))
    transform = dp.get_transform(model_name=model_name, mode="train")

    dataset = ClassificationDataset(
        input_dir=str(tmp_path / "data" / "original_crop" / "yolov2"),
        split="train",
        transform=transform,  # ✅ 실제 transform 적용
        verbose=True,
    )

    # ---------------------------------
    # 3️⃣ 샘플 로드 및 모델 입력 변환
    # ---------------------------------
    img_tensor, label = dataset[0]
    assert isinstance(img_tensor, torch.Tensor), "❌ Transform 후 Tensor가 아님"
    assert img_tensor.shape == (3, 360, 360), f"❌ 이미지 shape 오류: {img_tensor.shape}"

    x = img_tensor.unsqueeze(0)

    # ---------------------------------
    # 4️⃣ 모델 Forward
    # ---------------------------------
    model = get_model(model_name, num_classes=1)
    model.eval()

    with torch.no_grad():
        y = model(x)

    # ---------------------------------
    # 5️⃣ 결과 검증
    # ---------------------------------
    assert y.ndim == 2 and y.shape[1] == 1, f"{model_name} 출력 shape 오류: {y.shape}"
    loss_val = compute_loss(y)

    print(f"✅ {model_name.upper()} 실제 파이프라인 통과 (loss={loss_val:.4f})")
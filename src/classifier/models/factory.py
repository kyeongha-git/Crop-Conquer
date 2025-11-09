import torch.nn as nn
from src.classifier.models.mobilenet import MobileNetClassifier
from src.classifier.models.resnet152 import ResNet152Classifier
from src.classifier.models.vgg16 import VGGClassifier


def get_model(
    model_name: str,
    num_classes: int = 2,
    dropout_p: float = 0.5,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Factory function to instantiate a classification model by name.

    Args:
        model_name (str): one of ['vgg16', 'resnet152', 'mobilenet_v2', 'mobilenet_v3']
        num_classes (int): number of output classes (default=2)
        dropout_p (float): dropout probability for models that support it (default=0.5)
        freeze_backbone (bool): whether to freeze pretrained weights (default=True)

    Returns:
        nn.Module: the initialized model

    Raises:
        ValueError: if an unsupported model_name is provided
    """

    model_key = model_name.strip().lower()

    MODEL_MAP = {
        "vgg": ("VGG16", VGGClassifier, {"dropout_p": dropout_p}),
        "vgg16": ("VGG16", VGGClassifier, {"dropout_p": dropout_p}),

        # ResNet152 does not use dropout_p — intentionally excluded
        "resnet": ("ResNet152", ResNet152Classifier, {"freeze_backbone": freeze_backbone}),
        "resnet152": ("ResNet152", ResNet152Classifier, {"freeze_backbone": freeze_backbone}),
        
        "mobilenet_v2": (
            "MobileNetV2",
            MobileNetClassifier,
            {"dropout_p": dropout_p, "model_type": "mobilenet_v2", "freeze_backbone": freeze_backbone},
        ),
        "mobilenet_v3": (
            "MobileNetV3",
            MobileNetClassifier,
            {"dropout_p": dropout_p, "model_type": "mobilenet_v3", "freeze_backbone": freeze_backbone},
        ),
        "mobilenet": (
            "MobileNetV2",
            MobileNetClassifier,
            {"dropout_p": dropout_p, "model_type": "mobilenet_v2", "freeze_backbone": freeze_backbone},
        ),
    }

    if model_key not in MODEL_MAP:
        raise ValueError(
            f"❌ Unknown model name '{model_name}'. "
            f"Supported options: {list(MODEL_MAP.keys())}"
        )

    model_label, model_class, extra_kwargs = MODEL_MAP[model_key]
    print(f"🔹 Using {model_label} backbone")

    # ✅ Dropout은 ResNet에 적용되지 않음
    if "resnet" in model_key and "dropout_p" in extra_kwargs:
        extra_kwargs.pop("dropout_p", None)

    return model_class(num_classes=num_classes, **extra_kwargs)

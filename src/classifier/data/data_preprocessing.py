#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_preprocessing.py
---------------------
Unified preprocessing manager for classification models.

- Supports: VGG, ResNet, MobileNet family
- Handles train / eval pipelines
- Normalization settings model-aware
"""

from torchvision import transforms
from typing import Tuple, Dict


class DataPreprocessor:
    """Manage model-specific preprocessing pipelines."""

    _NORMALIZATION_MAP: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
        "vgg": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "vgg16": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "resnet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "resnet152": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "mobilenet_v1": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        "mobilenet_v2": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        "mobilenet_v3": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    }

    def __init__(self, img_size: Tuple[int, int] = (360, 360), augment_translate: float = 0.2, augment_scale: Tuple[float, float] = (0.8, 1.2)):
        self.img_size = img_size
        self.augment_translate = augment_translate
        self.augment_scale = augment_scale

    # -----------------------------
    # Core Augmentation
    # -----------------------------
    def _augmentation(self) -> transforms.RandomAffine:
        return transforms.RandomAffine(
            degrees=0,
            translate=(self.augment_translate, self.augment_translate),
            scale=self.augment_scale,
            fill=0,
        )

    def _compose(self, mean, std, augment: bool = False) -> transforms.Compose:
        ops = [transforms.Resize(self.img_size)]
        if augment:
            ops.append(self._augmentation())
        ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        return transforms.Compose(ops)

    # -----------------------------
    # Public API
    # -----------------------------
    def get_transform(self, model_name: str, mode: str = "train") -> transforms.Compose:
        """
        Return preprocessing pipeline for given model/mode.

        Args:
            model_name (str): One of ['vgg', 'resnet', 'mobilenet_v2', ...]
            mode (str): 'train' or 'eval'
        """
        if model_name not in self._NORMALIZATION_MAP:
            raise ValueError(f"❌ Unsupported model: {model_name}")

        mean, std = self._NORMALIZATION_MAP[model_name]
        augment = mode.lower() == "train"
        return self._compose(mean, std, augment=augment)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cnn_data_loader.py
-------------------
📦 Unified data loading module for image classification tasks.

Features:
- PyTorch-style Dataset class (lazy loading)
- Automatic label mapping (str <-> int)
- DataLoader helper for easy batch creation
- Compatible with train / valid / test splits
- Extensible for any custom dataset directory structure

Example:
    from src.classifier.data.cnn_data_loader import create_dataloader

    train_loader = create_dataloader(
        input_dir="data/original",
        split="train",
        batch_size=32,
        shuffle=True
    )
"""

import os
from typing import List, Tuple, Dict, Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# Utility Functions
# ============================================================
def list_image_paths(root_dir: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[Tuple[str, str]]:
    """Return (image_path, class_name) pairs from directory."""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"❌ Data path not found: {root_dir}")

    image_label_pairs = []
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            if filename.lower().endswith(exts):
                image_label_pairs.append((os.path.join(class_path, filename), class_name))
    if not image_label_pairs:
        raise ValueError(f"⚠️ No images found under {root_dir}")
    return image_label_pairs


def build_label_mappings(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return label_to_idx and idx_to_label mappings."""
    unique_labels = sorted(set(labels))
    label_to_idx = {name: idx for idx, name in enumerate(unique_labels)}
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


# ============================================================
# PyTorch Dataset Class
# ============================================================
class ClassificationDataset(Dataset):
    """
    Unified classification dataset (no dataset_type/version).

    Args:
        input_dir (str): e.g. 'data/original' or 'data/original_crop/yolov2'
        split (str): 'train', 'valid', or 'test'
        transform (callable, optional): Torch transform
        verbose (bool): Print dataset summary
    """

    def __init__(
        self,
        input_dir: str,
        split: str = "train",
        transform=None,
        verbose: bool = True,
    ):
        self.root_dir = os.path.join(input_dir, split)
        self.transform = transform

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"❌ Split path not found: {self.root_dir}")

        image_label_pairs = list_image_paths(self.root_dir)
        if not image_label_pairs:
            raise RuntimeError(f"❌ No images found in {self.root_dir}")

        self.image_paths, self.labels = zip(*image_label_pairs)
        self.label_to_idx, self.idx_to_label = build_label_mappings(self.labels)

        if verbose:
            self._log_summary(input_dir, split)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label_name = self.labels[idx]
        label = self.label_to_idx[label_name]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

    def _log_summary(self, input_dir: str, split: str) -> None:
        """Print dataset info."""
        print(f"✅ Loaded dataset from {input_dir}/{split}")
        print(f" - Samples: {len(self.image_paths)}")
        print(f" - Classes: {self.label_to_idx}")


# ============================================================
# DataLoader Helper
# ============================================================
def create_dataloader(
    input_dir: str,
    split: str = "train",
    transform=None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    verbose: bool = False,
) -> DataLoader:
    """
    Wrapper that builds both Dataset and DataLoader.

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = ClassificationDataset(
        input_dir=input_dir,
        split=split,
        transform=transform,
        verbose=verbose,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# ============================================================
# Optional: Lightweight Function for Quick Inspection
# ============================================================
def load_dataset_as_list(
    input_dir: str,
    split: str = "train",
    transform=None,
) -> Tuple[List, List, Dict[str, int], Dict[int, str]]:
    """
    Load dataset fully into memory as a simple (image, label) list.
    Use only for small-scale analysis or visualization (not for training).
    """
    root_dir = os.path.join(input_dir, split)
    image_label_pairs = list_image_paths(root_dir)
    images, labels = [], []
    label_to_idx, idx_to_label = build_label_mappings([label for _, label in image_label_pairs])

    for img_path, label_name in image_label_pairs:
        img = Image.open(img_path).convert("RGB")
        if transform:
            img = transform(img)
        images.append(img)
        labels.append(label_to_idx[label_name])

    return images, labels, label_to_idx, idx_to_label

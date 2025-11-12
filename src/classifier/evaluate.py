#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py
-----------
Performs end-to-end evaluation of trained classification models.

This module loads a saved model, runs inference on the test dataset,
computes key performance metrics (Accuracy, F1-score), and exports results
including a confusion matrix visualization.
Supports both local evaluation and Weights & Biases logging.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader

import wandb

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.classifier.data.cnn_data_loader import ClassificationDataset
from src.classifier.data.data_preprocessing import DataPreprocessor
from src.classifier.models.factory import get_model
from utils.logging import get_logger, setup_logging


class Evaluator:
    """
    Evaluate trained classifier on test dataset.

    Example:
        evaluator = Evaluator(
            input_dir="data/original_crop/yolov8s",
            model="mobilenet_v2",
            cfg=config["Classifier"]
        )
        evaluator.run()
    """

    def __init__(self, input_dir: str, model: str, cfg: dict, wandb_run=None):
        """
        Args:
            input_dir (str): 테스트 데이터셋 경로
            model (str): 평가할 모델 이름
            cfg (dict): Classifier 섹션 딕셔너리
        """
        setup_logging("logs/classifier_eval")
        self.logger = get_logger("Evaluator")

        # --- Config ---
        self.cfg = cfg
        self.data_cfg = cfg.get("data", {})
        self.train_cfg = cfg.get("train", {})
        self.wandb_cfg = cfg.get("wandb", {})

        # --- 기본 속성 ---
        self.input_dir = Path(input_dir)
        self.model_name = model.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_run = wandb_run  # ✅ 추가

        # --- Classifier가 이미 확장한 경로 그대로 사용 ---
        self.save_root = Path(self.train_cfg["save_dir"])
        self.metric_root = Path(self.train_cfg["metric_dir"])
        self.check_root = Path(
            self.train_cfg.get("check_dir", "./checkpoints/classifier")
        )

        self.logger.info(f"🚀 Device: {self.device}")
        self.logger.info(f"📂 Dataset: {self.input_dir}")
        self.logger.info(f"🧠 Model: {self.model_name}")
        self.logger.info(f"💾 Using Save Dir: {self.save_root}")
        self.logger.info(f"💾 Using Metric Dir: {self.metric_root}")

    # ======================================================
    # 🔄 Transform
    # ======================================================
    def _get_transform(self):
        return DataPreprocessor().get_transform(self.model_name, mode="eval")

    # ======================================================
    # 💾 Load Model
    # ======================================================
    def _load_model(self):
        model = get_model(self.model_name, num_classes=1)
        model_path = self.save_root / f"{self.model_name}.pt"

        if not model_path.exists():
            self.logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device).eval()
        self.logger.info(f"✅ 모델 로드 완료: {model_path}")
        return model

    # ======================================================
    # 📦 Load Data
    # ======================================================
    def _load_data(self, transform):
        test_dataset = ClassificationDataset(
            input_dir=self.input_dir,
            split="test",
            transform=transform,
            verbose=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        self.logger.info(f"✅ 테스트셋 로드 완료 — {len(test_dataset)}개 샘플")
        return test_loader

    # ======================================================
    # 🧠 Run Evaluation
    # ======================================================
    def run(self):
        transform = self._get_transform()
        test_loader = self._load_data(transform)
        model = self._load_model()

        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = torch.sigmoid(model(imgs))
                preds = (outputs > 0.5).long().cpu().numpy().flatten()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        self.logger.info(f"📊 Test Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

        self._save_results(y_true, y_pred, acc, f1)

        # ✅ wandb logging (세션 전달받은 경우에만)
        if self.wandb_run is not None:
            self.wandb_run.log({"test_accuracy": acc, "test_f1": f1})

        return acc, f1

    # ======================================================
    # 💾 Save Results
    # ======================================================
    def _save_results(self, y_true, y_pred, acc, f1):
        """Save metrics and confusion matrix images."""
        save_dir = self.metric_root / "classifier" / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = save_dir / "metrics.json"
        cm_path = save_dir / "cm.png"

        metrics_data = {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "num_samples": len(y_true),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=4)

        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(
            cmap="Blues", values_format="d"
        )
        plt.savefig(cm_path, dpi=200, bbox_inches="tight")
        plt.close()

        self.logger.info(f"💾 Metrics saved at {metrics_path}")
        self.logger.info(f"💾 Confusion matrix saved at {cm_path}")

    # ======================================================
    # 📡 wandb Logging
    # ======================================================
    def _wandb_log(self, acc, f1):
        try:
            run_name = f"{self.model_name}_eval"
            wandb.init(
                project=self.wandb_cfg.get("project", "default"),
                entity=self.wandb_cfg.get("entity", None),
                name=run_name,
                notes="Evaluation run",
            )
            wandb.log({"test_accuracy": acc, "test_f1": f1})
            wandb.finish()
            self.logger.info("✅ wandb logging 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ wandb logging 실패: {e}")

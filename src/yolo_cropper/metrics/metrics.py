import re
import csv
import json
from pathlib import Path
import numpy as np


def parse_darknet_eval_log(log_path: str):
    """
    Parse Darknet evaluation log and extract metrics:
      - Precision
      - Recall
      - mAP@0.50
    """
    log_text = Path(log_path).read_text(errors="ignore")

    # ----------------------------
    # mean Average Precision (mAP@0.5)
    # ----------------------------
    m = re.search(r"mean average precision.*?=\s*([0-9]*\.?[0-9]+)\s*%?", log_text, re.I)
    mAP = float(m.group(1)) if m else None
    mAP_pct = mAP if (mAP is not None and mAP > 1) else (None if mAP is None else mAP * 100)

    # ----------------------------
    # Precision / Recall
    # ----------------------------
    pr = re.search(
        r"for\s+conf_thresh\s*=?\s*[0-9]*\.?[0-9]+\s*[, ]+precision\s*[:=]\s*([0-9]*\.?[0-9]+)\s*[, ]+recall\s*[:=]\s*([0-9]*\.?[0-9]+)",
        log_text, re.I,
    )

    if pr:
        precision = float(pr.group(1))
        recall = float(pr.group(2))
    else:
        # Fallback: TP/FP/FN 방식
        tpfpfn = re.search(r"TP\s*=\s*(\d+).*?FP\s*=\s*(\d+).*?FN\s*=\s*(\d+)", log_text, re.I | re.S)
        if tpfpfn:
            TP, FP, FN = map(int, tpfpfn.groups())
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        else:
            precision = recall = None

    # ----------------------------
    # Return structured result
    # ----------------------------
    return {
        "precision": precision,
        "recall": recall,
        "mAP@0.5": mAP_pct,
    }


def print_darknet_eval_summary(metrics: dict):
    """Pretty print the metrics"""
    print("=== Overall Evaluation (IoU=0.50, 101-point, conf_thresh=0.25) ===")
    if metrics["precision"] is not None:
        print(f"Precision: {metrics['precision']:.4f}")
    if metrics["recall"] is not None:
        print(f"Recall   : {metrics['recall']:.4f}")
    if metrics["mAP@0.5"] is not None:
        print(f"mAP@0.50 : {metrics['mAP@0.5']:.2f}%")
        

def parse_yolov5_eval_log(log_path: str):
    """
    Parse YOLOv5 val.py log to extract overall metrics (Precision, Recall, mAP@0.5).
    Works with local (non-COCO) datasets.
    """
    text = Path(log_path).read_text(errors="ignore")
    match = re.search(
        r"all\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        text
    )

    if not match:
        print("[!] Could not find 'all' metrics line in YOLOv5 log.")
        return {"precision": None, "recall": None, "mAP@0.5": None}

    precision = float(match.group(1))
    recall = float(match.group(2))
    map50 = float(match.group(3))

    return {
        "precision": precision,
        "recall": recall,
        "mAP@0.5": map50 * 100,  # convert to percentage
    }

# ==========================================================
# 🔹 YOLOv8 Parser (results.json)
# ==========================================================
def parse_yolov8_results(json_path: str):
    """
    Parse YOLOv8 results.json (Ultralytics val.py export)
    Typically includes keys: metrics.precision / recall / map50 / map
    """
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"results.json not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", data)  # some exports nest under "metrics"

    precision = float(metrics.get("precision", 0))
    recall = float(metrics.get("recall", 0))
    map50 = float(metrics.get("map50", metrics.get("mAP@0.5", 0)))

    return {"precision": precision, "recall": recall, "mAP@0.5": map50}


# ==========================================================
# 🔹 Unified Parser Selector
# ==========================================================
def get_metrics_parser(model_name: str):
    """
    Return the appropriate parsing function based on model name.
    Example:
        parser = get_metrics_parser("yolov5")
        metrics = parser("path/to/results.csv")
    """
    model_name = model_name.lower()
    if "darknet" in model_name or "yolov2" in model_name or "yolov4" in model_name:
        return parse_darknet_eval_log
    elif "yolov5" in model_name:
        return parse_yolov5_eval_log
    elif "yolov8" in model_name:
        return parse_yolov8_results
    else:
        raise ValueError(f"❌ Unsupported model_name for metrics parsing: {model_name}")
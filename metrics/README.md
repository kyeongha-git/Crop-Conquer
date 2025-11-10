# 📊 Metrics Directory

This directory contains evaluation results and performance metrics generated after model training and testing.

> ⚠️ **Note:**  
> All experiments were conducted using **private datasets**.  
> The metric files are provided for documentation and reproducibility of the results described in the paper.  
> Actual values may not be reproducible without access to the private dataset.

---

## 📁 Folder Structure

metrics/
├── annotation_cleaner/
│ ├── metrics_full_image.csv
│ └── metrics_yolo_crop.csv
├── classifier/
│ ├── original/
│ │ ├── mobilenet_v2_cm.png
│ │ └── mobilenet_v2_metrics.json
│ ├── original_crop/
│ │ └── yolov8s/
│ │   ├── mobilenet_v2_cm.png
│ │   └── mobilenet_v2_metrics.json
│ ├── generation/
│ │ ├── mobilenet_v2_cm.png
│ │ └── mobilenet_v2_metrics.json
│ └── generation_crop/
│ │ └── yolov8s/
│ │   ├── mobilenet_v2_cm.png
│ │   └── mobilenet_v2_metrics.json
└── yolo_cropper/
│ │ └── yolov2_metrics.csv/
│ │ └── yolov4, yolov5, yolov8_metrics.csv/


---

## 🧩 Description

### 1️⃣ Annotation Cleaner Metrics

This subdirectory contains **image similarity evaluation results**  
after removing human-drawn annotations using a generative AI model.

- **Purpose:**  
  To assess how closely the generated clean images match the original ones after annotation removal.

- **Evaluation Types:**  
  - `metrics_full_image.csv` → Measures **global consistency** using full-image comparison.  
  - `metrics_yolo_crop.csv` → Measures **local representation** by comparing only cropped damaged regions.

- **Metrics Columns:**  
split, file, SSIM, Edge_IoU, L1
- **split:** Category of the damage type (e.g., repair / replace).  
- **file:** Image filename.  
- **SSIM:** Structural similarity index (global structure).  
- **Edge_IoU:** Edge overlap between original and generated images (local boundary similarity).  
- **L1:** Pixel-wise intensity difference.

---

### 2️⃣ Classifier Metrics

This subdirectory contains **evaluation results for CNN-based classification models**  
trained on original, generated, and cropped datasets.

- **Contents:**
- `*_cm.png` → Confusion matrix visualization.  
- `*_metrics.json` → Quantitative results including accuracy and F1-score.

- **Structure:**
classifier/
├── original/ → Results on original dataset
├── generation/ → Results on generative (bias-removed) dataset
├── original_crop/ → Results on YOLO-cropped original dataset
└── generation_crop/ → Results on YOLO-cropped generative dataset

- **Typical Metrics:**
```json
{
  "accuracy": 0.945,
  "f1_score": 0.932
}


### 3️⃣ YOLO Cropper Metrics

This subdirectory contains the **detection performance results** of YOLO models used for damage-region cropping.

---

#### 📌 Purpose
To evaluate and compare the detection accuracy of different YOLO versions  
(`YOLOv2`, `YOLOv4`, `YOLOv5`, `YOLOv8`, etc.) used in the cropping process.

---

#### 📂 Files Included
yolo_cropper/
├── yolov2_metrics.csv
├── yolov4_metrics.csv
├── yolov5_metrics.csv
└── yolov8_metrics.csv

---

#### 📊 Metrics Columns
Each CSV file includes quantitative detection metrics as follows:
model, precision, recall, mAP@0.5

---

#### 🧾 Metric Definitions
- **model** → YOLO version or configuration name  
- **precision** → Ratio of correctly detected positive bounding boxes  
- **recall** → Ratio of correctly detected ground-truth regions  
- **mAP@0.5** → Mean Average Precision at IoU threshold 0.5, measuring overall detection performance

---

#### 📈 Example Table

| Model  | Precision | Recall | mAP@0.5 |
|---------|------------|---------|----------|
| YOLOv2  | 0.66       | 0.53    | 34.60 |
| YOLOv4  | 0.74       | 0.70    | 51.09 |
| YOLOv5  | 0.60       | 0.58    | 55.60 |
| YOLOv8  | 0.59       | 0.56    | 56.26 |

---

> 🧠 **Note:**  
> These metrics were obtained after evaluating each YOLO model on private datasets.  
> The results demonstrate the relative detection performance of YOLO versions  
> used in the damage-region cropping stage.
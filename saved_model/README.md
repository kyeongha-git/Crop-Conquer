# 🧩 Saved Model Directory

This directory stores the **best-performing model weights** generated after training.

> ⚠️ **Note:**  
> All models were trained using **private datasets**, and thus the weight files (`.pt`, `.weights`) are **not publicly released**.  
> During training, the best checkpoints are automatically copied here from the `checkpoints/` directory.

---

## 📁 Folder Structure

saved_model/
├── classifier/
│ ├── original/
│ │ ├── vgg_best.pt
│ │ ├── resnet_best.pt
│ │ ├── mobilenet_v2_best.pt
│ │ └── mobilenet_v3_best.pt
│ ├── original_crop/
│ │ ├── vgg_best.pt
│ │ ├── resnet_best.pt
│ │ ├── mobilenet_v2_best.pt
│ │ └── mobilenet_v3_best.pt
│ ├── generation/
│ │ ├── vgg_best.pt
│ │ ├── resnet_best.pt
│ │ ├── mobilenet_v2_best.pt
│ │ └── mobilenet_v3_best.pt
│ └── generation_crop/
│ ├── vgg_best.pt
│ ├── resnet_best.pt
│ ├── mobilenet_v2_best.pt
│ └── mobilenet_v3_best.pt
└── yolo_cropper/
│ ├── yolov2.weights
│ ├── yolov4.weights
│ ├── yolov5.pt
│ ├── yolov8.pt


---

## 📦 Description

- **classifier/** → Contains the **best CNN classification weights** for each dataset variant (`original`, `generation`, `original_crop`, `generation_crop`).  
  Each file (e.g., `mobilenet_v2_best.pt`) is copied from the corresponding `checkpoints/` directory after training.  

- **yolo_cropper/** → Contains **best YOLO weights** (`yolov2`, `yolov4`, `yolov5`, `yolov8`) trained for damage-region detection and cropping.

---

## 🧠 Notes

- These weights are **not included** in the public release due to data confidentiality.  
- When the training pipeline runs, this directory is automatically populated with best-performing weights.  
- Ensure your `config.yaml` points to the correct path: `saved_model/`.
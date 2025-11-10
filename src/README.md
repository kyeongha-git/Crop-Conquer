# 🧩 Source Code Directory (`src/`)

This directory contains the core implementation of the AI pipeline,  
divided into modular components for annotation cleaning, YOLO-based cropping,  
data augmentation, and classification.

---

## 📁 Folder Overview
src/
├── annotation_cleaner/ # Removes human annotations using generative AI
├── yolo_cropper/ # Detects and crops damage regions using YOLO models
├── data_augmentor/ # Splits and augments datasets
├── classifier/ # Trains and evaluates CNN-based classification models
└── main.py # Unified pipeline entry point

---

Each subdirectory contains its own source code, configuration logic, and execution scripts.  
Please refer to the **root README** for details on how to execute the pipeline.
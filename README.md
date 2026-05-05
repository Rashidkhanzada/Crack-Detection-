# 🔍 Automated Crack Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.86%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow.svg)

> A deep learning system that automatically detects cracks in surface images using Convolutional Neural Networks and Transfer Learning (MobileNetV2). Achieves **99.86% accuracy** on 8,000 test images with only 11 wrong predictions.

---

## 📋 Table of Contents

- [About The Project](#about-the-project)
- [Results](#results)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [How To Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Team](#team)

---

## 📖 About The Project

Cracks in infrastructure — buildings, bridges, roads — are a serious safety hazard. Manual inspection is slow, expensive, and unreliable. This project uses **Deep Learning** to automatically classify surface images as either **Crack** or **No Crack** in real time.

We built and compared **two models**:
1. **Baseline CNN** — built from scratch (~80% accuracy)
2. **Transfer Learning with MobileNetV2** — our final model (**99.86% accuracy**)

This project was developed as a semester project for the **Artificial Intelligence & Machine Learning** course.

---

## 🏆 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Our CNN (Baseline) | ~80% | ~79% | ~81% | ~80% |
| MobileNetV2 (Transfer Learning) | **99.86%** | **100%** | **100%** | **100%** |

### Confusion Matrix (8,000 Test Images)

| | Predicted: No Crack | Predicted: Crack |
|---|---|---|
| **Actual: No Crack** | ✅ 4,078 (correct) | ❌ 2 (false alarm) |
| **Actual: Crack** | ❌ 9 (missed) | ✅ 3,911 (correct) |

- **Total correct:** 7,989 out of 8,000
- **Total wrong:** only 11 out of 8,000 **(0.14% error rate)**
- Training auto-stopped at **Epoch 12 / 20** using EarlyStopping
- Train vs Validation accuracy difference: only **0.04%** → No overfitting

---

## 📦 Dataset

- **Source:** [Kaggle — Surface Crack Detection Dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)
- **Total Images:** 40,000
  - Positive (Crack): 20,000 images
  - Negative (No Crack): 20,000 images
- **Split:** 80% Training (32,000) / 20% Validation (8,000)
- **Input Size:** Resized to 128×128 pixels
- **Format:** JPG, RGB color images
- **Balance:** Perfectly balanced — no class bias

**Download the dataset:**
1. Go to [Kaggle Dataset Link](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)
2. Download `archive.zip`
3. Upload to Google Colab when prompted

---

## 🧠 Model Architecture

### Model 2 — MobileNetV2 Transfer Learning (Final Model)

```
Layer                        Output Shape        Parameters
─────────────────────────────────────────────────────────
input_layer (InputLayer)     (None, 128, 128, 3)         0
sequential (Augmentation)    (None, 128, 128, 3)         0
MobileNetV2 (Frozen)         (None, 4, 4, 1280)  2,257,984
GlobalAveragePooling2D       (None, 1280)                0
Dropout (0.2)                (None, 1280)                0
Dense / Output               (None, 1)               1,281
─────────────────────────────────────────────────────────
Total params:        2,259,265
Trainable by us:         1,281  ← Our custom layer
Frozen (pretrained): 2,257,984  ← MobileNetV2 backbone
```

**Data Augmentation applied:**
- Random Horizontal & Vertical Flip
- Random Rotation (±10%)
- Random Zoom (±10%)
- Random Contrast adjustment

**Training Strategy:**
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- EarlyStopping (patience=5)
- ReduceLROnPlateau (factor=0.5, patience=3)
- ModelCheckpoint (saves best model automatically)

---

## 📁 Project Structure

```
crack-detection/
│
├── crack_detection.ipynb       ← Main training notebook (run this)
├── best_crack_model.h5         ← Saved trained model
├── README.md                   ← This file
│
├── dataset/                    ← Image folders (created after extraction)
│   ├── Positive/               ← 20,000 crack images
│   └── Negative/               ← 20,000 no-crack images
│
└── results/                    ← Output graphs and evaluation
    ├── confusion_matrix.png
    ├── training_history.png
    └── training_log.txt
```

---

## ▶️ How To Run

### Step 1 — Open in Google Colab
Click the button or go to [colab.research.google.com](https://colab.research.google.com) and upload `crack_detection.ipynb`

### Step 2 — Upload Dataset
When Cell 1 runs, it will ask you to upload a file. Upload your `archive.zip` downloaded from Kaggle.

### Step 3 — Run All Cells
Go to **Runtime → Run All** and wait. Training takes approximately:
- Google Colab (with GPU): ~1–2 minutes per epoch
- Local CPU only: ~5–15 minutes per epoch

### Step 4 — View Results
After training completes, the notebook will automatically show:
- ✅ Training accuracy & loss graphs
- ✅ Confusion matrix heatmap
- ✅ Full classification report (Precision, Recall, F1)

### Step 5 — Use the Prediction Function
```python
predict_crack("path/to/your/image.jpg")
```
Give it any image and it will output **CRACK DETECTED** or **NO CRACK** with confidence percentage.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.8+ | Programming language |
| TensorFlow 2.x / Keras | Deep learning framework |
| MobileNetV2 | Pretrained model backbone (Transfer Learning) |
| NumPy | Numerical computations |
| Matplotlib | Training graphs and visualizations |
| Seaborn | Confusion matrix heatmap |
| Scikit-learn | Precision, Recall, F1-Score metrics |
| Google Colab | Free cloud GPU for training |
| Kaggle | Dataset source |

---

## 👥 Team

**Course:** Artificial Intelligence & Machine Learning
**Semester Project:** Crack Detection
**Year:** 2026

| Name | Role |
|------|------|
| [Member 1 Name] | Model Development & Training |
| [Member 2 Name] | Data Preprocessing & Evaluation |
| [Member 3 Name] | Documentation & Presentation |

---

## 📊 Key Takeaways

- **Transfer Learning** is far more effective than training from scratch for image classification
- **Data Augmentation** is essential to prevent overfitting on visual datasets
- **MobileNetV2** is ideal for resource-constrained environments (no GPU required locally)
- **EarlyStopping** saved computation — training stopped at epoch 12 instead of running all 20
- A perfectly **balanced dataset** ensures the model doesn't bias toward one class

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> *"AI doesn't replace engineers — it gives them superpowers."*

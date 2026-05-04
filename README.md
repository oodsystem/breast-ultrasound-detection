# Breast Ultrasound Image Detection: A Comparative Study of Multi-task U-Net and EfficientNet-based Classifiers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/FINAL_PROJECT_Breast_Ultrasound_Image_Detection_CSE427.ipynb)

> **CSE427 — BRAC University**
> Tawsif Ahmed · Saiyed Mubasshir Mahmood · Meraj Mahamud

---

## Overview

Breast cancer accounts for over 2.3 million new cases worldwide annually, making early and accurate detection critical for reducing mortality. This project presents a comparative study of two deep learning approaches for breast ultrasound image analysis:

1. **Multi-task U-Net (PyTorch)** — jointly performs lesion segmentation and multi-class classification via a dual-branch encoder–decoder architecture.
2. **EfficientNetB0 + Classical Classifiers (TensorFlow/Sklearn)** — extracts deep features from a pretrained EfficientNetB0 and feeds them into Logistic Regression, Random Forest, and XGBoost classifiers.

Both pipelines are evaluated on the publicly available **BUSI dataset** under a unified preprocessing protocol for a fair and rigorous comparison.

---

## Dataset

**BUSI (Breast Ultrasound Images Dataset)** — available on [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

| Class | Samples |
|-----------|---------|
| Normal | 133 |
| Benign | 437 |
| Malignant | 210 |
| **Total** | **780** |

Images and corresponding segmentation masks are organized into `normal/`, `benign/`, and `malignant/` subdirectories. If no mask file is found for an image, a blank mask is used as a fallback.

**Preprocessing:**
- Resized to `224×224` pixels
- Normalized to `[0, 1]`
- Augmented with rotation, flipping, brightness variation, and elastic deformation to address class imbalance
- Split: **70% train / 20% val / 10% test** (stratified)

---

## Model Architectures

### 1. Multi-task U-Net
A custom U-Net built in PyTorch with two output heads:
- **Segmentation head** — standard encoder–decoder with skip connections; outputs a binary lesion mask using sigmoid activation
- **Classification head** — global average pooling at the bottleneck followed by a fully connected layer; outputs class logits for normal / benign / malignant

**Combined loss:**
```
L = L_seg + L_cls
L_seg = BCE + Dice Loss
L_cls = CrossEntropy Loss
```

### 2. EfficientNetB0 + Classical Classifiers
Grayscale ultrasound images are expanded to 3 channels and passed through EfficientNetB0 pretrained on ImageNet (TensorFlow/Keras). The extracted feature vectors train three lightweight classifiers:
- Logistic Regression
- Random Forest
- XGBoost

---

## Results

Validation performance across all models:

| Model | Accuracy | Precision | Recall | F1-score | Dice |
|----------------------|----------|-----------|--------|----------|------|
| **Multi-task U-Net** | **0.8636**| **0.8428**| **0.8639**| **0.8524**| **0.7263** |
| Logistic Regression | 0.8442 | 0.8478 | 0.8442 | 0.8448 | — |
| XGBoost | 0.8052 | 0.8057 | 0.8052 | 0.8044 | — |
| Random Forest | 0.7987 | 0.8089 | 0.7987 | 0.7939 | — |

The Multi-task U-Net achieves the best classification performance while also providing interpretable segmentation masks — a clinically valuable output. EfficientNet-based pipelines complete training in minutes versus hours, making them suitable for low-resource deployment.

---

## Project Structure

```
├── FINAL_PROJECT_Breast_Ultrasound_Image_Detection_CSE427.ipynb
├── README.md
└── Dataset_BUSI_with_GT/         ← download separately from Kaggle
    ├── normal/
    ├── benign/
    └── malignant/
```

---

## How to Run

### On Google Colab (Recommended)

1. Click the **Open in Colab** badge at the top of this README.
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Upload the BUSI dataset to your Drive under:
   `MyDrive/MLTRAINING/Dataset_BUSI_with_GT/`
4. Run all cells sequentially. A GPU runtime is strongly recommended (`Runtime → Change runtime type → T4 GPU`).

### Dependencies

All required libraries are available on Colab by default. The key packages used are:

| Package | Purpose |
|---------|---------|
| `torch` | Multi-task U-Net training |
| `tensorflow` / `keras` | EfficientNetB0 feature extraction |
| `scikit-learn` | Classical classifiers, metrics |
| `xgboost` | XGBoost classifier |
| `opencv-python` | Image loading and preprocessing |
| `numpy`, `matplotlib` | Data handling and visualization |

---

## Key Findings

- The Multi-task U-Net benefits from **segmentation supervision** — explicitly learning lesion boundaries produces more discriminative feature representations for classification.
- EfficientNet-based classifiers are significantly faster to train but lack spatial modeling capability, struggling with subtle morphological differences between benign and malignant lesions.
- Training for **50 epochs** was essential; early stopping at 2 epochs yielded near-random accuracy (~0.5).
- The **BUSI dataset's class imbalance** (437 benign vs 210 malignant vs 133 normal) required augmentation to prevent overfitting.

---

## Future Work

- Training on larger, multi-institutional datasets for better generalization
- Integration of explainability methods (Grad-CAM, saliency maps, attention visualization)
- Semi-supervised / self-supervised learning to leverage unlabeled ultrasound data
- Model compression (pruning, quantization, knowledge distillation) for mobile/edge deployment
- Federated learning for privacy-preserving multi-hospital training

---

## Citation

If you use this work, please cite:

```
Ahmed, T., Mahmood, S. M., & Mahamud, M. (2024). Breast Ultrasound Image Detection:
A Comparative Study of Multi-task U-Net and EfficientNet-based Classifiers.
CSE427 Project Report, BRAC University.
```

---

## Authors

| Name | Student ID | Email |
|------|-----------|-------|
| Tawsif Ahmed | 21201350 | tawsif.ahmed1@g.bracu.ac.bd |
| Saiyed Mubasshir Mahmood | 22101500 | saiyed.mubasshir.mahmood@g.bracu.ac.bd |
| Meraj Mahamud | 24241318 | md.meraj.mahamud@g.bracu.ac.bd |

---

## License

This project is for academic purposes only. The BUSI dataset is subject to its own terms of use — please refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) for details.

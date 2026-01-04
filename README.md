# Face Deblurring with SRCNN (CelebA Dataset)

This project implements a **deep learning pipeline for face deblurring / super-resolution** using a **SRCNN-inspired convolutional neural network** trained on a **CelebA-based face deblurring dataset**.  
The goal is to reconstruct **high-resolution, sharp face images** from **low-resolution / blurred inputs**.

---

## 📌 Project Overview

- **Task**: Face deblurring & super-resolution
- **Approach**: Super-Resolution CNN (SRCNN-like architecture)
- **Framework**: PyTorch
- **Dataset**: CelebA-based face deblurring dataset (Kaggle)
- **Training**: Mixed Precision (AMP)
- **Evaluation**: MSE, PSNR, SSIM
- **Visualization**: Side-by-side HR / LR / SR comparisons

---

## 🧠 Model Architecture

The model is inspired by **SRCNN**, enhanced with:
- Batch Normalization
- Residual learning (input + output)
- Bicubic upsampling
- Mixed-precision training

### Architecture
```

Input (LR Image)
↓
Conv(9×9) → BN → ReLU
↓
Conv(3×3) → BN → ReLU
↓
Conv(5×5)
↓
Residual Addition (x + output)
↓
Super-Resolved Image

```

---

## 📂 Dataset

Dataset used from Kaggle:

```

emrehakanerdemir/face-deblurring-dataset-using-celeba

```

Directory structure:
```

input/
├── train/
├── test/
└── val/

output/
├── train/
├── test/
└── val/

````

Only **ground-truth (HR) images** are loaded; **LR images are generated on-the-fly** using bicubic downsampling.

---

## ⚙️ Installation

```bash
pip install torch torchvision torchaudio
pip install kagglehub pytorch-msssim torchmetrics matplotlib
````

---

## 🚀 Training Configuration

* **Image size**: `128 × 128`
* **Batch size**: `16`
* **Optimizer**: Adam
* **Learning rate**: `5e-4`
* **Loss**: Mean Squared Error (MSE)
* **Precision**: Mixed precision (torch.cuda.amp)
* **Epochs**: 3 (configurable)

---

## 🧪 Metrics

The following metrics are used for evaluation:

* **MSE** – Mean Squared Error
* **PSNR** – Peak Signal-to-Noise Ratio
* **SSIM** – Structural Similarity Index

```python
SSIM(sr, hr)
PSNR = 10 * log10(1 / MSE)
```

---

## 👁️ Visualization

During validation, the model visualizes:

* High-Resolution Ground Truth (HR)
* Low-Resolution Input (LR)
* Super-Resolved Output (SR)

This allows **qualitative inspection** of reconstruction quality.

---

## 💾 Model Saving

Trained model weights are saved as:

```bash
/kaggle/working/CELEBA2.pth
```

You can reload them using:

```python
model.load_state_dict(torch.load("CELEBA2.pth"))
```

---

## 📈 Results (Qualitative)

* Noticeable improvement over bicubic interpolation
* Sharper facial contours
* Better structural similarity (SSIM)
* Stable training with AMP

---

## 🔬 Future Improvements

* Replace SRCNN with **EDSR / ESRGAN**
* Use **perceptual loss (VGG-based)**
* Add **Charbonnier loss**
* Train with higher scale factors (×4)
* Face-alignment preprocessing
* Real-world motion blur augmentation

---

## 📜 License

This project is intended for **research and educational purposes**.


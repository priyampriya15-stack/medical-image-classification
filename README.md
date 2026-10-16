# Medical Image Classification Using Transfer Learning
### A Comparative Study of ResNet50 and EfficientNetB0 on PneumoniaMNIST

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-5.0-646CFF.svg)](https://vitejs.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

This repository hosts a production-grade full-stack research platform designed for comparative transfer learning evaluation in medical image classification. Specifically, the study benchmarks **ResNet50** (a deep residual network with bottleneck blocks, 23.5M parameters) against **EfficientNetB0** (a compound-scaled lightweight architecture, 4.0M parameters) on pediatric chest radiographs from the standardized **PneumoniaMNIST** benchmark.

The platform includes an asynchronous **FastAPI** backend powering live neural inference, explainable AI (**Grad-CAM** visual attention maps), full evaluation metrics calculation, training telemetry, and an academic **React + TypeScript + Vite** research dashboard.

---

## 2. Research & Clinical Safety Disclaimer

> [!CAUTION]
> **Educational & Research Use Only**
>
> This system is strictly an academic research and educational platform. It is **not clinically validated** and **must not be used for medical diagnosis, clinical decision-making, screening, or patient management**. All model predictions, probabilities, and visual attention overlays represent algorithmic outputs for computer vision research benchmarking.

---

## 3. Key Features

- **Authentication & Role-Based Gate**: Academic researcher login flow with session persistence, route guards, and credential validation.
- **Live Transfer Learning Inference**: Real PyTorch model inference with class probability distributions and latency telemetry on CPU.
- **Explainable AI (Grad-CAM)**: Real-time gradient-weighted class activation mapping targeting convolutional layers (`layer4` on ResNet50, `features[-1]` on EfficientNetB0).
- **Comparative Diagnostic Metrics**: Complete benchmark evaluation including Accuracy, ROC-AUC, Sensitivity / Recall, Specificity, Precision, and F1-Score.
- **Interactive Dataset Explorer**: Direct access to PneumoniaMNIST metadata, train/validation/test splits, class distributions, and base64 radiograph sample retrieval.
- **Dynamic Training Telemetry**: Asynchronous background training monitoring with real-time epoch tracking, loss curves, and checkpoint serialization.
- **Research Analytics & Reporting**: Multi-metric ROC curves, Confusion Matrix, Precision-Recall curves, and automated report exports in JSON, CSV, and PDF formats.

---

## 4. Technology Stack

### Backend
- **Framework**: FastAPI (ASGI)
- **Runtime**: Python 3.14+
- **Deep Learning**: PyTorch, Torchvision
- **Data Processing**: NumPy, Pillow, SciPy, Scikit-learn
- **Server**: Uvicorn
- **Testing**: Pytest, AnyIO, HTTPX

### Frontend
- **Framework**: React 18, TypeScript
- **Bundler & Tooling**: Vite
- **Styling**: Tailwind CSS
- **Visualization & Charts**: Recharts
- **Icons**: Lucide React
- **HTTP Client**: Axios

---

## 5. Repository Structure

```
├── .gitignore                      # Root Git ignore rules (Python, Node, cache, .env)
├── .gitattributes                  # Git LFS tracking for *.pt model checkpoints
├── .env.example                    # Global environment template
├── README.md                       # Comprehensive platform documentation
├── backend/                        # FastAPI REST API & PyTorch ML Pipeline
│   ├── app/
│   │   ├── api/                    # REST API routes (predict, gradcam, training, analytics, dataset)
│   │   ├── core/                   # Configuration, settings, and logging
│   │   ├── evaluation/             # Metrics calculation and benchmark evaluation
│   │   ├── models/                 # PyTorch model definitions & registry
│   │   ├── schemas/                # Pydantic request/response validation schemas
│   │   └── services/               # Inference, Grad-CAM, training, and dataset services
│   ├── checkpoints/                # Model checkpoint storage (Git LFS)
│   │   ├── resnet50_best.pt        # Fine-tuned ResNet50 weights
│   │   └── efficientnet_b0_best.pt # Fine-tuned EfficientNetB0 weights
│   ├── data/                       # Dataset storage
│   │   └── pneumoniamnist.npz      # PneumoniaMNIST archive (28x28 grayscale)
│   ├── outputs/                    # Persisted evaluation reports & audit activity
│   ├── tests/                      # Automated test suite (60 tests)
│   ├── requirements.txt            # Python dependencies
│   ├── verify_production_suite.py  # 23-endpoint end-to-end API test suite
│   └── .env.example                # Backend environment template
└── frontend/                       # React + TypeScript + Vite Research Dashboard
    ├── src/
    │   ├── auth/                   # AuthContext, ProtectedRoute, and authService
    │   ├── components/             # Reusable UI components, cards, layouts, skeletons
    │   ├── pages/                  # Route views (Dashboard, Classifier, GradCam, Analytics, etc.)
    │   ├── services/               # Centralized Axios API client (api.ts)
    │   └── types/                  # TypeScript interface declarations
    ├── package.json                # Frontend dependencies and scripts
    ├── tailwind.config.js          # Design system & color palette tokens
    ├── tsconfig.json               # TypeScript compiler configuration
    ├── vite.config.ts              # Vite server & bundler configuration
    └── .env.example                # Frontend environment template
```

---

## 6. Dataset Information

The benchmark utilizes the **PneumoniaMNIST** dataset from the MedMNIST v2 archive:
- **Task**: Binary classification (`0: Normal`, `1: Pneumonia`).
- **Modality**: Pediatric chest X-ray radiographs.
- **Image Resolution**: $28 \times 28$ pixels (upscaled dynamically to $224 \times 224$ via bicubic interpolation for ImageNet backbones).
- **Total Samples**: 5,856 radiographs.
  - **Training Set**: 4,708 samples (1,214 Normal, 3,494 Pneumonia).
  - **Validation Set**: 524 samples (135 Normal, 389 Pneumonia).
  - **Test Set**: 624 samples (234 Normal, 390 Pneumonia).

---

## 7. Model Architectures & Transfer Learning

| Architecture | Parameters | Feature Extractor | Target Conv Layer (Grad-CAM) |
|---|---|---|---|
| **ResNet50** | 23.5M | Residual Bottleneck blocks (Conv 1x1, 3x3, 1x1) | `layer4` |
| **EfficientNetB0** | 4.0M | MBConv (Inverted Residuals + Squeeze-and-Excitation) | `features[-1]` |

- **Pretraining**: ImageNet weights loaded with head replacement for binary classification.
- **Loss Function**: Binary Cross Entropy with Logits (`BCEWithLogitsLoss`) or Cross Entropy.
- **Optimization**: AdamW with cosine learning rate scheduling.

---

## 8. Setup & Installation

### Prerequisites
- Python 3.10 to 3.14
- Node.js 18+ and npm
- Git & Git LFS

### 1. Clone Repository & Setup Git LFS
```bash
git clone https://github.com/<username>/<repository>.git
cd "Medical Image"
git lfs pull
```

### 2. Backend Setup
```bash
cd backend
python -m venv .venv

# Windows activation:
.venv\Scripts\activate
# Linux/macOS activation:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
cp .env.example .env
```

---

## 9. Running the Application

### Start Backend Service (Port 8001)
```bash
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```
*Backend health check:* [http://127.0.0.1:8001/api/health](http://127.0.0.1:8001/api/health)
*Interactive Swagger Docs:* [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

### Start Frontend Application
```bash
cd frontend
npm run dev
```
*Access Research Platform:* [http://localhost:5174](http://localhost:5174) (or displayed port).

**Default Researcher Login:**
- **Email**: `priya.m@medai.org`
- **Password**: `medai2026`

---

## 10. Automated Testing & Verification

### 1. Backend Pytest Suite (60 Tests)
```bash
cd backend
python -m pytest tests/ -v
```
*Validates API routes, data loaders, model factories, forward passes, Grad-CAM generation, and error handling.*

### 2. Full Production API Suite (23 Endpoints)
```bash
# Ensure backend is running on 127.0.0.1:8001
python backend/verify_production_suite.py
```
*Validates 100% of live endpoints against real payloads and asserts correct inference responses.*

### 3. Frontend Production Build
```bash
cd frontend
npm run build
```
*Validates TypeScript compilation, asset optimization, and Vite bundling.*

---

## 11. API Reference Summary

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | System health, PyTorch version, CPU/GPU telemetry |
| `/api/project/metadata` | `GET` | Platform specifications and dataset description |
| `/api/dataset/info` | `GET` | Sample count, splits, and class breakdown |
| `/api/dataset/samples` | `GET` | Base64-encoded radiograph samples with labels |
| `/api/models` | `GET` | Registered models, parameter counts, and checkpoints |
| `/api/predict/{model_name}` | `POST` | Inference on uploaded radiograph (PNG/JPG) |
| `/api/gradcam` | `POST` | Explainable AI heatmap and overlay generation |
| `/api/evaluation/comparison` | `GET` | Head-to-head metrics comparison |
| `/api/analytics/summary` | `GET` | Summary statistics and top performing model |
| `/api/analytics/roc` | `GET` | ROC curve coordinates for both models |
| `/api/analytics/confusion-matrix` | `GET` | Confusion matrix elements (TP, FP, TN, FN) |
| `/api/train/status` | `GET` | Telemetry of active and past training sessions |
| `/api/activity` | `GET` | Real-time audit events and inference logs |
| `/api/export/report` | `GET` | Comprehensive evaluation report download (PDF) |

---

## 12. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

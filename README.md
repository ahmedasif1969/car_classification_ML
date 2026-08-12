# Car Classification ML Project

This repository contains a complete, end-to-end Machine Learning pipeline for classifying cars. It supports training two different classification backbones: **ResNet-18 (PyTorch)** and **YOLOv8-Classify (Ultralytics)**, serving predictions via a **FastAPI** backend, and visualizing results through a modern **React + Vite** frontend.

---

## 📂 Directory Structure

*   `download_and_prepare.py` - Downloads raw datasets from Azure Blob, standardizes class mappings, deduplicates, and splits the data.
*   `merge_datasets.py` - Implements the dataset deduplication, class mapping, and train/val split logic.
*   `train.py` - Standard script to train ResNet-18, YOLOv8, or both models, and evaluate them on validation data.
*   `api/` - FastAPI backend application to serve predictions.
    *   `api/main.py` - Starts the API server and loads models.
    *   `api/routers/predict.py` - `/predict` endpoint logic.
*   `frontend/` - React (Vite) application providing a graphical user interface for image uploads and prediction comparison.
*   `src/` - Core library functions.
    *   `src/models/` - Python classes wrapping the models.
        *   `cnn_model.py` - PyTorch ResNet-18 implementation.
        *   `yolo_model.py` - Ultralytics YOLOv8 implementation.
    *   `src/inference.py` - Command-line test utility for inference.
*   `weights/` - Pre-trained and fine-tuned model weights (e.g., `resnet18_best.pth`, `yolov8_best.pt`), evaluation metrics, and class maps.
*   `requirements-api.txt` - Python packages required for the API backend.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your system:
*   Python (version 3.8 to 3.12 recommended)
*   Node.js & npm (for the frontend)

### 2. Python Virtual Environment Setup
Create and activate a virtual environment, then install the necessary dependencies:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements-api.txt
pip install ultralytics scikit-learn pandas numpy torch torchvision
```

### 3. Frontend Setup
Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
cd ..
```

---

## 🚀 How to Run

### Step 1: Download & Prepare Dataset
Run the data preparation pipeline. This will download the dataset archive from Azure Blobs, merge old/new raw files, resolve spelling duplicates, split them (80% train, 20% val), and write them to `data/split/`:

```bash
python download_and_prepare.py
```

### Step 2: Train the Classifier Models
Use `train.py` to train your classifier models. You can train either model separately or both at the same time:

```bash
# Train both ResNet-18 and YOLOv8 models (Default: 5 epochs)
python train.py --epochs 10 --model all

# Train ResNet-18 only
python train.py --epochs 10 --model resnet18

# Train YOLOv8-Classify only
python train.py --epochs 10 --model yolo
```
Once training completes, the best weights will be saved to the `weights/` directory:
*   ResNet-18: `weights/resnet18_best.pth` and `weights/resnet18_best_classes.json`
*   YOLOv8: `weights/yolov8_best.pt`
*   Evaluation results summary: `weights/evaluation_metrics.json`

### Step 3: Run the Backend API
Start the FastAPI application using Uvicorn. The server will scan the `weights/` directory and load the available models on startup:

```bash
# Start FastAPI server on port 8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*   **API Healthcheck**: `http://localhost:8000/health`
*   **Interactive API Docs (Swagger UI)**: `http://localhost:8000/docs`

### Step 4: Run the Frontend UI
Start the Vite development server for the user interface:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```
*   Access the web app by visiting: `http://localhost:5173`

---

## 🧪 Testing Inference via CLI
If you want to run a quick test prediction on a single image without running the server, use the command-line utility:

```bash
# Using YOLOv8
python src/inference.py --image path/to/car.jpg --model yolo --weights weights/yolov8_best.pt

# Using ResNet-18
python src/inference.py --image path/to/car.jpg --model resnet18 --weights weights/resnet18_best.pth --class-map weights/resnet18_best_classes.json
```

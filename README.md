# Car Classification Web Application

This repository contains a ready-to-run application for classifying cars. The project serves predictions using two pre-trained backbones: **ResNet-18 (PyTorch)** and **YOLOv8-Classify (Ultralytics)**. It features a **FastAPI** backend and a modern **React + Vite** frontend.

> [!IMPORTANT]
> **Retraining is disabled**: The raw dataset download URLs in Azure have expired, so `download_and_prepare.py` and `train.py` cannot be run out-of-the-box. Instead, the application runs using the pre-trained weights (`resnet18_best.pth` and `yolov8_best.pt`) already included in the `weights/` directory.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have the following installed:
*   Python 3 & pip (`python3` and `pip3`)
*   Node.js & npm (for the frontend)

### 2. Backend Setup (System-wide)
Install the required Python packages directly to your system:

```bash
# Install the core API requirements
pip3 install -r requirements-api.txt

# Install ML libraries (PyTorch, Ultralytics, etc.)
pip3 install ultralytics scikit-learn pandas numpy torch torchvision
```

> [!NOTE]
> If your operating system blocks global pip installations (giving an `externally-managed-environment` error), add the `--break-system-packages` flag to the commands:
> ```bash
> pip3 install --break-system-packages -r requirements-api.txt
> pip3 install --break-system-packages ultralytics scikit-learn pandas numpy torch torchvision
> ```

### 3. Frontend Setup
Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Running the Application

Follow these steps to start the application:

### Step 1: Start the Backend API
Run the FastAPI application from the project root. This will load the pre-trained ResNet-18 and YOLOv8 weights on startup:

```bash
# Start FastAPI server on port 8000
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*   **API Healthcheck**: `http://localhost:8000/health`
*   **Interactive API Docs (Swagger UI)**: `http://localhost:8000/docs`

### Step 2: Start the Frontend UI
In a separate terminal window, start the Vite development server:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```
*   **Web Application URL**: Open your browser and go to `http://localhost:5173`

Once both servers are running, you can upload car images in the web interface and compare predictions between the ResNet-18 and YOLOv8 models.

---

## 🧪 Command-Line Inference Tests
You can also run quick prediction tests on a single image directly from the command line using the pre-trained weights:

```bash
# Test using YOLOv8
python3 src/inference.py --image path/to/car.jpg --model yolo --weights weights/yolov8_best.pt

# Test using ResNet-18
python3 src/inference.py --image path/to/car.jpg --model resnet18 --weights weights/resnet18_best.pth --class-map weights/resnet18_best_classes.json
```

---

## 📂 Directory Layout

*   `api/` - FastAPI backend application.
    *   `main.py` - Main app entrypoint that loads the models.
    *   `routers/predict.py` - Inference `/predict` endpoint handler.
*   `frontend/` - React frontend application.
*   `weights/` - Pre-trained model weights.
    *   `resnet18_best.pth` & `resnet18_best_classes.json` - Custom ResNet-18 weights and class index map.
    *   `yolov8_best.pt` - Custom YOLOv8 weights.
*   `src/` - Helper source code for model definitions and CLI inference.
*   `requirements-api.txt` - Core Python packages required to run the API.

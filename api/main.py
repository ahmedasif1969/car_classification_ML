import sys
import os

# Ensure project root is on the path so `src` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.inference import get_predictor

# ---------------------------------------------------------------------------
# Weights paths — adjust if your weights live elsewhere
# ---------------------------------------------------------------------------
YOLO_WEIGHTS = "weights/yolov8_best.pt"
RESNET_WEIGHTS = "weights/resnet18_best.pth"
RESNET_CLASS_MAP = "weights/resnet18_best_classes.json"

# Global model registry — loaded once at startup
MODELS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    print("Loading models...")

    if os.path.exists(YOLO_WEIGHTS):
        MODELS["yolo"] = get_predictor("yolo", YOLO_WEIGHTS)
        print(f"  ✅ YOLO loaded from {YOLO_WEIGHTS}")
    else:
        print(f"  ⚠️  YOLO weights not found at {YOLO_WEIGHTS} — skipping")

    if os.path.exists(RESNET_WEIGHTS):
        MODELS["resnet18"] = get_predictor("resnet18", RESNET_WEIGHTS, RESNET_CLASS_MAP)
        print(f"  ✅ ResNet-18 loaded from {RESNET_WEIGHTS}")
    else:
        print(f"  ⚠️  ResNet-18 weights not found at {RESNET_WEIGHTS} — skipping")

    print(f"Models ready: {list(MODELS.keys())}")
    yield
    MODELS.clear()
    print("Models unloaded.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Car Classifier API",
    description="Upload a car image and get a classification prediction from YOLO or ResNet-18.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from api.routers.predict import router as predict_router
app.include_router(predict_router, tags=["Inference"])


@app.get("/health", tags=["Health"])
def health():
    """Check server status and which models are loaded."""
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
    }

import io
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image

router = APIRouter()


def get_models():
    """Import the model registry from main app state."""
    from api.main import MODELS
    return MODELS


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("resnet18", enum=["yolo", "resnet18"])
):
    """
    Upload an image and get a car classification prediction.

    - **file**: Image file (jpg, png, webp, etc.)
    - **model**: Which model to use — `yolo` or `resnet18`
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    models = get_models()
    if model not in models:
        raise HTTPException(status_code=503, detail=f"Model '{model}' is not loaded.")

    # Read image bytes and write to a temp file (models expect a file path)
    contents = await file.read()

    try:
        # Validate it's a real image
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Make sure you upload a valid image file.")

    # Save to a temp file so the model can read it
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        img.save(tmp_path)

    try:
        predictor = models[model]
        result = predictor.predict(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content={
        "model": model,
        "class_name": result["class_name"],
        "confidence": round(result["confidence"], 6),
        "class_index": result["class_index"],
        "probabilities": {k: round(v, 6) for k, v in result["probabilities"].items()}
    })

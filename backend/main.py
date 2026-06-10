"""
FastAPI Backend for Cattle Breed Recognition System
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.config import ALLOWED_UPLOAD_EXTENSIONS, INVALID_IMAGE_MESSAGE
from ml.preprocessing import validate_upload_file
from predict import BREED_INFO, predict_breed_with_top_k, predict_full

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cattle Breed Recognition API",
    description="API for identifying Indian cattle and buffalo breeds from images",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads")
MAX_FILE_SIZE = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _save_upload(file: UploadFile) -> tuple[str, str]:
    content = file.file.read()
    ok, message = validate_upload_file(file.filename, len(content), MAX_FILE_SIZE)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    ext = Path(file.filename).suffix.lower()
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as buffer:
        buffer.write(content)
    return filepath, filename


@app.get("/")
async def root():
    return {
        "message": "Cattle Breed Recognition API",
        "version": "2.0.0",
        "allowed_formats": sorted(ALLOWED_UPLOAD_EXTENSIONS),
        "endpoints": {
            "/predict": "POST - Breed prediction with top-3 and validation",
            "/predict/top-k": "POST - Top-K predictions only",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Cattle Breed Recognition API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Validate upload, reject non-cattle images, return breed + confidence + top-3.
    """
    filepath = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        filepath, filename = _save_upload(file)
        logger.info("Uploaded %s", filename)

        result = predict_full(filepath, top_k=3)

        if result.get("rejected"):
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "rejected": True,
                    "message": result.get("message", INVALID_IMAGE_MESSAGE),
                    "filename": filename,
                    "confidence": result.get("confidence", 0.0),
                    "validation_stage": result.get("validation_stage"),
                },
            )

        return JSONResponse(
            {
                "success": True,
                "rejected": False,
                "breed": result["breed"],
                "confidence": result["confidence"],
                "filename": filename,
                "details": result["details"],
                "top_predictions": result.get("top_predictions", []),
                "margin": result.get("margin"),
            }
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


@app.post("/predict/top-k")
async def predict_top_k(file: UploadFile = File(...), k: int = 3):
    filepath = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        if k < 1 or k > 5:
            raise HTTPException(status_code=400, detail="k must be between 1 and 5")

        filepath, filename = _save_upload(file)
        top_k_result = predict_breed_with_top_k(filepath, top_k=k)

        if top_k_result.get("rejected"):
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "rejected": True,
                    "message": top_k_result.get("message", INVALID_IMAGE_MESSAGE),
                    "filename": filename,
                },
            )

        predictions = [
            {"breed": breed, "confidence": conf}
            for breed, conf in top_k_result.get("predictions", [])
        ]
        return JSONResponse(
            {
                "success": True,
                "rejected": False,
                "filename": filename,
                "predictions": predictions,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


@app.get("/breeds")
async def get_breeds():
    breeds = []
    for breed_key, info in BREED_INFO.items():
        if breed_key == "not_cattle":
            continue
        breeds.append({"name": breed_key.replace("_", " ").title(), "key": breed_key, **info})
    return JSONResponse({"success": True, "breeds": breeds, "count": len(breeds)})


if __name__ == "__main__":
    import uvicorn

    print("Starting Cattle Breed Recognition API at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""Production inference: load model, validate, predict with top-k."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ml.config import (
    DEBUG_MODE,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEFAULT_TRAIN_CONFIG,
    INVALID_IMAGE_MESSAGE,
    NOT_CATTLE_CLASS,
)
from ml.model_io import load_trained_model
from ml.preprocessing import normalize_breed_key, preprocess_for_model
from ml.validation import ImageValidationPipeline, apply_validation_gate

logger = logging.getLogger(__name__)

DEBUG_VALIDATION = DEBUG_MODE

_MODEL = None
_CLASS_NAMES: dict[int, str] | None = None
_MODEL_META: dict | None = None


def load_model(force_reload: bool = False):
    global _MODEL, _CLASS_NAMES, _MODEL_META

    if force_reload:
        _MODEL = None
        _CLASS_NAMES = None
        _MODEL_META = None

    if _MODEL is None:
        base = Path(__file__).resolve().parent.parent / "models"
        _MODEL, _CLASS_NAMES, _MODEL_META = load_trained_model(base)

    img_size = int(_MODEL_META.get("img_size", DEFAULT_TRAIN_CONFIG["img_size"]))
    return _MODEL, _CLASS_NAMES, _MODEL_META, (img_size, img_size)


def _run_model(batch: np.ndarray) -> np.ndarray:
    model, _, _, _ = load_model()
    predictions = model.predict(batch, verbose=0)
    if predictions.ndim == 2 and predictions.shape[0] > 1:
        return np.mean(predictions, axis=0)
    return predictions[0]


def _cattle_indices(class_names: dict[int, str]) -> list[int]:
    return [idx for idx, name in class_names.items() if normalize_breed_key(name) != NOT_CATTLE_CLASS]


def _top_k_from_probs(
    pred_array: np.ndarray,
    class_names: dict[int, str],
    top_k: int = 3,
) -> list[dict]:
    cattle_indices = _cattle_indices(class_names)
    cattle_probs = pred_array[cattle_indices]
    cattle_probs = cattle_probs / max(float(np.sum(cattle_probs)), 1e-8)
    order = np.argsort(cattle_probs)[-top_k:][::-1]
    results = []
    for local_idx in order:
        idx = cattle_indices[int(local_idx)]
        results.append(
            {
                "breed": class_names.get(int(idx), "Unknown"),
                "confidence": round(float(cattle_probs[int(local_idx)] * 100), 2),
            }
        )
    return results


def _debug_log_predictions(pred_array: np.ndarray, class_names: dict[int, str], top_k: int = 5) -> None:
    order = np.argsort(pred_array)[-top_k:][::-1]
    print("\n[inference debug] top predictions:")
    for idx in order:
        name = class_names.get(int(idx), "Unknown")
        print(f"  {name}: {float(pred_array[idx]):.4f}")


def predict_production(
    image_path: str | Path,
    breed_info: dict,
    top_k: int = 3,
    bypass_validation: bool = False,
    debug_validation: bool | None = None,
) -> dict:
    path = Path(image_path)
    _, class_names, model_meta, img_size = load_model()
    debug = DEBUG_VALIDATION if debug_validation is None else debug_validation

    validator = ImageValidationPipeline(class_names, model_meta)

    integrity = validator.validate_file(path)
    if not integrity.is_valid:
        return {
            "rejected": True,
            "message": integrity.message or INVALID_IMAGE_MESSAGE,
            "breed": None,
            "confidence": 0.0,
            "details": None,
            "top_predictions": [],
            "margin": 0.0,
        }

    batch = preprocess_for_model(
        path,
        img_size=img_size,
        base_model=model_meta.get("base_model", "EfficientNetB0"),
        use_tta_flip=True,
    )
    pred_array = _run_model(batch)

    if debug:
        _debug_log_predictions(pred_array, class_names)

    if bypass_validation:
        top_predictions = _top_k_from_probs(pred_array, class_names, top_k=top_k)
        best = top_predictions[0]
        breed_key = normalize_breed_key(best["breed"])
        details = breed_info.get(breed_key, _default_details())
        return {
            "rejected": False,
            "message": "",
            "breed": best["breed"],
            "confidence": best["confidence"],
            "details": details,
            "top_predictions": top_predictions,
            "margin": 0.0,
            "validation_stage": "bypassed",
        }

    thresholds = {
        "min_confidence": float(model_meta.get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
        "min_margin": float(model_meta.get("min_margin", DEFAULT_MIN_MARGIN)),
    }

    accepted, reason = apply_validation_gate(
        pred_array, class_names, thresholds=thresholds, debug=debug
    )
    gate = validator.validate_prediction(pred_array, debug=debug)
    top_predictions = _top_k_from_probs(pred_array, class_names, top_k=top_k)

    if not accepted or not gate.is_valid:
        if debug:
            print(f"[inference] REJECTED — reason: {reason or gate.rejection_reason}")
        logger.info(
            "Rejected image: stage=%s reason=%s",
            gate.stage,
            reason or gate.rejection_reason,
        )
        return {
            "rejected": True,
            "message": gate.message or INVALID_IMAGE_MESSAGE,
            "breed": None,
            "confidence": round(gate.max_confidence_pct, 2),
            "details": None,
            "top_predictions": top_predictions,
            "margin": round(gate.margin, 4),
            "validation_stage": gate.stage,
            "rejection_reason": reason or gate.rejection_reason,
        }

    if debug:
        print(f"[inference] ACCEPTED — {reason}")

    best = top_predictions[0]
    breed_key = normalize_breed_key(best["breed"])
    details = breed_info.get(breed_key, _default_details())

    return {
        "rejected": False,
        "message": "",
        "breed": best["breed"],
        "confidence": best["confidence"],
        "details": details,
        "top_predictions": top_predictions,
        "margin": round(gate.margin, 4),
        "validation_stage": "ok",
    }


def _default_details() -> dict:
    return {
        "type": "Cattle breed",
        "milk_yield": "Information not available",
        "region": "Information not available",
        "characteristics": "Information not available",
        "lifespan": "Information not available",
        "diet": "Information not available",
    }


def predict_breed(image_path: str | Path, breed_info: dict) -> dict:
    result = predict_production(image_path, breed_info, top_k=1)
    if result["rejected"]:
        return {
            "rejected": True,
            "message": result["message"],
            "breed": None,
            "confidence": result.get("confidence", 0.0),
            "details": None,
            "margin": result.get("margin", 0.0),
        }
    return {
        "rejected": False,
        "message": "",
        "breed": result["breed"],
        "confidence": result["confidence"],
        "details": result["details"],
        "margin": result.get("margin", 0.0),
        "top_predictions": result.get("top_predictions", []),
    }


def predict_breed_with_top_k(image_path: str | Path, breed_info: dict, top_k: int = 3) -> dict:
    result = predict_production(image_path, breed_info, top_k=top_k)
    if result["rejected"]:
        return {"rejected": True, "message": result["message"], "predictions": []}
    pairs = [(p["breed"], p["confidence"]) for p in result["top_predictions"]]
    return {"rejected": False, "message": "", "predictions": pairs, "top_predictions": result["top_predictions"]}

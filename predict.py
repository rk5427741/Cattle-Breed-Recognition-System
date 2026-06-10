"""
Cattle Breed Recognition - Prediction Module
Delegates to ml.inference while preserving the original public API.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.config import DEBUG_MODE, INVALID_IMAGE_MESSAGE
from ml.inference import (
    load_model,
    predict_breed as _predict_breed,
    predict_breed_with_top_k as _predict_top_k,
    predict_production,
)
from breed_info import BREED_INFO
from ml.preprocessing import preprocess_for_model

INVALID_IMAGE_MSG = INVALID_IMAGE_MESSAGE
NOT_CATTLE_CLASS = "not_cattle"
DEFAULT_MIN_CONFIDENCE = 0.20
DEFAULT_MIN_MARGIN = 0.02
DEFAULT_MAX_NOT_CATTLE_PROB = 0.75


def preprocess_image(image_path: str):
    """Preprocess image for model prediction (backward compatible)."""
    _, class_names, model_meta, img_size = load_model()
    batch = preprocess_for_model(
        image_path,
        img_size=img_size,
        base_model=model_meta.get("base_model", "EfficientNetB0"),
        use_tta_flip=True,
    )
    return batch[:1]


def predict_without_validation(image_path: str, top_k: int = 3) -> dict:
    return predict_production(
        image_path,
        BREED_INFO,
        top_k=top_k,
        bypass_validation=True,
    )


def predict_breed(image_path: str) -> dict:
    return _predict_breed(image_path, BREED_INFO)


def predict_breed_with_top_k(image_path: str, top_k: int = 3) -> dict:
    return _predict_top_k(image_path, BREED_INFO, top_k=top_k)


def predict_full(image_path: str, top_k: int = 3, debug: bool = False) -> dict:
    result = predict_production(
        image_path,
        BREED_INFO,
        top_k=top_k,
        debug_validation=debug or DEBUG_MODE,
    )
    if debug or DEBUG_MODE:
        if result.get("rejected"):
            print(f"\n[debug] REJECTED — stage: {result.get('validation_stage')}")
            print(f"[debug] reason: {result.get('rejection_reason', 'n/a')}")
            print(f"[debug] cattle confidence: {result.get('confidence', 0)}%")
            for row in result.get("top_predictions", []):
                print(f"  - {row['breed']}: {row['confidence']}%")
        else:
            print(f"\n[debug] ACCEPTED — {result.get('breed')} ({result.get('confidence')}%)")
    return result


if __name__ == "__main__":
    import sys as _sys

    debug = "--debug" in _sys.argv or "-d" in _sys.argv
    args = [a for a in _sys.argv[1:] if a not in ("--debug", "-d", "--no-validation")]
    no_validation = "--no-validation" in _sys.argv

    if args:
        test_image = args[0]
        if os.path.exists(test_image):
            if no_validation:
                result = predict_without_validation(test_image, top_k=3)
            else:
                result = predict_full(test_image, top_k=3, debug=debug)
            if result["rejected"]:
                print(f"\nRejected: {result['message']}")
                if debug:
                    print(f"  stage: {result.get('validation_stage')}")
                    print(f"  reason: {result.get('rejection_reason', 'n/a')}")
            else:
                print(f"\nBreed: {result['breed']} ({result['confidence']}%)")
                print("Top predictions:")
                for row in result.get("top_predictions", []):
                    print(f"  - {row['breed']}: {row['confidence']}%")
        else:
            print(f"ERROR: Image not found: {test_image}")
    else:
        print("Usage: python predict.py <image_path> [--debug] [--no-validation]")

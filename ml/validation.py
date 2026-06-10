"""Multi-stage image validation before breed prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.config import (
    ABSOLUTE_MIN_CATTLE_CONFIDENCE,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEFAULT_TRAIN_CONFIG,
    HIGH_NOT_CATTLE_REJECT_PROB,
    INVALID_IMAGE_MESSAGE,
    NOT_CATTLE_CLASS,
)
from ml.preprocessing import normalize_breed_key, validate_image_on_disk, validate_upload_file

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    message: str = ""
    stage: str = "ok"
    max_confidence_pct: float = 0.0
    margin: float = 0.0
    rejection_reason: str = ""


def _cattle_indices_from_names(class_names: dict[int, str] | list[str]) -> list[int]:
    if isinstance(class_names, dict):
        return [
            idx
            for idx, name in class_names.items()
            if normalize_breed_key(name) != NOT_CATTLE_CLASS
        ]
    return [
        idx
        for idx, name in enumerate(class_names)
        if normalize_breed_key(name) != NOT_CATTLE_CLASS
    ]


def _cattle_stats(pred_array: np.ndarray, class_names: dict[int, str] | list[str]) -> dict[str, Any]:
    cattle_indices = _cattle_indices_from_names(class_names)
    if not cattle_indices:
        cattle_indices = list(range(len(pred_array)))

    cattle_probs = pred_array[cattle_indices]
    cattle_probs = cattle_probs / max(float(np.sum(cattle_probs)), 1e-8)

    top_local = int(np.argmax(cattle_probs))
    top_prob = float(cattle_probs[top_local])
    sorted_probs = np.sort(cattle_probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else top_prob

    global_top_idx = int(np.argmax(pred_array))
    if isinstance(class_names, dict):
        global_top_name = normalize_breed_key(class_names.get(global_top_idx, ""))
    else:
        global_top_name = normalize_breed_key(
            class_names[global_top_idx] if global_top_idx < len(class_names) else ""
        )
    global_top_prob = float(pred_array[global_top_idx])

    not_cattle_prob = 0.0
    if isinstance(class_names, dict):
        for idx, name in class_names.items():
            if normalize_breed_key(name) == NOT_CATTLE_CLASS:
                not_cattle_prob = float(pred_array[idx])
                break
    else:
        for idx, name in enumerate(class_names):
            if normalize_breed_key(name) == NOT_CATTLE_CLASS:
                not_cattle_prob = float(pred_array[idx])
                break

    return {
        "top_prob": top_prob,
        "margin": margin,
        "confidence_pct": top_prob * 100.0,
        "global_top_name": global_top_name,
        "global_top_prob": global_top_prob,
        "not_cattle_prob": not_cattle_prob,
    }


def _log_prediction_debug(
    stats: dict[str, Any],
    class_names: dict[int, str] | list[str],
    pred_array: np.ndarray,
    debug: bool,
) -> None:
    if not debug:
        return
    print("\n[validation debug]")
    print(f"  predicted_class: {stats['global_top_name']}")
    print(f"  global_confidence: {stats['global_top_prob']:.4f}")
    print(f"  cattle_confidence: {stats['top_prob']:.4f} ({stats['confidence_pct']:.2f}%)")
    print(f"  margin: {stats['margin']:.4f}")
    print(f"  not_cattle_prob: {stats['not_cattle_prob']:.4f}")
    order = np.argsort(pred_array)[-5:][::-1]
    print("  top_predictions:")
    for idx in order:
        if isinstance(class_names, dict):
            name = class_names.get(int(idx), "Unknown")
        else:
            name = class_names[int(idx)] if int(idx) < len(class_names) else "Unknown"
        print(f"    {name}: {float(pred_array[idx]):.4f}")


def should_accept_prediction(
    pred_array: np.ndarray,
    class_names: dict[int, str] | list[str],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    max_not_cattle_prob: float | None = None,  # noqa: ARG001 — kept for API compatibility
    debug: bool = False,
) -> tuple[bool, str]:
    """
    Accept cattle/buffalo images unless:
      - cattle (renormalized) confidence < 0.20, OR
      - global winner is not_cattle with confidence > 0.70

    Margin threshold is logged only (0.02 default); it does not reject valid cattle.
    """
    stats = _cattle_stats(pred_array, class_names)
    top_prob = stats["top_prob"]
    global_top_name = stats["global_top_name"]
    global_top_prob = stats["global_top_prob"]

    _log_prediction_debug(stats, class_names, pred_array, debug)

    floor = max(min_confidence, ABSOLUTE_MIN_CATTLE_CONFIDENCE)

    if top_prob < floor:
        reason = f"low_cattle_confidence ({top_prob:.3f} < {floor})"
        if debug:
            print(f"  REJECT: {reason}")
        logger.info("Validation reject: %s", reason)
        return False, reason

    if global_top_name == NOT_CATTLE_CLASS and global_top_prob > HIGH_NOT_CATTLE_REJECT_PROB:
        reason = (
            f"not_cattle_high_confidence ({global_top_prob:.3f} > {HIGH_NOT_CATTLE_REJECT_PROB})"
        )
        if debug:
            print(f"  REJECT: {reason}")
        logger.info("Validation reject: %s", reason)
        return False, reason

    if debug:
        if stats["margin"] < min_margin:
            print(f"  NOTE: low margin ({stats['margin']:.4f} < {min_margin}) — still accepted")
        print("  ACCEPT: valid cattle/buffalo image")
    return True, "accepted"


def apply_validation_gate(
    pred_array: np.ndarray,
    class_names: dict[int, str] | list[str],
    thresholds: dict | None = None,
    debug: bool = False,
) -> tuple[bool, str]:
    meta = thresholds or {}
    return should_accept_prediction(
        pred_array,
        class_names,
        min_confidence=float(meta.get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
        min_margin=float(meta.get("min_margin", DEFAULT_MIN_MARGIN)),
        debug=debug,
    )


class ImageValidationPipeline:
    def __init__(
        self,
        class_names: dict[int, str],
        model_meta: dict | None = None,
        reject_confidence_pct: float | None = None,
    ):
        meta = model_meta or {}
        self.class_names = class_names
        self.min_confidence = float(
            meta.get("min_confidence", DEFAULT_TRAIN_CONFIG["default_min_confidence"])
        )
        self.min_margin = float(
            meta.get("min_margin", DEFAULT_TRAIN_CONFIG["default_min_margin"])
        )
        self.reject_confidence_pct = float(
            reject_confidence_pct
            if reject_confidence_pct is not None
            else meta.get(
                "reject_confidence_pct",
                DEFAULT_TRAIN_CONFIG["default_reject_confidence_pct"],
            )
        )
        self.invalid_message = INVALID_IMAGE_MESSAGE

    def validate_upload(self, filename: str | None, file_size: int) -> ValidationResult:
        ok, msg = validate_upload_file(filename, file_size)
        if not ok:
            return ValidationResult(False, msg, stage="upload")
        return ValidationResult(True)

    def validate_file(self, image_path: str | Path) -> ValidationResult:
        ok, msg = validate_image_on_disk(image_path)
        if not ok:
            return ValidationResult(False, msg or self.invalid_message, stage="integrity")
        return ValidationResult(True)

    def validate_prediction(
        self,
        pred_array: np.ndarray,
        debug: bool = False,
    ) -> ValidationResult:
        stats = _cattle_stats(pred_array, self.class_names)
        accepted, reason = should_accept_prediction(
            pred_array,
            self.class_names,
            min_confidence=self.min_confidence,
            min_margin=self.min_margin,
            debug=debug,
        )

        if not accepted:
            stage = "not_cattle_class" if "not_cattle" in reason else "low_confidence"
            return ValidationResult(
                False,
                self.invalid_message,
                stage=stage,
                max_confidence_pct=stats["confidence_pct"],
                margin=stats["margin"],
                rejection_reason=reason,
            )

        return ValidationResult(
            True,
            "",
            stage="ok",
            max_confidence_pct=stats["confidence_pct"],
            margin=stats["margin"],
            rejection_reason=reason,
        )

    def run_full(
        self,
        filename: str | None,
        file_size: int,
        image_path: str | Path,
        pred_array: np.ndarray | None = None,
        debug: bool = False,
    ) -> ValidationResult:
        upload = self.validate_upload(filename, file_size)
        if not upload.is_valid:
            return upload

        integrity = self.validate_file(image_path)
        if not integrity.is_valid:
            return integrity

        if pred_array is not None:
            return self.validate_prediction(pred_array, debug=debug)

        return ValidationResult(True)

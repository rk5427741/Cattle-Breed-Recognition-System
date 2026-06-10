"""Shared configuration constants."""

from __future__ import annotations

INVALID_IMAGE_MESSAGE = (
    "Invalid image. Please upload an image of Indian cattle or buffalo."
)

NOT_CATTLE_CLASS = "not_cattle"

ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_TRAIN_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

SUPPORTED_BACKBONES = ("EfficientNetB0", "ResNet50", "MobileNetV2")

# Validation: reject only on very low cattle confidence or high-confidence not_cattle
DEFAULT_MIN_CONFIDENCE = 0.20
DEFAULT_MIN_MARGIN = 0.02
DEFAULT_MAX_NOT_CATTLE_PROB = 0.75
ABSOLUTE_MIN_CATTLE_CONFIDENCE = 0.20
HIGH_NOT_CATTLE_REJECT_PROB = 0.70

DEBUG_MODE = False

DEFAULT_TRAIN_CONFIG: dict = {
    "img_size": 224,
    "batch_size": 32,
    "epochs": 35,
    "seed": 1337,
    "base_model": "EfficientNetB0",
    "train_dir": "data_breeds/train",
    "val_dir": "data_breeds/val",
    "model_save_path": "models/cattle_breed_model.keras",
    "weights_save_path": "models/cattle_breed_weights.weights.h5",
    "class_names_save_path": "models/class_names.json",
    "model_meta_save_path": "models/model_meta.json",
    "logs_dir": "models/logs",
    "freeze_epochs": 10,
    "unfreeze_layers": 40,
    "lr": 3e-4,
    "fine_tune_lr": 1e-5,
    "label_smoothing": 0.1,
    "dropout": 0.35,
    "early_stopping_monitor": "val_loss",
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.001,
    "default_min_confidence": DEFAULT_MIN_CONFIDENCE,
    "default_min_margin": DEFAULT_MIN_MARGIN,
    "default_max_not_cattle_prob": DEFAULT_MAX_NOT_CATTLE_PROB,
    "default_reject_confidence_pct": 20.0,
    "not_cattle_target_fraction": 0.18,
}

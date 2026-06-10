"""Image loading, normalization, and upload validation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image

from ml.config import ALLOWED_UPLOAD_EXTENSIONS, ALLOWED_TRAIN_EXTENSIONS
from ml.data_utils import is_valid_image

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


def normalize_breed_key(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_").replace("-", "_")


def get_preprocess_fn(base_model_name: str):
    model_name = (base_model_name or "").strip().lower()
    if model_name == "resnet50":
        return keras.applications.resnet.preprocess_input
    if model_name == "mobilenetv2":
        return keras.applications.mobilenet_v2.preprocess_input
    return keras.applications.efficientnet.preprocess_input


def validate_upload_file(
    filename: str | None,
    file_size: int,
    max_bytes: int = 16 * 1024 * 1024,
) -> tuple[bool, str]:
    if not filename:
        return False, "No file provided."
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        return False, "Invalid file type. Allowed formats: JPG, JPEG, PNG only."
    if file_size <= 0:
        return False, "Empty file uploaded."
    if file_size > max_bytes:
        return False, f"File size exceeds {max_bytes // (1024 * 1024)}MB limit."
    return True, ""


def validate_image_on_disk(image_path: str | Path) -> tuple[bool, str]:
    path = Path(image_path)
    if not path.exists():
        return False, "Image file not found."
    if path.suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS and path.suffix.lower() not in ALLOWED_TRAIN_EXTENSIONS:
        return False, "Unsupported image format."
    if not is_valid_image(path):
        return False, "Corrupted or unreadable image file."
    return True, ""


def load_rgb_image(image_path: str | Path, target_size: tuple[int, int]) -> np.ndarray:
    """Load image as float32 RGB array in [0, 255]."""
    path = Path(image_path)
    if Image is not None:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            return np.asarray(img, dtype=np.float32)

    img = keras_image.load_img(path, target_size=target_size)
    return keras_image.img_to_array(img).astype(np.float32)


def preprocess_for_model(
    image_path: str | Path,
    img_size: tuple[int, int],
    base_model: str,
    use_tta_flip: bool = True,
) -> np.ndarray:
    """
    Resize, convert to RGB, apply backbone-specific normalization, return batch.
    Test-time augmentation: optional horizontal flip average (matches existing pipeline).
    """
    ok, reason = validate_image_on_disk(image_path)
    if not ok:
        raise ValueError(reason)

    preprocess_fn = get_preprocess_fn(base_model)
    rgb = load_rgb_image(image_path, img_size)
    batch = np.expand_dims(preprocess_fn(rgb.copy()), axis=0).astype(np.float32)

    if use_tta_flip:
        flipped_rgb = np.flip(rgb, axis=1)
        flipped = np.expand_dims(preprocess_fn(flipped_rgb), axis=0).astype(np.float32)
        batch = np.concatenate([batch, flipped], axis=0)

    return batch


def make_tf_datasets(
    train_dir: str,
    val_dir: str,
    class_names: list[str],
    img_size: int,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )
    autotune = tf.data.AUTOTUNE
    return train_ds.cache().prefetch(autotune), val_ds.cache().prefetch(autotune)

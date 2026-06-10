"""Save and load model artifacts (avoids corrupt partial .h5 checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import tensorflow as tf
from tensorflow import keras

from ml.model_builder import BackbonePreprocessLayer, build_classifier

MIN_VALID_BYTES = 100_000  # reject empty / partial checkpoint files


def get_custom_objects() -> dict:
    return {
        "BackbonePreprocessLayer": BackbonePreprocessLayer,
        "ml>BackbonePreprocessLayer": BackbonePreprocessLayer,
    }


def is_valid_model_file(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size < MIN_VALID_BYTES:
        return False
    if path.suffix.lower() == ".h5":
        try:
            with h5py.File(path, "r") as handle:
                return len(handle.keys()) > 0 or "model_config" in handle.attrs
        except Exception:
            return False
    return True


def model_paths(model_dir: str | Path) -> dict[str, Path]:
    root = Path(model_dir)
    return {
        "keras": root / "cattle_breed_model.keras",
        "weights": root / "cattle_breed_weights.weights.h5",
        "legacy_h5": root / "cattle_breed_model.h5",
        "meta": root / "model_meta.json",
        "classes": root / "class_names.json",
    }


def load_class_names(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8") as handle:
        indices = json.load(handle)
    return {int(k): v for k, v in indices.items()}


def load_meta(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_from_meta(meta: dict, for_inference: bool = False) -> keras.Model:
    class_names = meta.get("class_names") or []
    num_classes = int(meta.get("num_classes", len(class_names)))
    img_size = int(meta.get("img_size", 224))
    base_model = meta.get("base_model", "EfficientNetB0")
    dropout = float(meta.get("dropout", 0.35))
    model, _ = build_classifier(
        num_classes,
        img_size,
        base_model,
        dropout=dropout,
        for_inference=for_inference,
    )
    return model


def load_trained_model(model_dir: str | Path = "models") -> tuple[keras.Model, dict[int, str], dict]:
    paths = model_paths(model_dir)
    custom_objects = get_custom_objects()

    if not paths["classes"].exists():
        raise FileNotFoundError(f"Missing {paths['classes']}. Run: python train_model.py")

    class_names = load_class_names(paths["classes"])
    meta = load_meta(paths["meta"]) if paths["meta"].exists() else {}

    # 1) Native Keras format (preferred)
    if is_valid_model_file(paths["keras"]):
        model = keras.models.load_model(str(paths["keras"]), compile=False, custom_objects=custom_objects)
        return model, class_names, meta

    # 2) Weights + rebuilt architecture
    if is_valid_model_file(paths["weights"]):
        if not meta:
            raise FileNotFoundError(f"Missing {paths['meta']} required to rebuild model from weights.")
        # Weights come from the training graph (includes augment layer names).
        model = build_from_meta(meta, for_inference=False)
        model.load_weights(str(paths["weights"]))
        return model, class_names, meta

    # 3) Legacy full .h5
    if is_valid_model_file(paths["legacy_h5"]):
        model = keras.models.load_model(
            str(paths["legacy_h5"]),
            compile=False,
            custom_objects=custom_objects,
        )
        return model, class_names, meta

    raise FileNotFoundError(
        "No valid trained model found in models/. "
        "The .h5 file may be corrupted from an interrupted training run. "
        "Run: python train_model.py"
    )


def save_trained_model(
    model: keras.Model,
    model_dir: str | Path = "models",
    meta: dict | None = None,
) -> None:
    """Persist full model (.keras) and weights. Augment layers are inactive at inference."""
    paths = model_paths(model_dir)
    paths["keras"].parent.mkdir(parents=True, exist_ok=True)
    model.save(str(paths["keras"]))
    model.save_weights(str(paths["weights"]))
    print(f"Saved: {paths['keras']}")
    print(f"Saved: {paths['weights']}")


def export_keras_from_weights(model_dir: str | Path = "models") -> Path:
    """Build .keras from checkpoint weights (run after training if export step failed)."""
    model, _, meta = load_trained_model(model_dir)
    paths = model_paths(model_dir)
    model.save(str(paths["keras"]))
    print(f"Exported: {paths['keras']}")
    return paths["keras"]

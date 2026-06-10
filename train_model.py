"""
Cattle Breed Recognition - Model Training
Uses modular ml/ package; keeps the same CLI and output paths as before.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from ml.config import DEFAULT_TRAIN_CONFIG, SUPPORTED_BACKBONES
from ml.data_utils import calculate_class_weights, list_class_names, print_dataset_stats
from ml.model_builder import build_classifier, calibrate_rejection_thresholds
from ml.model_io import save_trained_model
from ml.preprocessing import make_tf_datasets

CONFIG = dict(DEFAULT_TRAIN_CONFIG)


def _build_callbacks() -> list:
    os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)
    os.makedirs(CONFIG["logs_dir"], exist_ok=True)

    monitor = str(CONFIG["early_stopping_monitor"])
    patience = int(CONFIG["early_stopping_patience"])
    min_delta = float(CONFIG["early_stopping_min_delta"])
    mode = "min" if "loss" in monitor else "max"

    weights_path = CONFIG.get("weights_save_path", "models/cattle_breed_weights.weights.h5")

    callbacks: list = [
        ModelCheckpoint(
            weights_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
            mode=mode,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(2, patience // 2),
            min_delta=min_delta,
            min_lr=1e-7,
            verbose=1,
            mode=mode,
        ),
    ]

    try:
        from tensorflow.keras.callbacks import TensorBoard

        callbacks.append(TensorBoard(log_dir=CONFIG["logs_dir"]))
    except Exception:
        print("NOTE: TensorBoard not available; continuing without TensorBoard logging.")

    return callbacks


def train_model() -> None:
    print("\n" + "=" * 80)
    print("TRAINING WITH FULL DATASET")
    print("=" * 80)
    print(f"   Base Model: {CONFIG['base_model']} (supported: {', '.join(SUPPORTED_BACKBONES)})")
    print(f"   Image Size: {CONFIG['img_size']}x{CONFIG['img_size']}")
    print(f"   Batch Size: {CONFIG['batch_size']}")
    print(f"   Epochs: {CONFIG['epochs']} (freeze: {CONFIG['freeze_epochs']})")
    print("=" * 80)

    total_train, total_val = print_dataset_stats(CONFIG["train_dir"], CONFIG["val_dir"])
    if total_train <= 0 or total_val <= 0:
        raise ValueError("Dataset is empty. Add images to data_breeds/train and data_breeds/val.")

    train_ds, val_ds, class_names = _make_datasets()
    num_classes = len(class_names)

    os.makedirs(os.path.dirname(CONFIG["class_names_save_path"]), exist_ok=True)
    with open(CONFIG["class_names_save_path"], "w", encoding="utf-8") as handle:
        json.dump({i: name for i, name in enumerate(class_names)}, handle, indent=2)

    model_meta = {
        "base_model": CONFIG["base_model"],
        "img_size": int(CONFIG["img_size"]),
        "class_names": class_names,
        "num_classes": num_classes,
        "dropout": float(CONFIG["dropout"]),
        "reject_confidence_pct": DEFAULT_TRAIN_CONFIG["default_reject_confidence_pct"],
        "validation_approach": "not_cattle_class + calibrated confidence/margin thresholds",
    }
    with open(CONFIG["model_meta_save_path"], "w", encoding="utf-8") as handle:
        json.dump(model_meta, handle, indent=2)

    print("\nBuilding model...")
    model, backbone = build_classifier(
        num_classes,
        int(CONFIG["img_size"]),
        CONFIG["base_model"],
        dropout=float(CONFIG["dropout"]),
    )
    model.summary()

    class_weights_dict = calculate_class_weights(CONFIG["train_dir"], class_names)
    if class_weights_dict:
        print(f"Class weights: {len(class_weights_dict)} classes")
    else:
        print("WARNING: Could not calculate class weights")

    loss = keras.losses.CategoricalCrossentropy(label_smoothing=float(CONFIG["label_smoothing"]))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(CONFIG["lr"])),
        loss=loss,
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name="top3"),
        ],
    )

    callbacks = _build_callbacks()
    freeze_epochs = int(min(max(1, CONFIG["freeze_epochs"]), CONFIG["epochs"]))

    print("\nPHASE 1: TRAIN CLASSIFIER HEAD (frozen backbone)")
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=freeze_epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1,
    )

    phase1_epochs = len(history_1.history.get("loss", []))
    history_2 = None
    total_epochs = int(CONFIG["epochs"])

    if total_epochs > phase1_epochs:
        print("\nPHASE 2: FINE-TUNE BACKBONE")
        unfreeze_layers = int(max(0, CONFIG["unfreeze_layers"]))
        backbone.trainable = True
        if unfreeze_layers > 0:
            for layer in backbone.layers[:-unfreeze_layers]:
                layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(CONFIG["fine_tune_lr"])),
            loss=loss,
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name="top3"),
            ],
        )
        history_2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=phase1_epochs,
            epochs=total_epochs,
            callbacks=callbacks,
            class_weight=class_weights_dict,
            verbose=1,
        )

    final_hist = history_2.history if history_2 is not None else history_1.history
    if final_hist and "val_accuracy" in final_hist:
        best = max(final_hist["val_accuracy"])
        print(f"\nBest validation accuracy: {best * 100:.2f}%")

    print("\nCalibrating rejection thresholds on validation set...")
    thresholds = calibrate_rejection_thresholds(model, val_ds, class_names)
    model_meta.update(thresholds)
    with open(CONFIG["model_meta_save_path"], "w", encoding="utf-8") as handle:
        json.dump(model_meta, handle, indent=2)
    print(f"Rejection thresholds saved: {thresholds}")

    print("\nSaving model artifacts (.keras + weights)...")
    save_trained_model(model, Path(CONFIG["model_meta_save_path"]).parent, meta=model_meta)
    print("Next: python evaluate_model.py")


def _make_datasets():
    class_names = list_class_names(CONFIG["train_dir"])
    train_ds, val_ds = make_tf_datasets(
        CONFIG["train_dir"],
        CONFIG["val_dir"],
        class_names,
        int(CONFIG["img_size"]),
        int(CONFIG["batch_size"]),
        int(CONFIG["seed"]),
    )
    return train_ds, val_ds, class_names


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CATTLE BREED RECOGNITION - MODEL TRAINING")
    print("=" * 80)

    parser = argparse.ArgumentParser(description="Train cattle breed recognition model")
    parser.add_argument("--base-model", type=str, default=CONFIG["base_model"], help="EfficientNetB0, ResNet50, or MobileNetV2")
    parser.add_argument("--img-size", type=int, default=CONFIG["img_size"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--freeze-epochs", type=int, default=CONFIG["freeze_epochs"])
    parser.add_argument("--unfreeze-layers", type=int, default=CONFIG["unfreeze_layers"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--fine-tune-lr", type=float, default=CONFIG["fine_tune_lr"])
    parser.add_argument("--label-smoothing", type=float, default=CONFIG["label_smoothing"])
    parser.add_argument("--dropout", type=float, default=CONFIG["dropout"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--train-dir", type=str, default=CONFIG["train_dir"])
    parser.add_argument("--val-dir", type=str, default=CONFIG["val_dir"])
    parser.add_argument("--model-save-path", type=str, default=CONFIG["model_save_path"])
    parser.add_argument("--class-names-save-path", type=str, default=CONFIG["class_names_save_path"])
    parser.add_argument("--model-meta-save-path", type=str, default=CONFIG["model_meta_save_path"])
    parser.add_argument("--logs-dir", type=str, default=CONFIG["logs_dir"])
    parser.add_argument("--early-stopping-patience", type=int, default=CONFIG["early_stopping_patience"])
    parser.add_argument("--early-stopping-monitor", type=str, default=CONFIG["early_stopping_monitor"])
    parser.add_argument("--early-stopping-min-delta", type=float, default=CONFIG["early_stopping_min_delta"])
    args = parser.parse_args()

    for key, arg_val in [
        ("base_model", args.base_model),
        ("img_size", max(64, args.img_size)),
        ("epochs", max(1, args.epochs)),
        ("freeze_epochs", max(1, args.freeze_epochs)),
        ("unfreeze_layers", max(0, args.unfreeze_layers)),
        ("batch_size", max(1, args.batch_size)),
        ("lr", args.lr),
        ("fine_tune_lr", args.fine_tune_lr),
        ("label_smoothing", args.label_smoothing),
        ("dropout", args.dropout),
        ("seed", args.seed),
        ("train_dir", args.train_dir),
        ("val_dir", args.val_dir),
        ("model_save_path", args.model_save_path),
        ("class_names_save_path", args.class_names_save_path),
        ("model_meta_save_path", args.model_meta_save_path),
        ("logs_dir", args.logs_dir),
        ("early_stopping_patience", max(1, args.early_stopping_patience)),
        ("early_stopping_monitor", args.early_stopping_monitor),
        ("early_stopping_min_delta", max(0.0, args.early_stopping_min_delta)),
    ]:
        CONFIG[key] = arg_val

    if CONFIG["base_model"] not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported --base-model. Choose from: {SUPPORTED_BACKBONES}")


    try:
        tf.keras.utils.set_random_seed(CONFIG["seed"])
        train_model()
    except KeyboardInterrupt:
        print("\nWARNING: Training interrupted by user.")
    except Exception as exc:
        print(f"\nERROR: Training failed: {exc}")
        raise

"""
Compare backbone architectures (EfficientNetB0, MobileNetV2, ResNet50) on validation data.
Uses fewer epochs for a quick benchmark — run full training with train_model.py for production.

Run: python compare_models.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from ml.config import DEFAULT_TRAIN_CONFIG, SUPPORTED_BACKBONES
from ml.data_utils import list_class_names, print_dataset_stats
from ml.metrics import evaluate_predictions, predict_dataset
from ml.model_builder import build_classifier, calibrate_rejection_thresholds
from ml.preprocessing import make_tf_datasets


def _quick_train(
    backbone_name: str,
    train_dir: str,
    val_dir: str,
    img_size: int,
    batch_size: int,
    epochs: int,
    seed: int,
) -> dict:
    class_names = list_class_names(train_dir)
    train_ds, val_ds = make_tf_datasets(train_dir, val_dir, class_names, img_size, batch_size, seed)
    num_classes = len(class_names)

    model, backbone = build_classifier(num_classes, img_size, backbone_name, dropout=0.35)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)

    y_true, y_pred = predict_dataset(model, val_ds)
    metrics = evaluate_predictions(y_true, y_pred, class_names, exclude_classes={"not_cattle"})
    thresholds = calibrate_rejection_thresholds(model, val_ds, class_names)
    metrics["thresholds"] = thresholds
    metrics["backbone"] = backbone_name
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CNN backbones")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_CONFIG["train_dir"])
    parser.add_argument("--val-dir", default=DEFAULT_TRAIN_CONFIG["val_dir"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8, help="Quick benchmark epochs per model")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="models/backbone_comparison.json")
    args = parser.parse_args()

    print_dataset_stats(args.train_dir, args.val_dir)
    results = []
    for backbone in SUPPORTED_BACKBONES:
        print(f"\n>>> Training benchmark: {backbone} ({args.epochs} epochs)")
        try:
            metrics = _quick_train(
                backbone,
                args.train_dir,
                args.val_dir,
                args.img_size,
                args.batch_size,
                args.epochs,
                args.seed,
            )
            results.append(metrics)
            print(
                f"    {backbone}: acc={metrics['accuracy']*100:.2f}% "
                f"F1={metrics['f1_macro']*100:.2f}%"
            )
        except Exception as exc:
            print(f"    {backbone} failed: {exc}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = [
        {
            "backbone": r["backbone"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
        }
        for r in results
    ]
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 60)
    print("BACKBONE COMPARISON (validation set, cattle classes)")
    print("=" * 60)
    for row in sorted(summary, key=lambda x: x["f1_macro"], reverse=True):
        print(
            f"{row['backbone']:16s}  Acc={row['accuracy']*100:6.2f}%  "
            f"F1={row['f1_macro']*100:6.2f}%"
        )
    print(f"\nSaved: {out}")
    print("Tip: Use the best backbone with full training: python train_model.py --base-model <name>")
    print("=" * 60)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(1337)
    main()

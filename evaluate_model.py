"""
Evaluate trained breed classifier on the validation set.
Run: python evaluate_model.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from ml.data_utils import list_class_names, print_dataset_stats
from ml.metrics import evaluate_predictions, plot_confusion_matrix, predict_dataset, save_evaluation_report
from ml.model_builder import get_backbone_and_preprocess
from ml.preprocessing import make_tf_datasets


from ml.model_io import load_trained_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cattle breed model")
    parser.add_argument("--model-path", default="models/cattle_breed_model.h5")
    parser.add_argument("--train-dir", default="data_breeds/train")
    parser.add_argument("--val-dir", default="data_breeds/val")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="models/evaluation")
    parser.add_argument("--exclude-not-cattle", action="store_true", help="Metrics on cattle breeds only")
    args = parser.parse_args()

    meta_path = Path("models/model_meta.json")
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        args.img_size = int(meta.get("img_size", args.img_size))

    print_dataset_stats(args.train_dir, args.val_dir)
    class_names = list_class_names(args.train_dir)
    _, val_ds = make_tf_datasets(
        args.train_dir,
        args.val_dir,
        class_names,
        args.img_size,
        args.batch_size,
        seed=1337,
    )

    print("\nLoading model from models/ ...")
    model, _, _ = load_trained_model("models")

    print("Running validation predictions...")
    y_true, y_pred = predict_dataset(model, val_ds)

    exclude = {"not_cattle"} if args.exclude_not_cattle else set()
    metrics = evaluate_predictions(y_true, y_pred, class_names, exclude_classes=exclude)

    report_path = save_evaluation_report(metrics, args.output_dir)
    cm_path = Path(args.output_dir) / "confusion_matrix.png"
    plot_confusion_matrix(metrics, cm_path)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Samples evaluated : {metrics['samples_evaluated']}")
    print(f"Accuracy          : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision (macro) : {metrics['precision_macro'] * 100:.2f}%")
    print(f"Recall (macro)    : {metrics['recall_macro'] * 100:.2f}%")
    print(f"F1 (macro)        : {metrics['f1_macro'] * 100:.2f}%")
    print(f"F1 (weighted)     : {metrics['f1_weighted'] * 100:.2f}%")
    print("\nPer-class accuracy:")
    for name, acc in sorted(metrics["per_class_accuracy"].items()):
        print(f"  {name:24s} {acc * 100:6.2f}%")
    print(f"\nReport saved to: {report_path}")
    if cm_path.exists():
        print(f"Confusion matrix: {cm_path}")
    print("=" * 60)


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(1337)
    main()

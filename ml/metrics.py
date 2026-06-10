"""Evaluation metrics for breed classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    exclude_classes: set[str] | None = None,
) -> dict:
    exclude = exclude_classes or {"not_cattle"}
    mask = np.array([class_names[i] not in exclude for i in y_true])
    if not np.any(mask):
        mask = np.ones_like(y_true, dtype=bool)

    yt = y_true[mask]
    yp = y_pred[mask]
    labels = sorted(set(yt.tolist()))

    report = classification_report(
        yt,
        yp,
        labels=labels,
        target_names=[class_names[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )

    per_class_acc: dict[str, float] = {}
    for idx in labels:
        class_mask = yt == idx
        if np.any(class_mask):
            per_class_acc[class_names[idx]] = float(np.mean(yp[class_mask] == idx))

    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision_macro": float(precision_score(yt, yp, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(yt, yp, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(yt, yp, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
        "confusion_labels": [class_names[i] for i in labels],
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
        "samples_evaluated": int(len(yt)),
    }


def predict_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch_x, batch_y in dataset:
        probs = model.predict(batch_x, verbose=0)
        y_true.extend(np.argmax(batch_y.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    return np.array(y_true), np.array(y_pred)


def save_evaluation_report(metrics: dict, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return report_path


def plot_confusion_matrix(metrics: dict, output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    matrix = np.array(metrics["confusion_matrix"])
    labels = metrics["confusion_labels"]

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) * 0.5)))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], "d"), ha="center", va="center", color="white" if matrix[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

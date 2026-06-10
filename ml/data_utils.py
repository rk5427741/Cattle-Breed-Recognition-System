"""Dataset scanning, class weights, and file integrity helpers."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from ml.config import ALLOWED_TRAIN_EXTENSIONS

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


def gather_images(directory: Path, extensions: tuple[str, ...] | None = None) -> list[str]:
    exts = extensions or ALLOWED_TRAIN_EXTENSIONS
    images: list[str] = []
    if not directory.exists():
        return images
    for ext in exts:
        images.extend([str(p) for p in directory.rglob(f"*{ext}")])
        images.extend([str(p) for p in directory.rglob(f"*{ext.upper()}")])
    return images


def list_class_names(train_dir: str | Path) -> list[str]:
    root = Path(train_dir)
    if not root.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class subfolders found inside: {train_dir}")
    return class_names


def file_hash(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_image(path: Path) -> bool:
    """Return False for corrupted or unreadable image files."""
    if not path.exists() or path.stat().st_size < 512:
        return False
    if Image is None:
        return True
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True
    except Exception:
        return False


def scan_corrupted_images(root: Path) -> list[Path]:
    corrupted: list[Path] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {e.lstrip(".") for e in ALLOWED_TRAIN_EXTENSIONS}:
            continue
        if path.is_file() and not is_valid_image(path):
            corrupted.append(path)
    return corrupted


def find_duplicate_groups(root: Path) -> dict[str, list[Path]]:
    """Group files by content hash (potential duplicates)."""
    buckets: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {ext for ext in ALLOWED_TRAIN_EXTENSIONS}:
            continue
        digest = file_hash(path)
        buckets.setdefault(digest, []).append(path)
    return {digest: paths for digest, paths in buckets.items() if len(paths) > 1}


def calculate_class_weights(train_dir: str | Path, class_names: list[str]) -> dict[int, float] | None:
    class_counts: Counter[int] = Counter()
    train_root = Path(train_dir)
    for idx, name in enumerate(class_names):
        count = len(gather_images(train_root / name))
        if count > 0:
            class_counts[idx] = count

    if not class_counts:
        return None

    total_samples = sum(class_counts.values())
    num_classes = len(class_names)
    weights: dict[int, float] = {}
    for class_idx in range(num_classes):
        count = class_counts.get(class_idx, 0)
        weights[class_idx] = (total_samples / (num_classes * count)) if count > 0 else 1.0
    return weights


def print_dataset_stats(train_dir: str | Path, val_dir: str | Path) -> tuple[int, int]:
    total_train = 0
    total_val = 0
    print("\nDataset Check:")
    for breed in sorted(list_class_names(train_dir)):
        train_count = len(gather_images(Path(train_dir) / breed))
        val_count = len(gather_images(Path(val_dir) / breed))
        total_train += train_count
        total_val += val_count
        if train_count > 0 or val_count > 0:
            print(f"[OK] {breed}: {train_count} train, {val_count} val")
    print(f"\nDataset ready: {total_train} training, {total_val} validation images")
    return total_train, total_val

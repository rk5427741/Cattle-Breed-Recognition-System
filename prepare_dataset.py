"""
Merge train/val folders, remove duplicate images, detect corrupted files,
and rebuild a clean stratified split.

Run from project root: python prepare_dataset.py
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from ml.config import ALLOWED_TRAIN_EXTENSIONS, DEFAULT_TRAIN_CONFIG
from ml.data_utils import file_hash, is_valid_image, scan_corrupted_images

SUPPORTED_EXTS = {ext.lstrip(".") for ext in ALLOWED_TRAIN_EXTENSIONS} | {e.upper() for e in ALLOWED_TRAIN_EXTENSIONS}


def _gather_class_images(root: Path, include_not_cattle: bool) -> dict[str, list[Path]]:
    by_class: dict[str, dict[str, Path]] = {}

    for split_dir in root.iterdir():
        if not split_dir.is_dir() or split_dir.name.startswith("_"):
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name == "not_cattle" and not include_not_cattle:
                continue
            bucket = by_class.setdefault(class_name, {})
            for path in class_dir.rglob("*"):
                if path.suffix.lower().replace(".", "") not in {e.replace(".", "") for e in ALLOWED_TRAIN_EXTENSIONS}:
                    continue
                if not is_valid_image(path):
                    print(f"  [skip corrupted] {path}")
                    continue
                digest = file_hash(path)
                if digest not in bucket:
                    bucket[digest] = path

    return {name: list(paths.values()) for name, paths in by_class.items()}


def _cap_not_cattle_fraction(
    class_images: dict[str, list[Path]],
    target_fraction: float = 0.18,
    seed: int = 1337,
) -> dict[str, list[Path]]:
    """Limit not_cattle to ~15–20% of total images so negatives do not dominate training."""
    if "not_cattle" not in class_images:
        return class_images

    cattle_total = sum(len(paths) for name, paths in class_images.items() if name != "not_cattle")
    if cattle_total == 0:
        return class_images

    max_neg = max(1, int(cattle_total * target_fraction / max(1.0 - target_fraction, 1e-6)))
    neg_paths = class_images["not_cattle"]
    if len(neg_paths) <= max_neg:
        return class_images

    rng = random.Random(seed)
    sampled = rng.sample(neg_paths, max_neg)
    print(
        f"\n  [balance] not_cattle capped: {len(neg_paths)} -> {len(sampled)} "
        f"(target ~{target_fraction * 100:.0f}% of dataset)"
    )
    updated = dict(class_images)
    updated["not_cattle"] = sampled
    return updated


def _copy_split(items: list[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(items):
        ext = src.suffix.lower() or ".jpg"
        dest = dest_dir / f"{src.stem}_{idx:04d}{ext}"
        shutil.copy2(src, dest)


def rebuild_dataset(
    data_root: Path,
    val_ratio: float = 0.15,
    seed: int = 1337,
    backup: bool = True,
    include_not_cattle: bool = True,
    remove_corrupted: bool = True,
) -> None:
    random.seed(seed)

    if remove_corrupted:
        corrupted = scan_corrupted_images(data_root)
        for path in corrupted:
            print(f"Removing corrupted file: {path}")
            path.unlink(missing_ok=True)

    class_images = _gather_class_images(data_root, include_not_cattle=include_not_cattle)
    if not class_images:
        raise ValueError(f"No images found under {data_root}")

    if include_not_cattle:
        target = float(DEFAULT_TRAIN_CONFIG.get("not_cattle_target_fraction", 0.18))
        class_images = _cap_not_cattle_fraction(class_images, target_fraction=target, seed=seed)

    train_root = data_root / "train"
    val_root = data_root / "val"
    backup_root = data_root / "_backup_before_split"

    if backup and train_root.exists():
        if backup_root.exists():
            shutil.rmtree(backup_root)
        backup_root.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val"):
            src = data_root / split_name
            if src.exists():
                shutil.copytree(src, backup_root / split_name)

    staging_root = data_root / "_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)

    staged_images: dict[str, list[Path]] = {}
    for class_name, paths in class_images.items():
        staged_dir = staging_root / class_name
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_paths: list[Path] = []
        for idx, src in enumerate(paths):
            if not src.exists():
                for split_name in ("train", "val"):
                    candidate = backup_root / split_name / class_name / src.name
                    if candidate.exists():
                        src = candidate
                        break
            if not src.exists() or not is_valid_image(src):
                continue
            ext = src.suffix.lower() or ".jpg"
            dest = staged_dir / f"{src.stem}_{idx:04d}{ext}"
            shutil.copy2(src, dest)
            staged_paths.append(dest)
        staged_images[class_name] = staged_paths

    preserve = {"not_cattle"} if not include_not_cattle else set()
    for split_root in (train_root, val_root):
        if not split_root.exists():
            split_root.mkdir(parents=True, exist_ok=True)
            continue
        for class_dir in split_root.iterdir():
            if class_dir.is_dir() and class_dir.name not in preserve:
                shutil.rmtree(class_dir)

    print("\nRebuilding dataset split:")
    print(f"  val_ratio={val_ratio}, seed={seed}, include_not_cattle={include_not_cattle}")
    print("-" * 60)

    total_train = 0
    total_val = 0
    counts = {name: len(paths) for name, paths in staged_images.items()}
    if counts:
        median = sorted(counts.values())[len(counts) // 2]
        print("\nClass balance report (unique images per class):")
        for name, count in sorted(counts.items()):
            flag = " [LOW]" if count < max(20, median * 0.4) else ""
            print(f"  {name}: {count}{flag}")
        if "not_cattle" not in counts:
            print("\nTip: Add non-cattle images with: python fetch_negative_samples.py")

    for class_name in sorted(staged_images):
        paths = staged_images[class_name]
        if len(paths) < 2:
            train_items, val_items = paths, []
        else:
            train_items, val_items = train_test_split(
                paths,
                test_size=val_ratio,
                random_state=seed,
                shuffle=True,
            )

        _copy_split(train_items, train_root / class_name)
        if val_items:
            _copy_split(val_items, val_root / class_name)

        total_train += len(train_items)
        total_val += len(val_items)
        print(f"  {class_name}: {len(train_items)} train, {len(val_items)} val (from {len(paths)} unique)")

    print("-" * 60)
    print(f"Done: {total_train} train, {total_val} val images")
    if backup:
        print(f"Backup: {backup_root}")
    if staging_root.exists():
        shutil.rmtree(staging_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cattle breed dataset splits")
    parser.add_argument("--data-root", type=str, default="data_breeds")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--skip-not-cattle", action="store_true", help="Do not rebuild not_cattle folders")
    parser.add_argument("--keep-corrupted", action="store_true", help="Do not scan/remove corrupted files")
    args = parser.parse_args()

    rebuild_dataset(
        data_root=Path(args.data_root),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        backup=not args.no_backup,
        include_not_cattle=not args.skip_not_cattle,
        remove_corrupted=not args.keep_corrupted,
    )


if __name__ == "__main__":
    main()

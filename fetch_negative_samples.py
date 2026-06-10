"""
Download diverse non-cattle images for the 'not_cattle' rejection class.
Run: python fetch_negative_samples.py
"""

from __future__ import annotations

import argparse
import random
import urllib.error
import urllib.request
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def _download(url: str, dest: Path, timeout: int = 20) -> bool:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "cattle-breed-recognition/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 5000:
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def fetch_negatives(
    data_root: Path,
    total: int = 180,
    val_ratio: float = 0.15,
    seed: int = 1337,
) -> None:
    random.seed(seed)
    train_dir = data_root / "train" / "not_cattle"
    val_dir = data_root / "val" / "not_cattle"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Picsum provides varied photos (people, landscapes, objects, etc.)
    indices = list(range(1, total * 4))
    random.shuffle(indices)

    saved: list[Path] = []
    for seed_id in indices:
        if len(saved) >= total:
            break
        url = f"https://picsum.photos/seed/cattle-neg-{seed_id}/400/400.jpg"
        dest = train_dir / f"neg_{seed_id:04d}.jpg"
        if dest.exists() and dest.stat().st_size > 5000:
            saved.append(dest)
            continue
        if _download(url, dest):
            saved.append(dest)
            print(f"  downloaded {len(saved)}/{total}: {dest.name}")

    if len(saved) < max(30, total // 3):
        raise RuntimeError(
            f"Only downloaded {len(saved)} negative images. Check your internet connection and retry."
        )

    val_count = max(1, int(len(saved) * val_ratio))
    random.shuffle(saved)
    for path in saved[:val_count]:
        target = val_dir / path.name
        if path.exists():
            path.replace(target)

    print(f"\nnot_cattle negatives ready: {len(list(train_dir.glob('*.jpg')))} train, {len(list(val_dir.glob('*.jpg')))} val")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch non-cattle negative training images")
    parser.add_argument("--data-root", type=str, default="data_breeds")
    parser.add_argument("--total", type=int, default=180)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    print("Downloading non-cattle (negative) sample images...")
    fetch_negatives(
        data_root=Path(args.data_root),
        total=int(args.total),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()

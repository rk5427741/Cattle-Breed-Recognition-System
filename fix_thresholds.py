"""
Update validation thresholds in models/model_meta.json.

Run from project root: python fix_thresholds.py
"""

from __future__ import annotations

import json
from pathlib import Path

META_PATH = Path(__file__).resolve().parent / "models" / "model_meta.json"

NEW_THRESHOLDS = {
    "min_confidence": 0.20,
    "min_margin": 0.02,
    "max_not_cattle_prob": 0.75,
    "reject_confidence_pct": 20.0,
}


def main() -> None:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing {META_PATH}. Train a model first.")

    with open(META_PATH, encoding="utf-8") as handle:
        meta = json.load(handle)

    old = {k: meta.get(k) for k in NEW_THRESHOLDS}
    meta.update(NEW_THRESHOLDS)

    with open(META_PATH, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print("Updated model_meta.json")
    print("  Old:", old)
    print("  New:", NEW_THRESHOLDS)
    print("\nRestart the FastAPI server: cd backend && python main.py")


if __name__ == "__main__":
    main()

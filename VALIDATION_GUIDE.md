# Image Validation & Production Guide

## Recommended approach (final-year project)

**Hybrid gate: `not_cattle` class + calibrated confidence thresholds**

| Approach | Fit for this project |
|----------|----------------------|
| `not_cattle` class + thresholds | **Best** — works with your existing multi-class pipeline |
| Confidence-only (70%) | Good fallback, already combined with calibration |
| Separate binary classifier | Optional later; more training overhead |
| Object detection (YOLO) | Heavy; only if you add a dedicated detection dataset |

## Pipeline stages

1. **Upload** — JPG / JPEG / PNG only, max 16MB  
2. **Integrity** — corrupted/unreadable files rejected (Pillow verify)  
3. **Model gate** — softmax `not_cattle` + min confidence + min margin between top-2 cattle breeds  
4. **Breed prediction** — top-3 breeds with probabilities (only if stages 1–3 pass)

Invalid images return:

> Invalid image. Please upload an image of Indian cattle or buffalo.

## Adding non-cattle images

```bash
python fetch_negative_samples.py --total 200
python prepare_dataset.py
python train_model.py
python evaluate_model.py
```

Suggested negative categories: people, cars, dogs, landscapes, buildings, food, other animals.

Target: **at least 15–20%** of train images in `not_cattle`, diverse scenes.

## Commands

| Script | Purpose |
|--------|---------|
| `python prepare_dataset.py` | Dedupe, remove corrupted, stratified split |
| `python train_model.py` | Full training (EfficientNetB0 default) |
| `python train_model.py --base-model MobileNetV2` | Lightweight backbone |
| `python compare_models.py` | Quick backbone benchmark |
| `python evaluate_model.py` | Accuracy, P/R/F1, confusion matrix |
| `python predict.py image.jpg` | CLI test |

## Advanced improvements (documentation / future work)

- **Grad-CAM** — visualize which image regions drove the prediction  
- **Ensemble** — average EfficientNet + MobileNet probabilities  
- **Fine-tuning** — already implemented (freeze head → unfreeze last N layers)  
- **TFLite export** — `model.export('models/breed_model.tflite')` for mobile  

## Folder structure

```
ml/
  config.py          # constants, invalid message
  data_utils.py      # dataset scan, duplicates, corruption
  preprocessing.py   # resize, normalize, upload validation
  model_builder.py   # EfficientNet / ResNet / MobileNet
  validation.py      # multi-stage gate
  metrics.py         # evaluation metrics
  inference.py       # production predict
train_model.py       # training entry (unchanged CLI)
predict.py           # prediction entry (backward compatible)
evaluate_model.py
compare_models.py
backend/main.py
frontend/
```

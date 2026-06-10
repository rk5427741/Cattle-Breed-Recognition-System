# 📁 Dataset Setup Guide

## Current Status

**No images found in the dataset.** You need to add cattle breed images before training the model.

## How to Organize Your Dataset

### Step 1: Create Directory Structure

The directory structure is already created at:
```
static/dataset/raw/
├── Gir/
├── Sahiwal/
├── Red_Sindhi/
├── Tharparkar/
└── Ongole/
```

### Step 2: Add Images

1. **Collect images** of each cattle breed
2. **Place images** in the corresponding breed folder:
   - `static/dataset/raw/Gir/` - Add Gir cattle images
   - `static/dataset/raw/Sahiwal/` - Add Sahiwal cattle images
   - `static/dataset/raw/Red_Sindhi/` - Add Red Sindhi cattle images
   - `static/dataset/raw/Tharparkar/` - Add Tharparkar cattle images
   - `static/dataset/raw/Ongole/` - Add Ongole cattle images

### Step 3: Image Requirements

- **Supported formats**: JPG, JPEG, PNG
- **Minimum images per breed**: 50-100 (more is better)
- **Recommended**: 100-200 images per breed for good accuracy
- **Image quality**: Clear, well-lit images with cattle in focus
- **Variety**: Include different angles, lighting conditions, and backgrounds

### Step 4: Preprocess Dataset

After adding images, run:
```bash
python preprocess_dataset.py static/dataset/raw static/dataset
```

This will:
- Split images into train/validation/test sets (70%/15%/15%)
- Organize them into proper directories
- Prepare them for training

### Step 5: Train Model

Once preprocessing is complete:
```bash
python train_model.py
```

## Example Dataset Structure

```
static/dataset/raw/
├── Gir/
│   ├── gir_001.jpg
│   ├── gir_002.jpg
│   ├── gir_003.jpg
│   └── ... (50-200 images)
├── Sahiwal/
│   ├── sahiwal_001.jpg
│   ├── sahiwal_002.jpg
│   └── ... (50-200 images)
├── Red_Sindhi/
│   └── ... (50-200 images)
├── Tharparkar/
│   └── ... (50-200 images)
└── Ongole/
    └── ... (50-200 images)
```

## Where to Get Images

1. **Your own photos**: Take photos of cattle
2. **Public datasets**: Search for cattle breed datasets online
3. **Image search**: Use image search engines (respect copyright)
4. **Agricultural databases**: Check agricultural research databases

## Quick Check

To check how many images you have:
```bash
# Windows PowerShell
Get-ChildItem -Path "static\dataset\raw" -Recurse -Include *.jpg,*.jpeg,*.png | Measure-Object

# Or check each breed
Get-ChildItem -Path "static\dataset\raw\Gir" -Include *.jpg,*.jpeg,*.png | Measure-Object
```

## Next Steps

1. ✅ Add images to `static/dataset/raw/[BreedName]/`
2. ✅ Run preprocessing: `python preprocess_dataset.py static/dataset/raw static/dataset`
3. ✅ Train model: `python train_model.py`
4. ✅ Use the trained model for predictions

---

**Note**: Training requires a significant amount of images. Start with at least 50 images per breed for basic functionality, but 100+ per breed is recommended for good accuracy.


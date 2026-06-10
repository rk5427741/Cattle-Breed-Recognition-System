# 🚀 Quick Start Guide

Get the Cattle Breed Recognition System up and running in minutes!

## Prerequisites Check

- ✅ Python 3.8+
- ✅ Node.js 16+
- ✅ pip and npm installed

## Step 1: Backend Setup (5 minutes)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run setup script (creates directories)
python setup.py
```

## Step 2: Prepare Dataset (Optional for Testing)

If you have a dataset:

```bash
# Organize your images in: static/dataset/raw/BreedName/
# Then preprocess:
python preprocess_dataset.py static/dataset/raw static/dataset
```

**For testing without a dataset**, you can skip training and use a dummy model. The system will work but predictions will be random.

## Step 3: Train Model (30 minutes - several hours)

```bash
python train_model.py
```

**Note**: This step is optional for testing. If you skip it, the API will return an error when predicting, but you can test the frontend UI.

## Step 4: Start Backend Server

```bash
cd backend
python main.py
```

Backend runs on: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## Step 5: Start Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on: `http://localhost:5173`

## Step 6: Use the System!

1. Open `http://localhost:5173` in your browser
2. Upload a cattle image
3. Click "Predict Breed"
4. View results!

## Testing Without Training

To test the system without training a model:

1. Create a dummy model file (or skip prediction endpoint)
2. Test the frontend UI and API endpoints
3. The `/breeds` endpoint will work without a model

## Common Issues

### Port Already in Use

If port 8000 or 5173 is busy:
- Backend: Change port in `backend/main.py` (line with `uvicorn.run`)
- Frontend: Change port in `frontend/vite.config.js`

### Module Not Found

```bash
# Make sure you're in the right directory
# Install missing packages
pip install -r requirements.txt
npm install  # in frontend directory
```

### CORS Error

Make sure backend is running before frontend, and check CORS settings in `backend/main.py`.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Add your own dataset
- Customize the model architecture
- Deploy to production

---

**Happy Coding! 🐄**


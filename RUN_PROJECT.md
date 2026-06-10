# 🚀 How to Run the Cattle Breed Recognition Project

## Quick Start Guide

Follow these steps to run the complete project on your own.

---

## Prerequisites

- ✅ Python 3.8 or higher installed
- ✅ Node.js 16+ and npm installed
- ✅ Git (optional, for cloning)

---

## Step 1: Install Python Dependencies

Open a terminal/command prompt in the project directory:

```bash
# Install all Python packages
pip install -r requirements.txt
```

**What this installs:**
- FastAPI (backend framework)
- TensorFlow (deep learning)
- NumPy, Pillow, scikit-learn (data processing)
- Other required packages

**Expected time:** 5-10 minutes

---

## Step 2: Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js packages
npm install

# Go back to project root
cd ..
```

**Expected time:** 2-5 minutes

---

## Step 3: Prepare Your Dataset (If Needed)

### Option A: Using Existing Dataset (data_breeds)

If you already have the `data_breeds` folder with train/val folders:
- ✅ Skip this step - dataset is ready!

### Option B: Organize New Dataset

1. **Create folder structure:**
   ```
   data_breeds/
   ├── train/
   │   ├── breed1/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   ├── breed2/
   │   └── ...
   └── val/
       ├── breed1/
       ├── breed2/
       └── ...
   ```

2. **Add images:**
   - Place training images in `data_breeds/train/[breed_name]/`
   - Place validation images in `data_breeds/val/[breed_name]/`
   - Minimum: 50-100 images per breed (100+ recommended)

3. **Update configuration** (if needed):
   - Edit `train_model.py` to point to your dataset paths
   - Update `predict.py` with breed information

---

## Step 4: Train the Model

```bash
# Make sure you're in the project root directory
python train_model.py
```

**What happens:**
- Loads images from `data_breeds/train/` and `data_breeds/val/`
- Builds ResNet50 transfer learning model
- Trains for up to 50 epochs
- Saves best model to `models/cattle_breed_model.h5`
- Creates class names file: `models/class_names.json`

**Expected time:** 30 minutes to several hours (depends on hardware and dataset size)

**Monitor progress:**
- Model file will appear in `models/` folder when first epoch completes
- Training history plot saved to `models/training_history.png` when done

**Note:** You can stop training early (Ctrl+C) and use the model that's been saved so far.

---

## Step 5: Start the Backend Server

Open a **new terminal/command prompt** (keep training terminal open if still training):

```bash
# Navigate to backend directory
cd backend

# Start FastAPI server
python main.py
```

**Or using uvicorn directly:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**You should see:**
```
Starting Cattle Breed Recognition API...
API will be available at http://localhost:8000
API docs available at http://localhost:8000/docs
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Backend is running when you see:**
- ✅ Server started message
- ✅ No error messages
- ✅ Can access http://localhost:8000/health

**Keep this terminal open!** The server needs to keep running.

---

## Step 6: Start the Frontend

Open **another new terminal/command prompt**:

```bash
# Navigate to frontend directory
cd frontend

# Start development server
npm run dev
```

**You should see:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**Frontend is running when you see:**
- ✅ Vite server started
- ✅ Local URL displayed (usually http://localhost:5173)

**Keep this terminal open too!**

---

## Step 7: Use the Application

1. **Open your web browser**
2. **Go to:** http://localhost:5173
3. **You should see:**
   - Upload interface
   - "Predict Breed" button
   - Navigation to "About Breeds" page

4. **Test the prediction:**
   - Click "Upload a file" or drag & drop an image
   - Click "Predict Breed"
   - View the results with confidence score and breed details

---

## Complete Running Setup

You should have **3 terminals/windows open**:

1. **Terminal 1:** Training (if still running) or closed if done
2. **Terminal 2:** Backend server (FastAPI) - **MUST stay running**
3. **Terminal 3:** Frontend server (Vite) - **MUST stay running**

---

## Troubleshooting

### Backend won't start

**Error: "Module not found"**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Error: "Port 8000 already in use"**
```bash
# Option 1: Kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Option 2: Change port in backend/main.py
# Change port=8000 to port=8001 (or another port)
```

**Error: "Model not found"**
- Make sure you've trained the model (Step 4)
- Check that `models/cattle_breed_model.h5` exists

### Frontend won't start

**Error: "npm not found"**
- Install Node.js from https://nodejs.org/

**Error: "Port 5173 already in use"**
```bash
# Change port in frontend/vite.config.js
# Add: server: { port: 5174 }
```

**Error: "Cannot connect to backend"**
- Make sure backend is running (Step 5)
- Check backend URL in `frontend/src/components/Home.jsx` (should be `http://localhost:8000`)

### Training issues

**Error: "No images found"**
- Check dataset paths in `train_model.py`
- Verify images exist in `data_breeds/train/` and `data_breeds/val/`
- Ensure image formats are: JPG, JPEG, PNG

**Training is too slow**
- Reduce batch size in `train_model.py` (change `batch_size: 32` to `16` or `8`)
- Reduce number of epochs (change `epochs: 50` to `20` or `10`)
- Use GPU if available (TensorFlow will detect automatically)

**Out of memory errors**
- Reduce batch size
- Reduce image size (change `img_size: (224, 224)` to `(128, 128)`)

---

## Quick Commands Reference

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Train model
python train_model.py

# Start backend
cd backend && python main.py

# Start frontend (in new terminal)
cd frontend && npm run dev

# Test prediction (Python)
python predict.py path/to/image.jpg

# Check API health
curl http://localhost:8000/health
```

---

## API Endpoints

Once backend is running, you can access:

- **Main API:** http://localhost:8000
- **Health Check:** http://localhost:8000/health
- **API Documentation:** http://localhost:8000/docs (Interactive Swagger UI)
- **Predict:** http://localhost:8000/predict (POST request with image)
- **Get Breeds:** http://localhost:8000/breeds (GET request)

---

## Project Structure Reminder

```
cattle_breed_recognition/
├── backend/              # FastAPI backend
│   └── main.py           # Run this to start backend
├── frontend/             # React frontend
│   └── (run npm run dev) # Run this to start frontend
├── data_breeds/          # Your dataset
│   ├── train/           # Training images
│   └── val/             # Validation images
├── models/              # Trained models (created after training)
│   ├── cattle_breed_model.h5
│   └── class_names.json
├── train_model.py       # Run this to train model
├── predict.py           # Prediction module
└── requirements.txt     # Python dependencies
```

---

## Stopping the Servers

**To stop backend or frontend:**
- Press `Ctrl + C` in the respective terminal
- Or close the terminal window

**To stop training:**
- Press `Ctrl + C` in the training terminal
- The model saved so far will still be available

---

## Next Steps After Setup

1. ✅ Train model with your dataset
2. ✅ Start backend server
3. ✅ Start frontend server
4. ✅ Open browser and test predictions
5. ✅ Customize breeds and information as needed
6. ✅ Deploy to production (optional)

---

## Need Help?

- Check `README.md` for detailed documentation
- Check `QUICKSTART.md` for quick setup
- Review error messages in terminal output
- Check API docs at http://localhost:8000/docs when backend is running

---

**Happy Coding! 🐄**


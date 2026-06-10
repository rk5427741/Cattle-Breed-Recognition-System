# 🐄 Cattle Breed Recognition System

A complete deep learning-based system for identifying cattle breeds from images. Built with Transfer Learning (ResNet50/EfficientNet), FastAPI backend, and React frontend.

## 📋 Features

- **Deep Learning Model**: Uses Transfer Learning with ResNet50 or EfficientNetB0
- **Supported Breeds**: Gir, Sahiwal, Red Sindhi, Tharparkar, Ongole (easily extensible)
- **FastAPI Backend**: RESTful API with CORS support and error handling
- **React Frontend**: Modern UI with Vite, Tailwind CSS, and responsive design
- **Real-time Prediction**: Upload images and get instant breed predictions with confidence scores
- **Breed Information**: Detailed information about each breed (milk yield, region, characteristics)
- **Dataset Management**: Tools for organizing and preprocessing datasets

## 🏗️ Project Structure

```
cattle_breed_recognition/
├── backend/
│   └── main.py                 # FastAPI backend server
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Home.jsx        # Main prediction interface
│   │   │   └── About.jsx       # Breed information page
│   │   ├── App.jsx             # Main app component
│   │   └── main.jsx            # React entry point
│   ├── package.json
│   └── vite.config.js
├── models/                     # Saved model files
│   ├── cattle_breed_model.h5  # Trained model (after training)
│   └── class_names.json        # Class name mappings
├── static/
│   ├── dataset/                # Dataset directory
│   │   ├── train/             # Training images
│   │   ├── val/               # Validation images
│   │   └── test/              # Test images
│   └── uploads/               # User uploaded images
├── train_model.py             # Model training script
├── preprocess_dataset.py      # Dataset preprocessing utility
├── predict.py                 # Prediction module
├── app.py                     # Flask app (legacy, optional)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm/yarn
- TensorFlow 2.x compatible GPU (optional but recommended for training)

### 1. Clone/Download the Project

```bash
cd cattle_breed_recognition
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Prepare Dataset

1. Organize your dataset:
   ```
   static/dataset/raw/
   ├── Gir/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── Sahiwal/
   │   └── ...
   ├── Red_Sindhi/
   │   └── ...
   ├── Tharparkar/
   │   └── ...
   └── Ongole/
       └── ...
   ```

2. Preprocess dataset (splits into train/val/test):
   ```bash
   python preprocess_dataset.py static/dataset/raw static/dataset
   ```

   Or create sample structure:
   ```bash
   python preprocess_dataset.py
   ```

#### Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess images
- Build a transfer learning model (ResNet50 by default)
- Train the model with data augmentation
- Fine-tune the model
- Save the best model to `models/cattle_breed_model.h5`
- Generate training history plots

**Note**: Training may take several hours depending on your dataset size and hardware. For testing, you can use a smaller dataset or fewer epochs.

#### Start Backend Server

```bash
# Using FastAPI (recommended)
cd backend
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

### 4. Build Frontend for Production

```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/`

## 📖 Usage

### Using the Web Interface

1. Start the backend server (port 8000)
2. Start the frontend (port 5173)
3. Open `http://localhost:5173` in your browser
4. Upload a cattle image
5. Click "Predict Breed"
6. View the prediction result with confidence score and breed details

### Using the API Directly

#### Predict Breed

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

Response:
```json
{
  "success": true,
  "breed": "Gir",
  "confidence": 95.23,
  "filename": "uuid.jpg",
  "details": {
    "type": "Indigenous Indian Cow",
    "milk_yield": "2500–3000 liters/year",
    "region": "Gujarat, Maharashtra",
    "characteristics": "Large hump, drooping ears, high milk production"
  }
}
```

#### Get Top K Predictions

```bash
curl -X POST "http://localhost:8000/predict/top-k?k=3" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

#### Get All Breeds Information

```bash
curl -X GET "http://localhost:8000/breeds"
```

### Using Python Script

```python
from predict import predict_breed

breed, confidence, details = predict_breed('path/to/image.jpg')
print(f"Breed: {breed}")
print(f"Confidence: {confidence}%")
print(f"Details: {details}")
```

## 🔧 Configuration

### Model Training Configuration

Edit `train_model.py` to change:
- Base model: `'ResNet50'` or `'EfficientNetB0'`
- Image size: `(224, 224)`
- Batch size: `32`
- Epochs: `50`
- Train/val/test ratios

### Backend Configuration

Edit `backend/main.py` to change:
- CORS origins
- Upload folder path
- Max file size
- Allowed file types

## 🎯 Model Training Tips

1. **Dataset Size**: Minimum 50-100 images per breed recommended
2. **Data Quality**: Use clear, well-lit images with cattle in focus
3. **Augmentation**: Already included in training script
4. **Transfer Learning**: Uses pre-trained ImageNet weights
5. **Fine-tuning**: Automatically unfreezes top layers for better accuracy
6. **Early Stopping**: Prevents overfitting
7. **Learning Rate**: Automatically reduced on plateau

## 📊 Model Performance

After training, the model will display:
- Training/validation accuracy and loss curves
- Test set accuracy
- Classification report with precision, recall, F1-score
- Confusion matrix

## 🛠️ Troubleshooting

### Model Not Found Error

If you see "Model not found" error:
1. Make sure you've trained the model: `python train_model.py`
2. Check that `models/cattle_breed_model.h5` exists
3. Verify the model path in `predict.py`

### Backend Connection Error

1. Ensure backend is running on port 8000
2. Check CORS settings in `backend/main.py`
3. Verify API URL in frontend code

### Training Issues

1. **Out of Memory**: Reduce batch size in `train_model.py`
2. **Slow Training**: Use GPU if available, or reduce image size
3. **Low Accuracy**: Increase dataset size, add more augmentation, train for more epochs

### Frontend Build Issues

1. Clear node_modules and reinstall: `rm -rf node_modules && npm install`
2. Check Node.js version: `node --version` (should be 16+)
3. Clear Vite cache: `rm -rf node_modules/.vite`

## 🔮 Future Enhancements

- [ ] Add more cattle breeds
- [ ] Implement model retraining with new images
- [ ] Add user authentication (JWT)
- [ ] Batch prediction for multiple images
- [ ] Model versioning and A/B testing
- [ ] Docker containerization
- [ ] Cloud deployment guides (AWS, GCP, Azure)
- [ ] Mobile app integration
- [ ] Real-time video prediction

## 📝 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on the repository.

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- FastAPI for modern Python web framework
- React and Vite for frontend development
- Tailwind CSS for styling

---

**Built with ❤️ using Deep Learning**


# Snake Venom Classifier (CMPT 310)

A machine learning-based image classifier that predicts whether a snake is **venomous** or **non-venomous** from an image. This project implements a K-Nearest Neighbors (KNN) baseline classifier using computer vision features.

## 🎯 Goal

Build a baseline image classifier using:
- **Simple image features**: edges/patterns, texture histograms, color histograms
- **KNN algorithm**: K-Nearest Neighbors for classification
- **Potential upgrade path**: CNN implementation if needed

## 🛠️ Tools & Technologies

- **Python** 3.8+
- **NumPy**: Numerical computations
- **Pandas**: Data handling
- **scikit-learn**: Machine learning (KNN, preprocessing, metrics)
- **OpenCV**: Image processing and feature extraction
- **scikit-image**: Advanced image processing (LBP, GLCM)
- **Matplotlib**: Visualization
- **Streamlit**: Interactive web UI (optional)

## 📁 Project Structure

```
main_assignment/
├── data/
│   ├── raw/              # Raw dataset (not committed)
│   │   ├── venomous/     # Venomous snake images
│   │   └── non_venomous/ # Non-venomous snake images
│   └── processed/        # Processed features (not committed)
├── models/               # Trained models (not committed)
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── features.py       # Feature extraction module
│   ├── classifier.py     # KNN classifier implementation
│   └── utils.py          # Utility functions
├── train.py              # Training script
├── predict.py            # Prediction script
├── app.py                # Streamlit web application
└── requirements.txt      # Python dependencies
```

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Silver-Crystal/main_assignment.git
cd main_assignment
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download a snake dataset from Kaggle or other public sources. **Do not commit the dataset to the repository.**

Recommended datasets:
- [Kaggle: Snake Species Dataset](https://www.kaggle.com/)
- [iNaturalist: Snake observations](https://www.inaturalist.org/)

Organize your dataset in the following structure:

```
data/raw/
├── venomous/
│   ├── snake1.jpg
│   ├── snake2.jpg
│   └── ...
└── non_venomous/
    ├── snake1.jpg
    ├── snake2.jpg
    └── ...
```

### 3. Train the Model

Train the KNN classifier on your dataset:

```bash
python train.py --data_dir data/raw --model_path models/knn_classifier.joblib
```

Options:
- `--data_dir`: Directory containing the dataset (default: `data/raw`)
- `--model_path`: Path to save the trained model (default: `models/knn_classifier.joblib`)
- `--n_neighbors`: Number of neighbors for KNN (default: 5)
- `--image_size`: Size to resize images to (default: 128)
- `--validation_split`: Fraction of data for validation (default: 0.2)

### 4. Make Predictions

Predict if a snake is venomous from a single image:

```bash
python predict.py --image path/to/snake.jpg --model_path models/knn_classifier.joblib
```

### 5. Run the Web Application (Optional)

Launch the interactive Streamlit web app:

```bash
streamlit run app.py
```

This will open a browser window where you can upload snake images and get real-time predictions.

## 🔬 How It Works

### Feature Extraction

The classifier extracts three types of features from snake images:

1. **Edge Features**
   - Canny edge detection
   - Edge density and strength
   - Grid-based edge distribution

2. **Texture Features**
   - Local Binary Patterns (LBP) histograms
   - Gray-Level Co-occurrence Matrix (GLCM) properties:
     - Contrast
     - Dissimilarity
     - Homogeneity
     - Energy
     - Correlation

3. **Color Features**
   - BGR color space histograms
   - HSV color space histograms

### Classification

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Feature Normalization**: StandardScaler
- **Validation**: Cross-validation and train/validation split
- **Metrics**: Accuracy, Precision, Recall, F1-score

## 📊 Model Performance

The model performance depends on the dataset quality and size. Typical metrics:
- Training accuracy: reported during training
- Validation accuracy: reported during training
- Cross-validation accuracy: reported during training

## 🔮 Future Enhancements

Potential upgrades for better performance:
- [ ] Implement CNN-based classifier (PyTorch/TensorFlow)
- [ ] Data augmentation for better generalization
- [ ] Ensemble methods combining multiple models
- [ ] Transfer learning using pre-trained models (ResNet, EfficientNet)
- [ ] Multi-class classification (specific snake species)
- [ ] Mobile deployment (TensorFlow Lite, ONNX)

## ⚠️ Disclaimer

This is an **educational project** and should **not** be used as the sole method for identifying venomous snakes in real-world situations. Always consult a professional herpetologist or wildlife expert for accurate snake identification.

## 📝 License

This project is for educational purposes as part of CMPT 310.

## 🤝 Contributing

This is a course assignment. Please do not submit pull requests.

## 📧 Contact

For questions or issues, please contact the course instructor or teaching assistants.

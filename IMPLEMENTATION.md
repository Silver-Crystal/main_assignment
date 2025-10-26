# Snake Venom Classifier - Implementation Summary

## Project Overview

This project implements a machine learning-based image classifier that predicts whether a snake is **venomous** or **non-venomous** based on visual features extracted from images.

## Implementation Details

### 1. Feature Extraction (`src/features.py`)

The `FeatureExtractor` class extracts three types of features from snake images:

#### Edge Features
- **Canny Edge Detection**: Detects edges in the image
- **Edge Density**: Percentage of pixels that are edges
- **Edge Strength**: Average intensity of edge pixels
- **Grid-based Analysis**: Divides image into 4x4 grid and calculates edge density per region
- **Total**: 18 features (2 global + 16 grid regions)

#### Texture Features
- **Local Binary Patterns (LBP)**: Captures texture patterns using uniform LBP
  - Radius: 3 pixels
  - Points: 24 (8 * radius)
  - Histogram bins: 26
- **Gray-Level Co-occurrence Matrix (GLCM)**: Captures spatial relationships
  - Properties: Contrast, Dissimilarity, Homogeneity, Energy, Correlation
  - Angles: 0°, 45°, 90°, 135°
  - Features per property: 4 (one per angle)
  - **Total**: 46 features (26 LBP + 20 GLCM)

#### Color Features
- **BGR Histograms**: 8 bins per channel = 24 features
- **HSV Histograms**: 8 bins per channel = 24 features
- **Total**: 48 features

**Combined Feature Vector**: 112 features per image

### 2. Classification (`src/classifier.py`)

The `SnakeVenomClassifier` class implements a K-Nearest Neighbors (KNN) classifier:

#### Key Components
- **Algorithm**: K-Nearest Neighbors (default k=5)
- **Feature Scaling**: StandardScaler for normalization
- **Distance Metric**: Euclidean (configurable)
- **Validation**: Train/validation split + k-fold cross-validation

#### Capabilities
- Training with automatic validation
- Prediction with probability estimates
- Comprehensive evaluation metrics
- Model persistence (save/load)

### 3. Utility Functions (`src/utils.py`)

Helper functions for:
- Loading images from disk
- Loading datasets from directory structure
- Preprocessing images (resize, histogram equalization)
- Visualizing sample images
- Printing dataset statistics

### 4. Training Script (`train.py`)

Command-line interface for training the classifier:

```bash
python train.py --data_dir data/raw --model_path models/knn_classifier.joblib
```

**Features**:
- Configurable hyperparameters
- Automatic dataset loading and validation
- Feature extraction with progress indication
- Training with validation split
- Cross-validation scoring
- Model and configuration saving
- Comprehensive result reporting

### 5. Prediction Script (`predict.py`)

Command-line interface for making predictions:

```bash
python predict.py --image path/to/snake.jpg --model_path models/knn_classifier.joblib
```

**Features**:
- Single image prediction
- Probability estimates
- Visual warnings for venomous predictions
- User-friendly output

### 6. Web Application (`app.py`)

Interactive Streamlit web interface:

```bash
streamlit run app.py
```

**Features**:
- Image upload functionality
- Real-time prediction
- Confidence visualization with progress bars
- Color-coded results (red for venomous, green for non-venomous)
- Educational information about the classifier
- Responsive design

### 7. Testing (`tests/test_classifier.py`)

Comprehensive unit tests covering:
- Feature extraction (edge, texture, color)
- Batch processing
- Grayscale image handling
- Classifier initialization
- Training and validation
- Prediction (single and batch)
- Probability estimation
- Model evaluation
- Error handling

**Test Results**: All 13 tests passing ✓

### 8. Example Demonstrations

#### Synthetic Demo (`examples/synthetic_demo.py`)
- Creates synthetic snake images with distinctive patterns
- Demonstrates the complete workflow without real data
- Shows feature extraction, training, and prediction
- Achieves 100% accuracy on synthetic data

#### Notebook Demo (`notebooks/demo.md`)
- Interactive tutorial for understanding the system
- Code examples for each component
- Visualization suggestions
- Next steps guide

### 9. Documentation

#### README.md
- Project overview and goals
- Installation instructions
- Usage examples for all interfaces
- Feature description
- Performance metrics
- Future enhancement ideas
- Disclaimer and safety warnings

#### DATASET_GUIDE.md
- Dataset requirements and recommendations
- Sources for obtaining datasets
- Directory structure requirements
- Quality guidelines
- Kaggle CLI instructions
- Data augmentation suggestions
- Troubleshooting tips

## Technical Stack

- **Python**: 3.8+
- **NumPy**: Numerical computations
- **Pandas**: Data handling (ready for future use)
- **OpenCV**: Image processing and feature extraction
- **scikit-image**: Advanced image processing (LBP, GLCM)
- **scikit-learn**: Machine learning algorithms and utilities
- **Matplotlib**: Visualization
- **Streamlit**: Web application framework
- **joblib**: Model serialization

## Project Structure

```
main_assignment/
├── data/
│   ├── raw/                    # Raw dataset (gitignored)
│   │   ├── venomous/          # Venomous snake images
│   │   └── non_venomous/      # Non-venomous snake images
│   ├── processed/             # Processed features (gitignored)
│   └── DATASET_GUIDE.md       # Dataset setup instructions
├── models/                    # Trained models (gitignored)
├── src/                       # Source code
│   ├── features.py           # Feature extraction
│   ├── classifier.py         # KNN classifier
│   └── utils.py              # Utility functions
├── tests/                     # Unit tests
│   └── test_classifier.py
├── examples/                  # Example scripts
│   └── synthetic_demo.py
├── notebooks/                 # Jupyter notebooks
│   └── demo.md
├── train.py                   # Training script
├── predict.py                 # Prediction script
├── app.py                     # Streamlit web app
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                  # Main documentation
```

## Key Features

1. **Comprehensive Feature Extraction**: 112 features combining edges, textures, and colors
2. **Flexible Architecture**: Easy to extend with additional features or classifiers
3. **Multiple Interfaces**: CLI scripts and web UI for different use cases
4. **Model Persistence**: Save and load trained models
5. **Validation**: Multiple validation strategies (split, cross-validation)
6. **User-Friendly**: Clear warnings, progress indicators, helpful error messages
7. **Well-Tested**: Unit tests for all core functionality
8. **Well-Documented**: Comprehensive README, guides, and examples
9. **Production-Ready**: Proper .gitignore, requirements.txt, project structure

## Usage Workflow

1. **Obtain Dataset**: Download snake images from Kaggle or other sources
2. **Organize Data**: Place images in `data/raw/venomous/` and `data/raw/non_venomous/`
3. **Train Model**: Run `python train.py` to extract features and train classifier
4. **Make Predictions**: 
   - CLI: `python predict.py --image snake.jpg`
   - Web UI: `streamlit run app.py`
5. **Iterate**: Adjust hyperparameters, add more data, or upgrade to CNN if needed

## Performance Considerations

- **Feature Extraction**: ~100-200ms per image (128x128)
- **Training**: Depends on dataset size, typically < 1 minute for 100 images
- **Prediction**: Near-instant (<10ms) after feature extraction
- **Memory**: Minimal, suitable for running on standard laptops

## Future Enhancements

- [ ] CNN implementation using PyTorch or TensorFlow
- [ ] Transfer learning with pre-trained models (ResNet, EfficientNet)
- [ ] Data augmentation pipeline
- [ ] Multi-class classification (specific species)
- [ ] Ensemble methods
- [ ] Mobile deployment
- [ ] Real-time video analysis
- [ ] Confidence calibration
- [ ] Explainability features (feature importance, attention maps)

## Safety and Disclaimer

⚠️ **Important**: This is an educational project and should **NOT** be used as the sole method for identifying venomous snakes in real-world situations. Always consult a professional herpetologist for accurate identification.

## License

Educational project for CMPT 310.

## Authors

Implementation completed as part of CMPT 310 coursework.

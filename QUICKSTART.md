# Quick Start Guide

Get started with the Snake Venom Classifier in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/Silver-Crystal/main_assignment.git
cd main_assignment

# Install dependencies
pip install -r requirements.txt
```

## Quick Demo (No Dataset Required)

Run the synthetic demo to see the classifier in action:

```bash
python examples/synthetic_demo.py
```

This will:
- Generate synthetic snake images
- Extract features
- Train a KNN classifier
- Make predictions
- Achieve 100% accuracy on synthetic data

## Using Real Snake Data

### Step 1: Get a Dataset

Download a snake dataset from:
- [Kaggle](https://www.kaggle.com/datasets) - Search for "snake species"
- [iNaturalist](https://www.inaturalist.org/)

See `data/DATASET_GUIDE.md` for detailed instructions.

### Step 2: Organize Your Data

Place images in this structure:

```
data/raw/
├── venomous/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── non_venomous/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### Step 3: Train the Model

```bash
python train.py --data_dir data/raw --model_path models/knn_classifier.joblib
```

Optional parameters:
- `--n_neighbors 5` - Number of neighbors for KNN
- `--image_size 128` - Size to resize images
- `--validation_split 0.2` - Validation fraction

### Step 4: Make Predictions

**Command Line:**
```bash
python predict.py --image path/to/snake.jpg --model_path models/knn_classifier.joblib
```

**Web Interface:**
```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501

## Example Session

```bash
# 1. Install
pip install -r requirements.txt

# 2. Try the demo
python examples/synthetic_demo.py

# 3. Download and organize your dataset
# (Place images in data/raw/venomous/ and data/raw/non_venomous/)

# 4. Train
python train.py

# 5. Predict
python predict.py --image test_snake.jpg

# Or use the web UI
streamlit run app.py
```

## Understanding the Output

### Training Output

```
Training accuracy: 0.9500     # How well it fits training data
Validation accuracy: 0.8750   # How well it generalizes
Cross-validation: 0.8800 (+/- 0.0450)  # Robustness estimate
```

### Prediction Output

```
Prediction: VENOMOUS
Confidence:
  Non-venomous: 15.0%
  Venomous:     85.0%  ⚠️ WARNING
```

## Troubleshooting

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "No images found in dataset"
- Check directory structure: `data/raw/venomous/` and `data/raw/non_venomous/`
- Verify image formats: .jpg, .jpeg, .png, .bmp
- Ensure images are directly in these folders (not in subfolders)

### "Model not found"
- Train the model first: `python train.py`
- Check the model path matches in both train.py and predict.py

### Low accuracy
- Increase dataset size (aim for 200+ images per class)
- Check label accuracy
- Try different values for `--n_neighbors`
- Ensure good image quality

## Next Steps

1. **Experiment** with different hyperparameters
2. **Collect more data** for better performance
3. **Try the web UI** for easy predictions
4. **Read IMPLEMENTATION.md** to understand how it works
5. **Check README.md** for advanced features

## Testing

Run unit tests to verify everything works:

```bash
python -m unittest tests.test_classifier -v
```

## Important Safety Note

⚠️ **This is an educational tool.** Do NOT use it as the sole method for identifying venomous snakes in real situations. Always consult a professional herpetologist.

## Getting Help

- Read the full README.md
- Check data/DATASET_GUIDE.md for dataset help
- Review IMPLEMENTATION.md for technical details
- Run the examples in notebooks/demo.md

## Features Overview

**What it does:**
- Extracts 112 features from snake images (edges, textures, colors)
- Uses K-Nearest Neighbors for classification
- Provides confidence scores
- Works via command line or web interface

**What you need:**
- Python 3.8+
- A dataset of labeled snake images
- 5-10 minutes for training

**What you get:**
- Binary classification (venomous vs non-venomous)
- Probability estimates
- Saved model for future use
- Web interface for easy predictions

Happy classifying! 🐍

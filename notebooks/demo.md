# Snake Venom Classifier - Quick Start Demo

This notebook demonstrates the snake venom classifier functionality.

## Setup

```python
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join('..', 'src'))

from features import FeatureExtractor
from classifier import SnakeVenomClassifier
```

## 1. Feature Extraction Demo

Let's create a synthetic snake-like image and extract features from it:

```python
# Create a synthetic image (you can replace this with a real snake image)
test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

# Add some patterns to make it more interesting
cv2.circle(test_image, (100, 100), 50, (0, 255, 0), -1)
cv2.rectangle(test_image, (50, 50), (150, 150), (255, 0, 0), 2)

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.title('Test Image')
plt.axis('off')
plt.show()

# Extract features
extractor = FeatureExtractor(image_size=(128, 128))
features = extractor.extract_features(test_image)

print(f"Total features extracted: {len(features)}")
print(f"Feature vector shape: {features.shape}")
print(f"Sample features: {features[:10]}")
```

## 2. Individual Feature Types

```python
# Edge features
edge_features = extractor.extract_edge_features(test_image)
print(f"Edge features: {len(edge_features)} values")
print(f"Edge density: {edge_features[0]:.4f}")
print(f"Edge strength: {edge_features[1]:.4f}")

# Texture features
texture_features = extractor.extract_texture_features(test_image)
print(f"\nTexture features: {len(texture_features)} values")
print(f"Sample LBP histogram: {texture_features[:5]}")

# Color features
color_features = extractor.extract_color_features(test_image)
print(f"\nColor features: {len(color_features)} values")
print(f"Sample BGR histogram: {color_features[:8]}")
```

## 3. Classifier Demo with Synthetic Data

```python
# Create synthetic training data
np.random.seed(42)
n_samples = 100
n_features = len(features)

# Simulate venomous snake features (cluster 1)
venomous_features = np.random.randn(n_samples // 2, n_features) + 2

# Simulate non-venomous snake features (cluster 2)
non_venomous_features = np.random.randn(n_samples // 2, n_features) - 2

# Combine and create labels
X_train = np.vstack([venomous_features, non_venomous_features])
y_train = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

# Shuffle
indices = np.random.permutation(n_samples)
X_train = X_train[indices]
y_train = y_train[indices]

print(f"Training data shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")
print(f"Venomous samples: {np.sum(y_train == 1)}")
print(f"Non-venomous samples: {np.sum(y_train == 0)}")
```

## 4. Train the Classifier

```python
# Initialize and train
classifier = SnakeVenomClassifier(n_neighbors=5)
results = classifier.train(X_train, y_train, validation_split=0.2)

# Print results
print("\n" + "="*60)
print("Training Results")
print("="*60)
print(f"Training accuracy: {results['train_accuracy']:.4f}")
print(f"Validation accuracy: {results['val_accuracy']:.4f}")
print(f"Cross-validation mean: {results['cv_mean']:.4f}")
print(f"Cross-validation std: {results['cv_std']:.4f}")
print("="*60)
```

## 5. Make Predictions

```python
# Create test samples
test_venomous = np.random.randn(1, n_features) + 2
test_non_venomous = np.random.randn(1, n_features) - 2

# Predict
pred_v = classifier.predict(test_venomous)[0]
prob_v = classifier.predict_proba(test_venomous)[0]

pred_nv = classifier.predict(test_non_venomous)[0]
prob_nv = classifier.predict_proba(test_non_venomous)[0]

# Display results
print("\nTest Sample 1 (simulated venomous):")
print(f"  Prediction: {'VENOMOUS' if pred_v == 1 else 'NON-VENOMOUS'}")
print(f"  Probabilities: Non-venomous={prob_v[0]:.2%}, Venomous={prob_v[1]:.2%}")

print("\nTest Sample 2 (simulated non-venomous):")
print(f"  Prediction: {'VENOMOUS' if pred_nv == 1 else 'NON-VENOMOUS'}")
print(f"  Probabilities: Non-venomous={prob_nv[0]:.2%}, Venomous={prob_nv[1]:.2%}")
```

## 6. Save and Load Model

```python
# Save model
model_path = '../models/demo_classifier.joblib'
classifier.save_model(model_path)
print(f"\nModel saved to: {model_path}")

# Load model
new_classifier = SnakeVenomClassifier()
new_classifier.load_model(model_path)
print(f"Model loaded successfully!")

# Verify it works
test_pred = new_classifier.predict(test_venomous)[0]
print(f"Prediction with loaded model: {'VENOMOUS' if test_pred == 1 else 'NON-VENOMOUS'}")
```

## Next Steps

To use this classifier with real snake images:

1. **Download a dataset** following the instructions in `data/DATASET_GUIDE.md`
2. **Organize images** into `data/raw/venomous/` and `data/raw/non_venomous/`
3. **Train the model** using: `python train.py`
4. **Make predictions** using: `python predict.py --image path/to/snake.jpg`
5. **Use the web UI**: `streamlit run app.py`

## References

- Feature extraction uses OpenCV and scikit-image
- Classification uses scikit-learn's KNeighborsClassifier
- Features include: edge patterns, texture (LBP, GLCM), and color histograms

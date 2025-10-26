"""
Example script demonstrating the snake venom classifier with synthetic data.
This shows how the classifier works without requiring a real dataset.

Run: python examples/synthetic_demo.py
"""

import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import FeatureExtractor
from classifier import SnakeVenomClassifier


def create_synthetic_snake_image(is_venomous=True, size=(200, 200)):
    """
    Create a synthetic snake-like image with different patterns.
    
    Args:
        is_venomous: If True, create a pattern typical of venomous snakes
        size: Image size
        
    Returns:
        Synthetic image
    """
    image = np.random.randint(50, 150, (*size, 3), dtype=np.uint8)
    
    if is_venomous:
        # Venomous snakes often have distinct patterns
        # Add diamond/triangular patterns
        for i in range(0, size[0], 40):
            pts = np.array([[i, 50], [i+20, 80], [i+40, 50], [i+20, 20]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(image, [pts], (0, 0, 255))
        
        # Add warning colors (reds, yellows)
        cv2.rectangle(image, (10, 10), (30, 30), (0, 255, 255), -1)
    else:
        # Non-venomous snakes often have stripes or uniform colors
        # Add horizontal stripes
        for i in range(0, size[0], 20):
            cv2.rectangle(image, (0, i), (size[1], i+10), (100, 150, 100), -1)
        
        # Add calmer colors
        cv2.circle(image, (size[0]//2, size[1]//2), 30, (50, 100, 50), -1)
    
    return image


def main():
    print("\n" + "="*70)
    print("Snake Venom Classifier - Synthetic Data Demo")
    print("="*70)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    np.random.seed(42)
    
    n_samples_per_class = 30
    image_size = (150, 150)
    
    images = []
    labels = []
    
    # Create venomous snake images
    print(f"   Generating {n_samples_per_class} synthetic venomous snake images...")
    for i in range(n_samples_per_class):
        img = create_synthetic_snake_image(is_venomous=True, size=image_size)
        images.append(img)
        labels.append(1)
    
    # Create non-venomous snake images
    print(f"   Generating {n_samples_per_class} synthetic non-venomous snake images...")
    for i in range(n_samples_per_class):
        img = create_synthetic_snake_image(is_venomous=False, size=image_size)
        images.append(img)
        labels.append(0)
    
    labels = np.array(labels)
    print(f"   Total images: {len(images)}")
    print(f"   Venomous: {np.sum(labels == 1)}, Non-venomous: {np.sum(labels == 0)}")
    
    # Extract features
    print("\n2. Extracting features from images...")
    extractor = FeatureExtractor(image_size=(128, 128))
    features = extractor.extract_features_batch(images)
    print(f"   Feature extraction complete")
    print(f"   Feature vector size: {features.shape[1]}")
    print(f"   Total samples: {features.shape[0]}")
    
    # Train classifier
    print("\n3. Training KNN classifier...")
    classifier = SnakeVenomClassifier(n_neighbors=5)
    results = classifier.train(features, labels, validation_split=0.2)
    
    print("\n" + "-"*70)
    print("Training Results:")
    print("-"*70)
    print(f"Training samples: {results['n_samples']}")
    print(f"Validation samples: {results['n_val_samples']}")
    print(f"Feature dimensions: {results['n_features']}")
    print(f"Training accuracy: {results['train_accuracy']:.2%}")
    print(f"Validation accuracy: {results['val_accuracy']:.2%}")
    print(f"Cross-validation: {results['cv_mean']:.2%} (+/- {results['cv_std']:.2%})")
    print("-"*70)
    
    # Test predictions
    print("\n4. Testing predictions on new synthetic images...")
    
    # Create test images
    test_venomous = create_synthetic_snake_image(is_venomous=True, size=image_size)
    test_non_venomous = create_synthetic_snake_image(is_venomous=False, size=image_size)
    
    # Extract features and predict
    features_v = extractor.extract_features(test_venomous)
    pred_v = classifier.predict(features_v)[0]
    prob_v = classifier.predict_proba(features_v)[0]
    
    features_nv = extractor.extract_features(test_non_venomous)
    pred_nv = classifier.predict(features_nv)[0]
    prob_nv = classifier.predict_proba(features_nv)[0]
    
    # Display results
    print("\n" + "-"*70)
    print("Test Image 1 (synthetic venomous pattern):")
    print("-"*70)
    print(f"Prediction: {'VENOMOUS ⚠️' if pred_v == 1 else 'NON-VENOMOUS ✓'}")
    print(f"Confidence: Non-venomous={prob_v[0]:.1%}, Venomous={prob_v[1]:.1%}")
    print(f"Correct: {'YES ✓' if pred_v == 1 else 'NO ✗'}")
    
    print("\n" + "-"*70)
    print("Test Image 2 (synthetic non-venomous pattern):")
    print("-"*70)
    print(f"Prediction: {'VENOMOUS ⚠️' if pred_nv == 1 else 'NON-VENOMOUS ✓'}")
    print(f"Confidence: Non-venomous={prob_nv[0]:.1%}, Venomous={prob_nv[1]:.1%}")
    print(f"Correct: {'YES ✓' if pred_nv == 0 else 'NO ✗'}")
    print("-"*70)
    
    # Save model
    model_path = os.path.join('models', 'synthetic_demo_classifier.joblib')
    os.makedirs('models', exist_ok=True)
    classifier.save_model(model_path)
    print(f"\n5. Model saved to: {model_path}")
    
    # Summary
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("""
This demo used synthetic images to demonstrate the classifier.

To use with REAL snake images:
1. Download a snake dataset (see data/DATASET_GUIDE.md)
2. Organize images into data/raw/venomous/ and data/raw/non_venomous/
3. Run: python train.py --data_dir data/raw
4. Run: python predict.py --image path/to/snake.jpg
5. Or use the web UI: streamlit run app.py

The classifier extracts:
- Edge features (Canny detection, grid-based density)
- Texture features (Local Binary Patterns, GLCM properties)
- Color features (BGR and HSV histograms)

And uses K-Nearest Neighbors for classification.
    """)


if __name__ == '__main__':
    main()

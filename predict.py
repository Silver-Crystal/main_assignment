"""
Prediction script for the snake venom classifier.

Usage:
    python predict.py --image path/to/snake.jpg --model_path models/knn_classifier.joblib
"""

import argparse
import os
import sys
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from features import FeatureExtractor
from classifier import SnakeVenomClassifier


def main():
    parser = argparse.ArgumentParser(description='Predict if a snake is venomous')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the snake image')
    parser.add_argument('--model_path', type=str, default='models/knn_classifier.joblib',
                        help='Path to the trained model')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Size to resize images to (should match training)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        print("Please train the model first using train.py")
        return
    
    # Load image
    print(f"\nLoading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print("Error: Could not load image")
        return
    
    # Load model
    print(f"Loading model: {args.model_path}")
    classifier = SnakeVenomClassifier()
    classifier.load_model(args.model_path)
    
    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor(image_size=(args.image_size, args.image_size))
    features = feature_extractor.extract_features(image)
    
    # Make prediction
    print("Making prediction...")
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    
    # Display results
    print("\n" + "="*60)
    print("Prediction Results")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"\nPrediction: {'VENOMOUS' if prediction == 1 else 'NON-VENOMOUS'}")
    print(f"\nConfidence:")
    print(f"  Non-venomous: {probabilities[0]:.2%}")
    print(f"  Venomous:     {probabilities[1]:.2%}")
    print("="*60)
    
    # Display warning if venomous
    if prediction == 1:
        print("\n⚠️  WARNING: This snake is predicted to be VENOMOUS!")
        print("    Exercise extreme caution and keep a safe distance.")
    else:
        print("\n✓  This snake is predicted to be NON-VENOMOUS.")
        print("   However, always exercise caution around wild animals.")
    
    print()


if __name__ == '__main__':
    main()

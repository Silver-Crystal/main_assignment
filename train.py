"""
Training script for the snake venom classifier.

Usage:
    python train.py --data_dir data/raw --model_path models/knn_classifier.joblib
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from features import FeatureExtractor
from classifier import SnakeVenomClassifier
from utils import load_dataset_from_directory, print_dataset_info


def main():
    parser = argparse.ArgumentParser(description='Train snake venom classifier')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory containing the dataset')
    parser.add_argument('--model_path', type=str, default='models/knn_classifier.joblib',
                        help='Path to save the trained model')
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Size to resize images to (square)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Snake Venom Classifier - Training")
    print("="*60)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"\nError: Data directory not found: {args.data_dir}")
        print("\nPlease download a snake dataset and organize it as:")
        print("  data/raw/")
        print("    venomous/")
        print("      snake1.jpg")
        print("      snake2.jpg")
        print("      ...")
        print("    non_venomous/")
        print("      snake1.jpg")
        print("      snake2.jpg")
        print("      ...")
        print("\nRecommended datasets:")
        print("  - Kaggle: Snake Species Dataset")
        print("  - iNaturalist: Snake observations")
        return
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    images, labels, filenames = load_dataset_from_directory(args.data_dir)
    
    if len(images) == 0:
        print("\nError: No images found in the dataset.")
        print("Please check the directory structure and image formats.")
        return
    
    # Print dataset info
    print_dataset_info(images, labels)
    
    # Extract features
    print("Extracting features from images...")
    feature_extractor = FeatureExtractor(image_size=(args.image_size, args.image_size))
    features = feature_extractor.extract_features_batch(images)
    print(f"Feature extraction complete. Feature vector size: {features.shape[1]}")
    
    # Train classifier
    print(f"\nTraining KNN classifier (k={args.n_neighbors})...")
    classifier = SnakeVenomClassifier(n_neighbors=args.n_neighbors)
    results = classifier.train(features, labels, validation_split=args.validation_split)
    
    # Print results
    print("\n" + "="*60)
    print("Training Results")
    print("="*60)
    print(f"Training samples: {results['n_samples']}")
    print(f"Feature dimensions: {results['n_features']}")
    print(f"Training accuracy: {results['train_accuracy']:.4f}")
    
    if 'val_accuracy' in results:
        print(f"Validation samples: {results['n_val_samples']}")
        print(f"Validation accuracy: {results['val_accuracy']:.4f}")
    
    print(f"Cross-validation accuracy: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
    print("="*60)
    
    # Save model
    print(f"\nSaving model to {args.model_path}...")
    classifier.save_model(args.model_path)
    
    # Save feature extractor config
    config_path = args.model_path.replace('.joblib', '_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"image_size={args.image_size}\n")
        f.write(f"n_neighbors={args.n_neighbors}\n")
        f.write(f"n_features={results['n_features']}\n")
    
    print(f"Feature extractor config saved to {config_path}")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

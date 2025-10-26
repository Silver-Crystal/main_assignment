"""
Unit tests for the snake venom classifier.
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import FeatureExtractor
from classifier import SnakeVenomClassifier


class TestFeatureExtractor(unittest.TestCase):
    """Test feature extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor(image_size=(128, 128))
        # Create a simple test image
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
    def test_extract_edge_features(self):
        """Test edge feature extraction."""
        features = self.extractor.extract_edge_features(self.test_image)
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        
    def test_extract_texture_features(self):
        """Test texture feature extraction."""
        features = self.extractor.extract_texture_features(self.test_image)
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        
    def test_extract_color_features(self):
        """Test color feature extraction."""
        features = self.extractor.extract_color_features(self.test_image)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 48)  # 3 channels * 8 bins * 2 color spaces
        
    def test_extract_features(self):
        """Test combined feature extraction."""
        features = self.extractor.extract_features(self.test_image)
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
        
    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        images = [self.test_image, self.test_image, self.test_image]
        features = self.extractor.extract_features_batch(images)
        self.assertEqual(features.shape[0], 3)
        self.assertGreater(features.shape[1], 0)
        
    def test_grayscale_image(self):
        """Test feature extraction with grayscale image."""
        gray_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        features = self.extractor.extract_features(gray_image)
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)


class TestSnakeVenomClassifier(unittest.TestCase):
    """Test classifier functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = SnakeVenomClassifier(n_neighbors=3)
        
        # Create dummy training data
        np.random.seed(42)
        self.X_train = np.random.randn(50, 20)  # 50 samples, 20 features
        self.y_train = np.random.randint(0, 2, 50)  # Binary labels
        
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertEqual(self.classifier.n_neighbors, 3)
        self.assertFalse(self.classifier.is_trained)
        
    def test_train(self):
        """Test classifier training."""
        results = self.classifier.train(self.X_train, self.y_train, validation_split=0.2)
        
        self.assertTrue(self.classifier.is_trained)
        self.assertIn('train_accuracy', results)
        self.assertIn('val_accuracy', results)
        self.assertIn('cv_mean', results)
        self.assertGreater(results['train_accuracy'], 0)
        
    def test_predict_without_training(self):
        """Test that predict raises error without training."""
        X_test = np.random.randn(5, 20)
        with self.assertRaises(ValueError):
            self.classifier.predict(X_test)
            
    def test_predict(self):
        """Test prediction after training."""
        self.classifier.train(self.X_train, self.y_train, validation_split=0)
        
        X_test = np.random.randn(5, 20)
        predictions = self.classifier.predict(X_test)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(p in [0, 1] for p in predictions))
        
    def test_predict_proba(self):
        """Test probability prediction."""
        self.classifier.train(self.X_train, self.y_train, validation_split=0)
        
        X_test = np.random.randn(5, 20)
        probabilities = self.classifier.predict_proba(X_test)
        
        self.assertEqual(probabilities.shape, (5, 2))
        # Check that probabilities sum to 1
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        
    def test_predict_single_sample(self):
        """Test prediction with single sample."""
        self.classifier.train(self.X_train, self.y_train, validation_split=0)
        
        X_test = np.random.randn(20)  # Single sample
        predictions = self.classifier.predict(X_test)
        
        self.assertEqual(len(predictions), 1)
        self.assertIn(predictions[0], [0, 1])
        
    def test_evaluate(self):
        """Test model evaluation."""
        self.classifier.train(self.X_train, self.y_train, validation_split=0)
        
        X_test = np.random.randn(20, 20)
        y_test = np.random.randint(0, 2, 20)
        
        results = self.classifier.evaluate(X_test, y_test)
        
        self.assertIn('accuracy', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1_score', results)
        self.assertIn('confusion_matrix', results)


if __name__ == '__main__':
    unittest.main()

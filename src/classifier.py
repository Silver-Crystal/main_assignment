"""
Snake venom classifier using K-Nearest Neighbors (KNN).
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os


class SnakeVenomClassifier:
    """KNN-based classifier for venomous vs non-venomous snakes."""
    
    def __init__(self, n_neighbors=5, metric='euclidean'):
        """
        Initialize the classifier.
        
        Args:
            n_neighbors: Number of neighbors to use for KNN
            metric: Distance metric for KNN ('euclidean', 'manhattan', 'minkowski')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.scaler = StandardScaler()
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        self.is_trained = False
        
    def train(self, X_train, y_train, validation_split=0.2):
        """
        Train the classifier on the training data.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - 0 for non-venomous, 1 for venomous
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training results
        """
        # Split data for validation
        if validation_split > 0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, random_state=42, stratify=y_train
            )
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = None, None
        
        # Normalize features
        X_tr_scaled = self.scaler.fit_transform(X_tr)
        
        # Train the classifier
        self.classifier.fit(X_tr_scaled, y_tr)
        self.is_trained = True
        
        # Evaluate on training set
        train_score = self.classifier.score(X_tr_scaled, y_tr)
        
        # Evaluate on validation set if available
        results = {
            'train_accuracy': train_score,
            'n_samples': len(X_tr),
            'n_features': X_tr.shape[1]
        }
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.classifier.score(X_val_scaled, y_val)
            results['val_accuracy'] = val_score
            results['n_val_samples'] = len(X_val)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.classifier, X_tr_scaled, y_tr, cv=min(5, len(X_tr)), scoring='accuracy'
        )
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        return results
    
    def predict(self, X):
        """
        Predict labels for given features.
        
        Args:
            X: Features (n_samples, n_features) or (n_features,)
            
        Returns:
            Predicted labels (0 for non-venomous, 1 for venomous)
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions")
        
        # Handle single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.classifier.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for given features.
        
        Args:
            X: Features (n_samples, n_features) or (n_features,)
            
        Returns:
            Predicted probabilities (n_samples, 2) - [non-venomous, venomous]
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions")
        
        # Handle single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        probabilities = self.classifier.predict_proba(X_scaled)
        
        return probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test: Test features (n_samples, n_features)
            y_test: Test labels (n_samples,)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before evaluation")
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate accuracy
        accuracy = self.classifier.score(X_test_scaled, y_test)
        
        # Get predictions
        predictions = self.classifier.predict(X_test_scaled)
        
        # Calculate per-class metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        cm = confusion_matrix(y_test, predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'n_samples': len(X_test)
        }
        
        return results
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before saving")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model components
        model_data = {
            'scaler': self.scaler,
            'classifier': self.classifier,
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']
        self.n_neighbors = model_data['n_neighbors']
        self.metric = model_data['metric']
        self.is_trained = model_data['is_trained']

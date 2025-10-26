"""
Feature extraction module for snake image classification.
Extracts edge features, texture histograms, and color histograms from images.
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


class FeatureExtractor:
    """Extract multiple types of features from snake images."""
    
    def __init__(self, image_size=(128, 128)):
        """
        Initialize the feature extractor.
        
        Args:
            image_size: Tuple of (height, width) to resize images to
        """
        self.image_size = image_size
        
    def extract_edge_features(self, image):
        """
        Extract edge features using Canny edge detection.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Edge feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge statistics
        edge_density = np.mean(edges > 0)
        edge_strength = np.mean(edges)
        
        # Divide image into grid and calculate edge density per region
        h, w = edges.shape
        grid_size = 4
        grid_h = h // grid_size
        grid_w = w // grid_size
        
        grid_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                region = edges[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                grid_features.append(np.mean(region > 0))
        
        # Combine all edge features
        edge_features = np.array([edge_density, edge_strength] + grid_features)
        
        return edge_features
    
    def extract_texture_features(self, image):
        """
        Extract texture features using Local Binary Patterns (LBP) and GLCM.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Texture feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # LBP features
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate LBP histogram
        n_bins = n_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # GLCM features (Gray-Level Co-occurrence Matrix)
        # Normalize grayscale to reduce levels for GLCM computation
        gray_scaled = (gray / 16).astype(np.uint8)
        
        # Compute GLCM for different angles
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_scaled, distances, angles, levels=16, symmetric=True, normed=True)
        
        # Extract GLCM properties
        contrast = graycoprops(glcm, 'contrast').ravel()
        dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
        homogeneity = graycoprops(glcm, 'homogeneity').ravel()
        energy = graycoprops(glcm, 'energy').ravel()
        correlation = graycoprops(glcm, 'correlation').ravel()
        
        # Combine all texture features
        glcm_features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
        texture_features = np.concatenate([lbp_hist, glcm_features])
        
        return texture_features
    
    def extract_color_features(self, image):
        """
        Extract color histogram features in multiple color spaces.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Color feature vector
        """
        if len(image.shape) != 3:
            # Grayscale image - create dummy color features
            return np.zeros(48)
        
        # BGR color histograms
        bgr_features = []
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [8], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            bgr_features.extend(hist)
        
        # HSV color histograms
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_features = []
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [8], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            hsv_features.extend(hist)
        
        # Combine color features
        color_features = np.array(bgr_features + hsv_features)
        
        return color_features
    
    def extract_features(self, image):
        """
        Extract all features from an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Combined feature vector
        """
        # Resize image to standard size
        resized = cv2.resize(image, self.image_size)
        
        # Extract different types of features
        edge_features = self.extract_edge_features(resized)
        texture_features = self.extract_texture_features(resized)
        color_features = self.extract_color_features(resized)
        
        # Concatenate all features
        all_features = np.concatenate([edge_features, texture_features, color_features])
        
        return all_features
    
    def extract_features_batch(self, images):
        """
        Extract features from a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            2D array of features (n_samples, n_features)
        """
        features_list = []
        for image in images:
            features = self.extract_features(image)
            features_list.append(features)
        
        return np.array(features_list)

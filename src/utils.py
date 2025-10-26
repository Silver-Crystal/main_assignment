"""
Utility functions for loading and preprocessing snake images.
"""

import os
import cv2
import numpy as np
from pathlib import Path


def load_image(image_path):
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image in BGR format, or None if loading fails
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_dataset_from_directory(data_dir, venomous_subdir='venomous', non_venomous_subdir='non_venomous'):
    """
    Load images from a directory structure.
    Expected structure:
        data_dir/
            venomous/
                img1.jpg
                img2.jpg
                ...
            non_venomous/
                img1.jpg
                img2.jpg
                ...
    
    Args:
        data_dir: Root directory containing the dataset
        venomous_subdir: Name of subdirectory containing venomous snake images
        non_venomous_subdir: Name of subdirectory containing non-venomous snake images
        
    Returns:
        Tuple of (images, labels, filenames)
    """
    data_path = Path(data_dir)
    images = []
    labels = []
    filenames = []
    
    # Load venomous images (label = 1)
    venomous_path = data_path / venomous_subdir
    if venomous_path.exists():
        for img_file in venomous_path.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                img = load_image(img_file)
                if img is not None:
                    images.append(img)
                    labels.append(1)
                    filenames.append(str(img_file))
    else:
        print(f"Warning: Venomous directory not found: {venomous_path}")
    
    # Load non-venomous images (label = 0)
    non_venomous_path = data_path / non_venomous_subdir
    if non_venomous_path.exists():
        for img_file in non_venomous_path.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                img = load_image(img_file)
                if img is not None:
                    images.append(img)
                    labels.append(0)
                    filenames.append(str(img_file))
    else:
        print(f"Warning: Non-venomous directory not found: {non_venomous_path}")
    
    return images, np.array(labels), filenames


def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess an image for feature extraction.
    
    Args:
        image: Input image
        target_size: Target size to resize to
        
    Returns:
        Preprocessed image
    """
    # Resize
    processed = cv2.resize(image, target_size)
    
    # Optional: Apply histogram equalization for better contrast
    # This can help normalize lighting conditions
    if len(processed.shape) == 3:
        # Convert to YUV and equalize Y channel
        yuv = cv2.cvtColor(processed, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        processed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        processed = cv2.equalizeHist(processed)
    
    return processed


def visualize_samples(images, labels, n_samples=5):
    """
    Visualize sample images from the dataset.
    
    Args:
        images: List of images
        labels: Array of labels
        n_samples: Number of samples to visualize per class
    """
    import matplotlib.pyplot as plt
    
    # Get indices for each class
    venomous_indices = np.where(labels == 1)[0]
    non_venomous_indices = np.where(labels == 0)[0]
    
    # Sample random indices
    n_venomous = min(n_samples, len(venomous_indices))
    n_non_venomous = min(n_samples, len(non_venomous_indices))
    
    venomous_samples = np.random.choice(venomous_indices, n_venomous, replace=False)
    non_venomous_samples = np.random.choice(non_venomous_indices, n_non_venomous, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
    
    # Plot venomous samples
    for i, idx in enumerate(venomous_samples):
        if n_samples == 1:
            ax = axes[0]
        else:
            ax = axes[0, i]
        img_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f'Venomous {i+1}')
        ax.axis('off')
    
    # Plot non-venomous samples
    for i, idx in enumerate(non_venomous_samples):
        if n_samples == 1:
            ax = axes[1]
        else:
            ax = axes[1, i]
        img_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f'Non-venomous {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def print_dataset_info(images, labels):
    """
    Print information about the dataset.
    
    Args:
        images: List of images
        labels: Array of labels
    """
    n_venomous = np.sum(labels == 1)
    n_non_venomous = np.sum(labels == 0)
    
    print("\n" + "="*50)
    print("Dataset Information")
    print("="*50)
    print(f"Total images: {len(images)}")
    print(f"Venomous snakes: {n_venomous} ({n_venomous/len(images)*100:.1f}%)")
    print(f"Non-venomous snakes: {n_non_venomous} ({n_non_venomous/len(images)*100:.1f}%)")
    
    if len(images) > 0:
        sample_shape = images[0].shape
        print(f"Sample image shape: {sample_shape}")
    
    print("="*50 + "\n")

"""
Streamlit web application for snake venom classification.

Usage:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from features import FeatureExtractor
from classifier import SnakeVenomClassifier


# Page configuration
st.set_page_config(
    page_title="Snake Venom Classifier",
    page_icon="🐍",
    layout="centered"
)

# Title and description
st.title("🐍 Snake Venom Classifier")
st.markdown("""
This application uses machine learning to predict whether a snake is **venomous** or **non-venomous** 
based on an image. Upload a snake image to get started!

**Disclaimer:** This is an educational tool. Always consult a professional herpetologist for 
accurate snake identification in real-world situations.
""")

# Sidebar for settings
st.sidebar.title("Settings")
model_path = st.sidebar.text_input("Model Path", "models/knn_classifier.joblib")
image_size = st.sidebar.slider("Image Size", 64, 256, 128, 32)
n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 15, 5)

# Check if model exists
if not os.path.exists(model_path):
    st.error(f"⚠️ Model not found at: {model_path}")
    st.info("""
    Please train the model first using:
    ```
    python train.py --data_dir data/raw --model_path models/knn_classifier.joblib
    ```
    """)
    st.stop()

# Load model (with caching)
@st.cache_resource
def load_model(model_path):
    classifier = SnakeVenomClassifier()
    classifier.load_model(model_path)
    return classifier

@st.cache_resource
def load_feature_extractor(image_size):
    return FeatureExtractor(image_size=(image_size, image_size))

try:
    classifier = load_model(model_path)
    feature_extractor = load_feature_extractor(image_size)
    st.sidebar.success("✓ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload a snake image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Make prediction
    with st.spinner("Analyzing image..."):
        # Extract features
        features = feature_extractor.extract_features(image_cv)
        
        # Predict
        prediction = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
    
    # Display results
    with col2:
        st.subheader("Prediction Results")
        
        if prediction == 1:
            st.error("### 🔴 VENOMOUS")
            st.warning("""
            **⚠️ WARNING:** This snake is predicted to be venomous!
            
            - Exercise extreme caution
            - Keep a safe distance
            - Do not attempt to handle
            - Contact a professional if needed
            """)
        else:
            st.success("### 🟢 NON-VENOMOUS")
            st.info("""
            **✓** This snake is predicted to be non-venomous.
            
            However, always exercise caution around wild animals.
            """)
        
        # Show confidence
        st.subheader("Confidence Levels")
        st.progress(probabilities[0], text=f"Non-venomous: {probabilities[0]:.1%}")
        st.progress(probabilities[1], text=f"Venomous: {probabilities[1]:.1%}")
    
    # Additional information
    st.divider()
    st.subheader("ℹ️ How it works")
    st.markdown("""
    This classifier uses a K-Nearest Neighbors (KNN) algorithm with the following features:
    - **Edge features**: Detected using Canny edge detection
    - **Texture features**: Local Binary Patterns (LBP) and GLCM
    - **Color features**: Histograms in BGR and HSV color spaces
    
    The model analyzes these features to make its prediction.
    """)
    
    # Show feature extraction details (expandable)
    with st.expander("🔬 Feature Details"):
        st.write(f"Total features extracted: {len(features)}")
        st.write(f"Image size used: {image_size}x{image_size}")
        st.write(f"K (neighbors): {n_neighbors}")

# Instructions when no image is uploaded
else:
    st.info("👆 Please upload a snake image to get started")
    
    # Example usage
    st.subheader("📝 Example Usage")
    st.markdown("""
    1. **Upload an image** of a snake using the file uploader above
    2. The model will **analyze** the image using computer vision techniques
    3. You'll receive a **prediction** indicating if the snake is venomous or not
    4. **Confidence levels** show how certain the model is about its prediction
    """)
    
    st.subheader("⚙️ Model Training")
    st.markdown("""
    To train or retrain the model with your own dataset:
    
    ```bash
    python train.py --data_dir data/raw --model_path models/knn_classifier.joblib
    ```
    
    Organize your data as:
    ```
    data/raw/
      ├── venomous/
      │   ├── snake1.jpg
      │   ├── snake2.jpg
      │   └── ...
      └── non_venomous/
          ├── snake1.jpg
          ├── snake2.jpg
          └── ...
    ```
    """)

# Footer
st.divider()
st.caption("🐍 Snake Venom Classifier | Built with Streamlit and scikit-learn")

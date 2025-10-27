import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import local_binary_pattern
from tqdm import tqdm


DATA_DIR = "Snake_Classification_data/train"
CATEGORIES = ["Non Venomous", "Venomous"]

# FEATURE EXTRACTION
def extract_features(img_path):
    """Extract color histogram + texture (LBP) features from an image."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        # Color histogram 
        hist_r = cv2.calcHist([img.astype('float32')], [0], None, [8], [0, 1])
        hist_g = cv2.calcHist([img.astype('float32')], [1], None, [8], [0, 1])
        hist_b = cv2.calcHist([img.astype('float32')], [2], None, [8], [0, 1])
        color_hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
        color_hist /= (np.sum(color_hist) + 1e-6)

        # Texture (LBP)
        gray = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-6)

        # Combine features
        features = np.hstack([color_hist, hist_lbp])
        return features

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


# LOAD DATA
print("🔍 Loading dataset...")
data = []
labels = []

for label, category in enumerate(CATEGORIES):
    folder = os.path.join(DATA_DIR, category)
    if not os.path.exists(folder):
        print(f"⚠️ Warning: Folder not found - {folder}")
        continue

    for file in tqdm(os.listdir(folder), desc=f"Processing {category}"):
        path = os.path.join(folder, file)
        features = extract_features(path)
        if features is not None:
            data.append(features)
            labels.append(label)


X = np.array(data)
y = np.array(labels)

print(f"Dataset loaded: {X.shape}, Labels: {y.shape}")

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RANDOM FOREST  AS CLASSIFIER
print("\n Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# EVALUATION
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# VISUALIZING SAMPLES
plt.figure(figsize=(5, 4))
plt.bar(CATEGORIES, [np.mean(y_test == 0), np.mean(y_test == 1)], color=["#4CAF50", "#E53935"])
plt.title("Class Distribution in Test Set")
plt.ylabel("Proportion")
plt.tight_layout()
plt.show()


import json
from pathlib import Path
import numpy as np
import cv2
from joblib import dump
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings

DATA_ROOT = Path("Snake_Images")
TRAIN_DIRS = [
    DATA_ROOT / "train" / "Non Venomous",
    DATA_ROOT / "train" / "Venomous",
]
TEST_DIRS = [
    DATA_ROOT / "test" / "Non Venomous",
    DATA_ROOT / "test" / "Venomous",
]

CLASS_NAMES = ["Non Venomous", "Venomous"] 
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".PNG", ".JPEG", ".WEBP"}

def hsv_hist(img_rgb, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    return hist.astype(np.float32)

def hog_feat(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    patch = cv2.resize(gray, win_size, interpolation=cv2.INTER_AREA)
    return hog.compute(patch).flatten().astype(np.float32)

def edge_density(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.array([edges.mean() / 255.0], dtype=np.float32)

def image_to_feature(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.concatenate([hsv_hist(img_rgb), hog_feat(img_rgb), edge_density(img_rgb)])

def list_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix in EXTS]

def load_split(dirs):
    X, y = [], []
    for label, folder in enumerate(dirs):
        paths = list_images(folder)
        for p in tqdm(paths, desc=f"Loading {folder}"):
            img = cv2.imread(str(p))
            if img is None:
                continue
            feat = image_to_feature(img)
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)

# --- balancing ---
def downsample_to_match(X, y, seed=42):
    """Downsample the larger class to match the smaller class."""
    X = np.asarray(X)
    y = np.asarray(y)
    cls0_idx = np.where(y == 0)[0]
    cls1_idx = np.where(y == 1)[0]
    n = min(len(cls0_idx), len(cls1_idx))
    rng = np.random.default_rng(seed)
    keep0 = rng.choice(cls0_idx, n, replace=False)
    keep1 = rng.choice(cls1_idx, n, replace=False)
    keep = np.concatenate([keep0, keep1])
    rng.shuffle(keep)
    return X[keep], y[keep]

def plot_confusion(cm, labels, out_path="reports/confusion_matrix.png"):
    Path("reports").mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (KNN)",
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    X_train, y_train = load_split(TRAIN_DIRS)
    X_test, y_test = load_split(TEST_DIRS)

    X_train, y_train = downsample_to_match(X_train, y_train, seed=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski"],  
        "p": [1, 2],          
    }
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, n_jobs=1, verbose=1)
    grid.fit(X_train_s, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Test Accuracy: {acc:.4f}\n")
    print(report_text)

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    dump(best, "models/knn.joblib")
    dump(scaler, "models/scaler.joblib")
    plot_confusion(cm, CLASS_NAMES, "reports/confusion_matrix.png")
    with open("reports/metrics.json", "w") as f:
        json.dump({"best_params": grid.best_params_, "accuracy": acc, "report": report_dict}, f, indent=2)

if __name__ == "__main__":
    main()

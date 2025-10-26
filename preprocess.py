import cv2
from pathlib import Path
from tqdm import tqdm

dataset_dir = Path("Snake_Images")
output_size = (128, 128)

CLASS_DIRS = {
    "train": ["Venomous", "Non Venomous"],
    "test":  ["Venomous", "Non Venomous"],
}

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"} 

def iter_images(folder: Path):
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p

def resize_images_in_folder(folder_path: Path):
    paths = list(iter_images(folder_path))
    for img_path in tqdm(paths, desc=f"Resizing {folder_path}"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipped unreadable: {img_path}")
            continue
        resized = cv2.resize(img, output_size)
        cv2.imwrite(str(img_path), resized)

def check_and_resize(root: Path):
    for split, classes in CLASS_DIRS.items():
        for cls in classes:
            folder = root / split / cls
            if not folder.exists():
                print(f"Missing folder: {folder}")
                continue
            resize_images_in_folder(folder)
    print("All images resized successfully!")

def dataset_stats(root: Path):
    print("\nDataset Summary:")
    for split, classes in CLASS_DIRS.items():
        for cls in classes:
            folder = root / split / cls
            count = sum(1 for _ in iter_images(folder)) if folder.exists() else 0
            print(f"{split}/{cls}: {count} images")

if __name__ == "__main__":
    print("CWD:", Path.cwd())
    print("Root exists?", dataset_dir.exists())
    check_and_resize(dataset_dir)
    dataset_stats(dataset_dir)

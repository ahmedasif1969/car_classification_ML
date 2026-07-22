import os
import shutil
import zipfile
import requests
from pathlib import Path
import random

# Configuration
SAS_URL = "https://mlp.blob.core.windows.net/cars-dataset/archive.zip?sp=r&st=2026-07-17T06:35:31Z&se=2026-07-17T14:50:31Z&spr=https&sv=2026-02-06&sr=b&sig=AZxrls8qdjPc2vyDotktsubkCq7HE15Ck7bC69u%2BE90%3D"
ZIP_PATH = Path("archive.zip")
EXTRACT_DIR = Path("data/raw_dataset")
SPLIT_DIR = Path("data/split")
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
SPLIT_RATIO = 0.8  # 80% train, 20% validation
SEED = 42

def download_file(url, dest_path):
    print(f"Downloading dataset from Azure Blob to {dest_path}...")
    # Use streaming to handle potentially large files without memory issues
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        # Print progress every 10%
                        if int(percent) % 10 == 0 and int((downloaded - len(chunk)) / total_size * 100) != int(percent):
                            print(f"Downloaded: {percent:.1f}% ({downloaded}/{total_size} bytes)")
    print("Download completed.")

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction completed.")

def find_class_root(directory):
    # Find the directory containing classes. A directory contains classes if it has subdirectories
    # which contain image files, and no image files directly in itself.
    # Alternatively, let's find folders that contain images.
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # We walk the tree and find the deepest directories that contain files with image extensions.
    # The parent of those directories is likely the class root.
    leaf_dirs_with_images = set()
    for root, dirs, files in os.walk(directory):
        has_images = any(Path(f).suffix.lower() in image_extensions for f in files)
        if has_images:
            leaf_dirs_with_images.add(Path(root))
            
    if not leaf_dirs_with_images:
        raise ValueError("No images found in the extracted files.")
        
    # Find the common parent of these leaf directories.
    # Usually they all share the same immediate parent directory.
    # Let's group them by parent.
    parents = {}
    for path in leaf_dirs_with_images:
        parent = path.parent
        parents[parent] = parents.get(parent, 0) + 1
        
    # The parent with the most subdirectories is likely our class root.
    best_parent = max(parents, key=parents.get)
    print(f"Found class root directory: {best_parent} containing {parents[best_parent]} class subdirectories.")
    return best_parent

def split_dataset(class_root_dir, train_dir, val_dir, split_ratio=0.8):
    print("Splitting dataset into train and validation sets...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    random.seed(SEED)
    
    # Clean previous splits if they exist
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
        
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    classes = [d for d in class_root_dir.iterdir() if d.is_dir()]
    
    stats = {}
    
    for cls in classes:
        cls_name = cls.name
        # Find all images in this class folder
        cls_images = [f for f in cls.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        
        if len(cls_images) == 0:
            print(f"Warning: Class {cls_name} contains no images. Skipping.")
            continue
            
        random.shuffle(cls_images)
        
        # Determine split index
        split_idx = max(1, int(len(cls_images) * split_ratio))
        
        train_images = cls_images[:split_idx]
        val_images = cls_images[split_idx:]
        
        # Create directories
        (train_dir / cls_name).mkdir(parents=True, exist_ok=True)
        (val_dir / cls_name).mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for img in train_images:
            shutil.copy(img, train_dir / cls_name / img.name)
        for img in val_images:
            shutil.copy(img, val_dir / cls_name / img.name)
            
        stats[cls_name] = {'train': len(train_images), 'val': len(val_images), 'total': len(cls_images)}
        
    print("\nDataset Split Summary:")
    print("-" * 50)
    print(f"{'Class Name':<30} | {'Train':<6} | {'Val':<5} | {'Total':<5}")
    print("-" * 50)
    total_train = 0
    total_val = 0
    for cls_name, counts in sorted(stats.items()):
        print(f"{cls_name:<30} | {counts['train']:<6} | {counts['val']:<5} | {counts['total']:<5}")
        total_train += counts['train']
        total_val += counts['val']
    print("-" * 50)
    print(f"{'TOTAL':<30} | {total_train:<6} | {total_val:<5} | {total_train + total_val:<5}")
    print("-" * 50)
    
    return list(stats.keys())

def main():
    try:
        # Download ZIP
        if not ZIP_PATH.exists():
            download_file(SAS_URL, ZIP_PATH)
        else:
            print(f"Zip file {ZIP_PATH} already exists. Skipping download.")
            
        # Extract ZIP
        if not EXTRACT_DIR.exists():
            extract_zip(ZIP_PATH, EXTRACT_DIR)
        else:
            print(f"Extraction directory {EXTRACT_DIR} already exists. Skipping extraction.")
            
        # Find raw classes root
        class_root = find_class_root(EXTRACT_DIR)
        
        # Split dataset
        classes = split_dataset(class_root, TRAIN_DIR, VAL_DIR, SPLIT_RATIO)
        
        # Optional: clean up raw extract and zip to save disk space
        print("Cleaning up raw extraction to save space...")
        shutil.rmtree(EXTRACT_DIR)
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
            
        print("Dataset downloaded and prepared successfully!")
        
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        raise e

if __name__ == "__main__":
    main()

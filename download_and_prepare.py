import os
import shutil
import zipfile
import requests
from pathlib import Path
import random
from merge_datasets import merge_and_split

# Configuration & Data Source SAS URLs
OLD_DATASET_SAS_URL = "https://mlp.blob.core.windows.net/cars-dataset/archive.zip?sp=r&st=2026-07-17T06:35:31Z&se=2026-07-17T14:50:31Z&spr=https&sv=2026-02-06&sr=b&sig=AZxrls8qdjPc2vyDotktsubkCq7HE15Ck7bC69u%2BE90%3D"
NEW_DATASET_SAS_URL = "https://mlp.blob.core.windows.net/cars-dataset/Dataset.zip?sp=r&st=2026-07-22T10:36:31Z&se=2026-07-22T18:51:31Z&spr=https&sv=2026-02-06&sr=b&sig=Hll9OkbAEZKnuoe3l9FlOG9IEefQr7s6ZUnd%2B7vhRnU%3D"

OLD_ZIP_PATH = Path("archive.zip")
NEW_ZIP_PATH = Path("Dataset.zip")

RAW_NEW_DIR = Path("data/new_raw_dataset")
SPLIT_DIR = Path("data/split")

def download_file(url, dest_path):
    print(f"Downloading dataset from Azure Blob to {dest_path}...")
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
                        if int(percent) % 10 == 0 and int((downloaded - len(chunk)) / total_size * 100) != int(percent):
                            print(f"Downloaded: {percent:.1f}% ({downloaded}/{total_size} bytes)")
    print("Download completed.")

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction completed.")

def main():
    try:
        # If merged split dataset already exists, skip
        if SPLIT_DIR.exists() and any(SPLIT_DIR.iterdir()):
            print(f"Split dataset already exists at {SPLIT_DIR}. If you wish to re-download and re-merge, delete {SPLIT_DIR} first.")
            return

        # 1. Download & extract new dataset if not present
        if not RAW_NEW_DIR.exists():
            if not NEW_ZIP_PATH.exists():
                download_file(NEW_DATASET_SAS_URL, NEW_ZIP_PATH)
            extract_zip(NEW_ZIP_PATH, RAW_NEW_DIR)

        # 2. Merge old and new datasets, deduplicate, and split
        merge_and_split()
        
        print("Dataset downloaded, merged, standardized, and prepared successfully!")

    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        raise e

if __name__ == "__main__":
    main()

import os
import shutil
import hashlib
import random
from pathlib import Path

# Configuration
NEW_RAW_DIR = Path("data/new_raw_dataset/Dataset")
OLD_SPLIT_DIR = Path("data/split")
TARGET_SPLIT_DIR = Path("data/split")
SPLIT_RATIO = 0.8  # 80% train, 20% validation
SEED = 42

# Mapping from old class names to standard class names
CLASS_MAPPING = {
    'Daiatsu_Core': 'Daihatsu_Coure',
    'Daiatsu_Hijet': 'Daihatsu_Hijet',
    'Daiatsu_Mira': 'Daihatsu_Mira',
    'Honda_Vezell': 'Honda_Vezel',
    'Honda_city_1994': 'Honda_city_1998-2002',
    'Honda_city_2000': 'Honda_city_1998-2002',
    'Honda_civic_1994': 'Honda_civic_1992-1995',
    'Honda_civic_2005': 'Honda_civic_2004-2005',
    'Honda_civic_2007': 'Honda_civic_2007-2010',
    'Honda_civic_2015': 'Honda_civic_2011-2015',
    'Honda_civic_2018': 'Honda_civic_2016-2021',
    'Suzuki_kyber': 'Suzuki_khyber',
    'Suzuki_margala': 'Suzuki_margalla',
    'Toyota HIACE 2000': 'Hiace_1990-2003',
    'Toyota_Landcruser': 'Toyota_Landcruiser',
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def get_file_hash(filepath):
    """Calculate MD5 hash of file content for fast deduplication."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def collect_images_by_class():
    """Gather all image paths grouped by target class name from both datasets."""
    class_to_files = {}  # target_class -> list of (file_path, file_hash, original_filename)

    # 1. Collect from old split directory if it exists
    if OLD_SPLIT_DIR.exists():
        print("Collecting images from existing dataset (data/split)...")
        for split in ['train', 'val']:
            split_path = OLD_SPLIT_DIR / split
            if not split_path.exists():
                continue
            for cls_folder in split_path.iterdir():
                if not cls_folder.is_dir():
                    continue
                orig_cls_name = cls_folder.name
                target_cls_name = CLASS_MAPPING.get(orig_cls_name, orig_cls_name)

                for img_file in cls_folder.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                        class_to_files.setdefault(target_cls_name, []).append(img_file)

    # 2. Collect from new raw dataset directory if it exists
    if NEW_RAW_DIR.exists():
        print("Collecting images from new raw dataset (data/new_raw_dataset/Dataset)...")
        for split in ['train', 'valid', 'test']:
            split_path = NEW_RAW_DIR / split
            if not split_path.exists():
                continue
            for cls_folder in split_path.iterdir():
                if not cls_folder.is_dir():
                    continue
                orig_cls_name = cls_folder.name
                target_cls_name = CLASS_MAPPING.get(orig_cls_name, orig_cls_name)

                for img_file in cls_folder.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                        class_to_files.setdefault(target_cls_name, []).append(img_file)

    return class_to_files

def merge_and_split():
    print("Starting dataset merge and re-splitting process...")
    random.seed(SEED)

    class_to_files = collect_images_by_class()

    print(f"Total target classes identified: {len(class_to_files)}")

    # Temporary directory to build clean split dataset
    temp_split_dir = Path("data/merged_split")
    temp_train_dir = temp_split_dir / "train"
    temp_val_dir = temp_split_dir / "val"

    if temp_split_dir.exists():
        shutil.rmtree(temp_split_dir)

    temp_train_dir.mkdir(parents=True, exist_ok=True)
    temp_val_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    total_images_processed = 0
    total_duplicates_skipped = 0

    for cls_name, file_list in class_to_files.items():
        # Deduplicate files by filename first, then by MD5 hash
        seen_hashes = set()
        seen_filenames = set()
        unique_files = []

        for f in file_list:
            fname = f.name
            if fname in seen_filenames:
                total_duplicates_skipped += 1
                continue

            fhash = get_file_hash(f)
            if fhash in seen_hashes:
                total_duplicates_skipped += 1
                continue

            seen_filenames.add(fname)
            seen_hashes.add(fhash)
            unique_files.append(f)

        random.shuffle(unique_files)
        total_count = len(unique_files)

        if total_count == 0:
            continue

        split_idx = max(1, int(total_count * SPLIT_RATIO))
        train_files = unique_files[:split_idx]
        val_files = unique_files[split_idx:]

        cls_train_dir = temp_train_dir / cls_name
        cls_val_dir = temp_val_dir / cls_name
        cls_train_dir.mkdir(parents=True, exist_ok=True)
        cls_val_dir.mkdir(parents=True, exist_ok=True)

        for f in train_files:
            shutil.copy2(f, cls_train_dir / f.name)

        for f in val_files:
            shutil.copy2(f, cls_val_dir / f.name)

        stats[cls_name] = {'train': len(train_files), 'val': len(val_files), 'total': total_count}
        total_images_processed += total_count

    # Replace old data/split with new temp_split_dir
    print("Replacing data/split with merged dataset...")
    if TARGET_SPLIT_DIR.exists():
        shutil.rmtree(TARGET_SPLIT_DIR)

    shutil.move(temp_split_dir, TARGET_SPLIT_DIR)

    # Clean up raw extract directories and zip files to save disk space
    raw_zip = Path("data/new_dataset.zip")
    raw_dir = Path("data/new_raw_dataset")
    if raw_zip.exists():
        raw_zip.unlink()
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    # Display Summary Table
    print("\n" + "=" * 65)
    print("                MERGED DATASET SPLIT SUMMARY")
    print("=" * 65)
    print(f"{'Class Name':<32} | {'Train':<6} | {'Val':<5} | {'Total':<5}")
    print("-" * 65)
    total_train = 0
    total_val = 0
    for cls_name, counts in sorted(stats.items()):
        print(f"{cls_name:<32} | {counts['train']:<6} | {counts['val']:<5} | {counts['total']:<5}")
        total_train += counts['train']
        total_val += counts['val']
    print("-" * 65)
    print(f"{'TOTAL':<32} | {total_train:<6} | {total_val:<5} | {total_train + total_val:<5}")
    print(f"Total Duplicate Images Skipped: {total_duplicates_skipped}")
    print("=" * 65 + "\n")

    print(f"Dataset successfully merged and split into {TARGET_SPLIT_DIR}!")

if __name__ == "__main__":
    merge_and_split()

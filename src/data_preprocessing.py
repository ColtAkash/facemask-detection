"""
data_preprocessing.py
-----------------------------------------------------------------------------
Converts the andrewmvd/face-mask-detection Kaggle dataset (Pascal VOC format)
into a structured classification dataset ready for the three experiments.

Raw dataset layout (after unzipping the Kaggle download):
    <raw_dir>/
    +-- images/          (.png images)
    +-- annotations/     (.xml files, Pascal VOC bounding boxes)

Output layout:
    dataset_processed/
    +-- Train/
    |   +-- with_mask/
    |   +-- without_mask/
    |   +-- mask_weared_incorrect/
    +-- Validation/
    |   +-- ...
    +-- Test/
        +-- ...

Usage:
    python src/data_preprocessing.py --raw_dir path/to/raw_dataset

    # or with explicit output dir:
    python src/data_preprocessing.py --raw_dir raw/ --out_dir dataset_processed/
"""

import os
import sys
import random
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow is not installed. Run: pip install Pillow")
    sys.exit(1)

CLASSES      = ['with_mask', 'without_mask', 'mask_weared_incorrect']
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO  = 0.15  (remainder)
PADDING      = 10     # pixels of context to keep around each face crop
RANDOM_SEED  = 42


# --- XML parser ---------------------------------------------------------------

def parse_annotation(xml_path):
    """Return (filename, list_of_objects) from a Pascal VOC XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_node = root.find('filename')
    filename = filename_node.text if filename_node is not None else None

    objects = []
    for obj in root.findall('object'):
        name_node = obj.find('name')
        if name_node is None:
            continue
        name = name_node.text.strip()

        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        try:
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
        except (TypeError, ValueError):
            continue

        if xmax > xmin and ymax > ymin:
            objects.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})

    return filename, objects


def find_image_path(images_dir, filename):
    """Try multiple extensions in case the XML filename doesn't match exactly."""
    direct = os.path.join(images_dir, filename)
    if os.path.isfile(direct):
        return direct

    base = os.path.splitext(filename)[0]
    for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
        candidate = os.path.join(images_dir, base + ext)
        if os.path.isfile(candidate):
            return candidate

    return None


# --- Crop + save --------------------------------------------------------------

def save_crop(img, bbox, cls_name, split, output_dir, counter):
    """Crop bounding box from image (with padding) and save as JPEG."""
    xmin, ymin, xmax, ymax = bbox
    w, h = img.size

    # Add padding, clamp to image bounds
    xmin_p = max(0, xmin - PADDING)
    ymin_p = max(0, ymin - PADDING)
    xmax_p = min(w, xmax + PADDING)
    ymax_p = min(h, ymax + PADDING)

    crop = img.crop((xmin_p, ymin_p, xmax_p, ymax_p))
    save_dir = os.path.join(output_dir, split, cls_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'{cls_name}_{counter:06d}.jpg')
    crop.save(out_path, 'JPEG', quality=95)
    return out_path


# --- Main ---------------------------------------------------------------------

def process_dataset(raw_dir, output_dir):
    images_dir = os.path.join(raw_dir, 'images')
    annots_dir = os.path.join(raw_dir, 'annotations')

    for d in [images_dir, annots_dir]:
        if not os.path.isdir(d):
            print(f"[ERROR] Expected directory not found: {d}")
            print("  Make sure --raw_dir points to the folder that contains")
            print("  'images/' and 'annotations/' sub-folders.")
            sys.exit(1)

    xml_files = sorted([f for f in os.listdir(annots_dir) if f.endswith('.xml')])
    print(f"\n  Found {len(xml_files)} annotation XML files.")

    # -- Collect all (image_path, bbox, class) tuples -------------------------
    all_crops    = []
    skipped_imgs = 0
    skipped_cls  = 0
    class_totals = defaultdict(int)

    for xml_file in xml_files:
        xml_path = os.path.join(annots_dir, xml_file)
        filename, objects = parse_annotation(xml_path)

        if filename is None:
            skipped_imgs += 1
            continue

        img_path = find_image_path(images_dir, filename)
        if img_path is None:
            skipped_imgs += 1
            continue

        for obj in objects:
            cls = obj['name']
            if cls not in CLASSES:
                skipped_cls += 1
                continue
            all_crops.append((img_path, obj['bbox'], cls))
            class_totals[cls] += 1

    total = len(all_crops)
    print(f"\n  Total face crops to process: {total:,}")
    print(f"  Skipped (image not found): {skipped_imgs}")
    print(f"  Skipped (unknown class):   {skipped_cls}")
    print(f"\n  Class distribution (before split):")
    for cls in CLASSES:
        n = class_totals[cls]
        pct = 100 * n / total if total else 0
        print(f"    {cls:<30} {n:>6,}  ({pct:.1f}%)")

    if total == 0:
        print("\n[ERROR] No valid crops found. Check your raw_dir path.")
        sys.exit(1)

    # -- Shuffle and split by class to preserve class ratios ------------------
    random.seed(RANDOM_SEED)

    by_class = defaultdict(list)
    for item in all_crops:
        by_class[item[2]].append(item)

    splits = {'Train': [], 'Validation': [], 'Test': []}
    for cls, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        splits['Train']      += items[:n_train]
        splits['Validation'] += items[n_train:n_train + n_val]
        splits['Test']       += items[n_train + n_val:]

    # Shuffle within each split
    for split_items in splits.values():
        random.shuffle(split_items)

    # -- Save crops ------------------------------------------------------------
    print(f"\n  Saving crops to: {output_dir}")
    counters = defaultdict(int)
    errors   = 0

    for split_name, items in splits.items():
        for img_path, bbox, cls_name in items:
            try:
                img = Image.open(img_path).convert('RGB')
                counters[(split_name, cls_name)] += 1
                save_crop(img, bbox, cls_name, split_name, output_dir,
                          counters[(split_name, cls_name)])
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [Warning] Could not process {img_path}: {e}")

    # -- Summary ---------------------------------------------------------------
    print(f"\n  {'-'*54}")
    print(f"  {'Split':<14} {'Class':<30} {'Count':>6}")
    print(f"  {'-'*54}")
    grand_total = 0
    for split_name in ('Train', 'Validation', 'Test'):
        for cls in CLASSES:
            n = counters.get((split_name, cls), 0)
            grand_total += n
            print(f"  {split_name:<14} {cls:<30} {n:>6,}")
        print(f"  {'-'*54}")
    print(f"  {'Total':<45} {grand_total:>6,}")
    if errors:
        print(f"\n  [Warning] {errors} crops could not be saved (see above).")

    print(f"\n  [Done] Processed dataset saved to: {output_dir}")
    print(f"\n  Next step:")
    print(f"    python run_all.py --data_dir \"{output_dir}\"")
    return output_dir


# --- Entry point --------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess andrewmvd/face-mask-detection dataset')
    parser.add_argument(
        '--raw_dir',
        required=True,
        help='Path to unzipped Kaggle download (must contain images/ and annotations/)')
    parser.add_argument(
        '--out_dir',
        default=os.path.join(os.path.dirname(__file__), '..', 'dataset'),
        help='Where to save the processed Train/Validation/Test folders '
             '(default: ./dataset/)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  FACE MASK DATASET PREPROCESSING")
    print(f"  Raw dir : {os.path.abspath(args.raw_dir)}")
    print(f"  Out dir : {os.path.abspath(args.out_dir)}")
    print(f"  Split   : {int(TRAIN_RATIO*100)}% train / "
          f"{int(VAL_RATIO*100)}% val / "
          f"{int((1-TRAIN_RATIO-VAL_RATIO)*100)}% test")
    print(f"  Classes : {CLASSES}")
    print('='*60)

    process_dataset(
        raw_dir=os.path.abspath(args.raw_dir),
        output_dir=os.path.abspath(args.out_dir))

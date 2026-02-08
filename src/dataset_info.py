import os
import sys
from pathlib import Path
from collections import Counter

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, DATASET_CONFIG
from src.utils import print_colored

def show_dataset_info():

    print_colored("\n" + "=" * 65, "cyan")
    print_colored("  DATASET INFORMATION", "bold")
    print_colored("=" * 65, "cyan")

    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        print_colored("\n  Dataset not set up yet! Run setup_dataset.py first.", "red")
        return

    print_colored("\n  IMAGE COUNT PER SPLIT", "bold")
    print("-" * 45)

    total_images = 0
    total_labels = 0

    for split in ["train", "val", "test"]:
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"

        img_count = 0
        lbl_count = 0

        if img_dir.exists():
            img_count = len([f for f in img_dir.iterdir()
                             if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if lbl_dir.exists():
            lbl_count = len([f for f in lbl_dir.iterdir() if f.suffix == '.txt'])

        total_images += img_count
        total_labels += lbl_count
        print(f"  {split.upper():<8s}: {img_count:>6,} images  |  {lbl_count:>6,} labels")

    print(f"  {'TOTAL':<8s}: {total_images:>6,} images  |  {total_labels:>6,} labels")

    if total_images > 0:
        print_colored("\n  SPLIT RATIOS", "bold")
        print("-" * 45)
        for split in ["train", "val", "test"]:
            img_dir = DATA_DIR / split / "images"
            if img_dir.exists():
                count = len([f for f in img_dir.iterdir()
                             if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                pct = count / total_images * 100
                bar = "#" * int(pct / 2)
                print(f"  {split.upper():<8s}: {pct:5.1f}%  {bar}")

    print_colored("\n  IMAGE SIZE ANALYSIS", "bold")
    print("-" * 45)
    print("  Scanning image dimensions (sampling up to 500 images)...")

    all_images = []
    for split in ["train", "val", "test"]:
        img_dir = DATA_DIR / split / "images"
        if img_dir.exists():
            all_images.extend(
                [f for f in img_dir.iterdir()
                 if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            )

    widths = []
    heights = []
    sizes_counter = Counter()
    sample = all_images[:500]

    for img_path in tqdm(sample, desc="  Scanning", leave=False):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                sizes_counter[(w, h)] += 1
        except Exception:
            continue

    if widths:
        print(f"  Sampled          : {len(widths)} images")
        print(f"  Min dimensions   : {min(widths)} x {min(heights)}")
        print(f"  Max dimensions   : {max(widths)} x {max(heights)}")
        print(f"  Average          : {sum(widths)//len(widths)} x {sum(heights)//len(heights)}")

        unique_sizes = len(sizes_counter)
        if unique_sizes == 1:
            size = list(sizes_counter.keys())[0]
            print_colored(f"  All same size    : YES ({size[0]}x{size[1]})", "green")
        else:
            print_colored(f"  All same size    : NO ({unique_sizes} different sizes)", "yellow")
            print(f"\n  Most common sizes:")
            for (w, h), count in sizes_counter.most_common(5):
                pct = count / len(widths) * 100
                print(f"    {w}x{h}: {count} images ({pct:.1f}%)")

        ratios = [w / h for w, h in zip(widths, heights)]
        avg_ratio = sum(ratios) / len(ratios)
        landscape = sum(1 for r in ratios if r > 1.05)
        portrait = sum(1 for r in ratios if r < 0.95)
        square = len(ratios) - landscape - portrait

        print(f"\n  Aspect Ratio (avg): {avg_ratio:.2f}")
        print(f"  Landscape         : {landscape} ({landscape/len(ratios)*100:.1f}%)")
        print(f"  Portrait          : {portrait} ({portrait/len(ratios)*100:.1f}%)")
        print(f"  Square            : {square} ({square/len(ratios)*100:.1f}%)")

    print_colored("\n  CLASS DISTRIBUTION (from labels)", "bold")
    print("-" * 45)

    class_counts = Counter()
    total_annotations = 0
    empty_labels = 0

    for split in ["train", "val", "test"]:
        lbl_dir = DATA_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.iterdir():
            if lbl_file.suffix != '.txt':
                continue
            try:
                with open(lbl_file, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                if not lines:
                    empty_labels += 1
                    continue
                for line in lines:
                    parts = line.split()
                    if parts:
                        cls_id = int(parts[0])
                        class_counts[cls_id] += 1
                        total_annotations += 1
            except Exception:
                continue

    class_names = DATASET_CONFIG.get("classes", [])

    if class_counts:
        print(f"  Total annotations  : {total_annotations:,}")
        print(f"  Empty label files  : {empty_labels:,}")
        print(f"  Avg per image      : {total_annotations / max(total_labels, 1):.1f}")
        print()
        print(f"  {'Class':<35s} {'Count':>8s} {'Pct':>7s}")
        print(f"  {'-----':<35s} {'-----':>8s} {'---':>7s}")
        for cls_id in sorted(class_counts.keys()):
            name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
            count = class_counts[cls_id]
            pct = count / total_annotations * 100
            bar = "#" * int(pct / 2)
            print(f"  {name:<35s} {count:>8,} {pct:>6.1f}%  {bar}")

    print_colored("\n  DISK USAGE", "bold")
    print("-" * 45)

    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if split_dir.exists():
            size_bytes = sum(f.stat().st_size for f in split_dir.rglob("*") if f.is_file())
            size_gb = size_bytes / (1024 ** 3)
            if size_gb >= 1:
                print(f"  {split.upper():<8s}: {size_gb:.2f} GB")
            else:
                size_mb = size_bytes / (1024 ** 2)
                print(f"  {split.upper():<8s}: {size_mb:.1f} MB")

    total_size = sum(
        f.stat().st_size
        for split in ["train", "val", "test"]
        if (DATA_DIR / split).exists()
        for f in (DATA_DIR / split).rglob("*")
        if f.is_file()
    )
    total_gb = total_size / (1024 ** 3)
    print(f"  {'TOTAL':<8s}: {total_gb:.2f} GB")

    print_colored("\n  CONFIGURED COUNTRIES", "bold")
    print("-" * 45)
    countries = DATASET_CONFIG.get("countries", [])
    for c in countries:
        print(f"  - {c}")

    print_colored("\n" + "=" * 65, "cyan")

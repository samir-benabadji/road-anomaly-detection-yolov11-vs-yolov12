import os
import sys
import shutil
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

CLASS_MAPPING = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
}

CLASS_NAMES = ["D00_Longitudinal_Crack", "D10_Transverse_Crack",
               "D20_Alligator_Crack", "D40_Pothole"]

COUNTRIES = ["Norway", "Japan", "India", "Czech", "United_States"]

def update_config_splits(train_ratio: float, val_ratio: float, test_ratio: float):
    config_path = PROJECT_ROOT / "config.py"
    if not config_path.exists():
        print("  Warning: config.py not found, split ratios not updated.")
        return

    try:
        config_text = config_path.read_text(encoding="utf-8")

        replacements = {
            r'("train_split"\s*:\s*)[0-9.]+(,)': rf"\g<1>{train_ratio:.2f}\g<2>",
            r'("val_split"\s*:\s*)[0-9.]+(,)': rf"\g<1>{val_ratio:.2f}\g<2>",
            r'("test_split"\s*:\s*)[0-9.]+(,)': rf"\g<1>{test_ratio:.2f}\g<2>",
        }

        has_all_fields = all(re.search(pattern, config_text) for pattern in replacements)
        if not has_all_fields:
            print("  Warning: could not locate split fields in config.py.")
            return

        updated_text = config_text
        for pattern, repl in replacements.items():
            updated_text = re.sub(pattern, repl, updated_text, count=1)

        if updated_text != config_text:
            config_path.write_text(updated_text, encoding="utf-8")
            print(
                f"  Updated config.py splits to "
                f"{int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}"
            )
        else:
            print(
                f"  config.py splits already set to "
                f"{int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}"
            )
    except Exception as exc:
        print(f"  Warning: failed to update config.py split ratios ({exc})")

def convert_xml_to_yolo(xml_path: Path) -> list:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        labels = []

        for obj in root.findall("object"):
            class_elem = obj.find("n")
            if class_elem is None:
                class_elem = obj.find("name")
            if class_elem is None:
                continue

            class_name = class_elem.text
            if class_name not in CLASS_MAPPING:
                continue

            class_id = CLASS_MAPPING[class_name]

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            x_center = max(0, min(1, ((xmin + xmax) / 2) / img_width))
            y_center = max(0, min(1, ((ymin + ymax) / 2) / img_height))
            width = max(0, min(1, (xmax - xmin) / img_width))
            height = max(0, min(1, (ymax - ymin) / img_height))

            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return labels

    except Exception as e:
        print(f"  Error parsing {xml_path}: {e}")
        return []

def find_rdd2022_path() -> Path:
    print("\n" + "=" * 60)
    print("  RDD2022 DATASET SETUP")
    print("=" * 60)

    default_path = DATA_DIR / "RDD2022_all_countries"

    print(f"\nDefault path: {default_path}")
    user_input = input("\nPress Enter to use default, or enter your path: ").strip()

    if user_input:
        rdd_path = Path(user_input)
    else:
        rdd_path = default_path

    if not rdd_path.exists():
        print(f"\nPath does not exist: {rdd_path}")
        print(f"\nPlease enter the path to the RDD2022 folder")
        print(f"(Should contain: {', '.join(COUNTRIES)})")
        user_input = input("\nPath: ").strip()
        rdd_path = Path(user_input)

    return rdd_path

def setup_dataset():
    rdd_path = find_rdd2022_path()

    if not rdd_path.exists():
        print(f"\nPath does not exist: {rdd_path}")
        return False

    print(f"\nFound dataset at: {rdd_path}")

    all_data = []

    print("\nScanning for images and annotations...")

    for country in COUNTRIES:
        country_path = rdd_path / country
        if not country_path.exists():
            print(f"  Skipping {country} (not found)")
            continue

        train_path = country_path / "train"
        if not train_path.exists():
            nested = country_path / country
            if nested.exists() and (nested / "train").exists():
                country_path = nested
                train_path = country_path / "train"
                print(f"  {country}: using nested folder ({country}/{country}/)")
            else:
                print(f"  Skipping {country} (no train folder)")
                continue

        images_path = train_path / "images"
        annotations_path = train_path / "annotations" / "xmls"

        if not images_path.exists():
            print(f"  Skipping {country} (no images folder)")
            continue

        if not annotations_path.exists():
            annotations_path = train_path / "annotations"
            if not annotations_path.exists():
                print(f"  Skipping {country} (no annotations folder)")
                continue

        images = list({f for f in images_path.iterdir()
                       if f.suffix.lower() in ('.jpg', '.jpeg', '.png')})
        count = 0

        for img_path in images:
            xml_path = annotations_path / f"{img_path.stem}.xml"
            if not xml_path.exists():
                xml_path = annotations_path / "xmls" / f"{img_path.stem}.xml"

            if xml_path.exists():
                all_data.append((img_path, xml_path, country))
                count += 1

        print(f"  {country}: {count} annotated images")

    total = len(all_data)
    print(f"\nTotal annotated images found: {total}")

    if total == 0:
        print("\nNo annotated images found!")
        return False

    print("\nChoose dataset split ratio:")
    print("  [1] 70% Train / 15% Val / 15% Test (default)")
    print("  [2] 80% Train / 10% Val / 10% Test")
    split_choice = input("\n  Your choice [1]: ").strip()

    if split_choice == "2":
        train_ratio, val_ratio, test_ratio = 0.80, 0.10, 0.10
        test_size = 0.20
    else:
        train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15
        test_size = 0.30

    print(f"\nSplitting dataset ({int(train_ratio*100)}% train, "
          f"{int(val_ratio*100)}% val, {int(test_ratio*100)}% test)...")

    train_data, temp_data = train_test_split(all_data, test_size=test_size, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)

    print(f"  Train: {len(train_data)} images ({len(train_data)/total*100:.1f}%)")
    print(f"  Val:   {len(val_data)} images ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} images ({len(test_data)/total*100:.1f}%)")

    print("\nCreating directory structure...")

    for split in ["train", "val", "test"]:
        (DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    splits = [("train", train_data), ("val", val_data), ("test", test_data)]

    for split_name, split_data in splits:
        print(f"\nProcessing {split_name}...")

        for img_path, xml_path, country in tqdm(split_data, desc=f"  {split_name}"):
            dst_img = DATA_DIR / split_name / "images" / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            labels = convert_xml_to_yolo(xml_path)

            label_path = DATA_DIR / split_name / "labels" / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(labels))

    print("\nCreating data.yaml...")

    yaml_content = f"""# Road Anomaly Detection Dataset (RDD2022)
# Converted for YOLO11/YOLO12 training
# Split: {int(train_ratio*100)}% Train / {int(val_ratio*100)}% Val / {int(test_ratio*100)}% Test
# Countries: {', '.join(COUNTRIES)}
# Projet Fin d'Etude - Master 2

path: {DATA_DIR.absolute()}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {len(CLASS_MAPPING)}

# Class names
names:
  0: D00_Longitudinal_Crack
  1: D10_Transverse_Crack
  2: D20_Alligator_Crack
  3: D40_Pothole
"""

    yaml_path = DATA_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"  Saved: {yaml_path}")
    update_config_splits(train_ratio, val_ratio, test_ratio)

    print("\n" + "=" * 60)
    print("  DATASET SETUP COMPLETE!")
    print("=" * 60)
    print(f"\nDataset location: {DATA_DIR}")
    print(f"Total images: {total}")
    print(f"Classes: {len(CLASS_MAPPING)}")
    print(f"Split: {int(train_ratio*100)}% train / {int(val_ratio*100)}% val / {int(test_ratio*100)}% test")
    print(f"\nCountries: {', '.join(COUNTRIES)}")
    print("\nClass distribution:")
    for code, idx in CLASS_MAPPING.items():
        print(f"  [{idx}] {code}: {CLASS_NAMES[idx]}")

    print("\nYou can now run: python main.py")

    return True

def setup_individual_country_splits():
    rdd_path = find_rdd2022_path()

    if not rdd_path.exists():
        print(f"\nPath does not exist: {rdd_path}")
        return False

    print(f"\nFound dataset at: {rdd_path}")
    print("\nMode: Per-Country Individual Splits (80% train / 10% val / 10% test EACH)")

    individual_dir = DATA_DIR / "individual"

    for country in COUNTRIES:
        print(f"\n{'='*50}")
        print(f"  Processing {country}...")
        print(f"{'='*50}")

        country_path = rdd_path / country
        if not country_path.exists():
            print(f"  Skipping {country} (not found)")
            continue

        train_path = country_path / "train"
        if not train_path.exists():
            nested = country_path / country
            if nested.exists() and (nested / "train").exists():
                country_path = nested
                train_path = country_path / "train"
            else:
                print(f"  Skipping {country} (no train folder)")
                continue

        images_path = train_path / "images"
        annotations_path = train_path / "annotations" / "xmls"

        if not images_path.exists():
            print(f"  Skipping {country} (no images folder)")
            continue

        if not annotations_path.exists():
            annotations_path = train_path / "annotations"
            if not annotations_path.exists():
                print(f"  Skipping {country} (no annotations folder)")
                continue

        country_data = []
        images = list({f for f in images_path.iterdir()
                       if f.suffix.lower() in ('.jpg', '.jpeg', '.png')})

        for img_path in images:
            xml_path = annotations_path / f"{img_path.stem}.xml"
            if not xml_path.exists():
                xml_path = annotations_path / "xmls" / f"{img_path.stem}.xml"
            if xml_path.exists():
                country_data.append((img_path, xml_path))

        total = len(country_data)
        print(f"  Found {total} annotated images")

        if total == 0:
            print(f"  Skipping {country} (no annotated images)")
            continue

        train_data, temp_data = train_test_split(country_data, test_size=0.20, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)

        print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

        country_dir = individual_dir / country
        for split in ["train", "val", "test"]:
            (country_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (country_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        splits = [("train", train_data), ("val", val_data), ("test", test_data)]

        for split_name, split_data in splits:
            for img_path, xml_path in tqdm(split_data, desc=f"  {country}/{split_name}"):
                dst_img = country_dir / split_name / "images" / img_path.name
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

                labels = convert_xml_to_yolo(xml_path)
                label_path = country_dir / split_name / "labels" / f"{img_path.stem}.txt"
                with open(label_path, "w") as f:
                    f.write("\n".join(labels))

        yaml_content = f"""# Individual Country Dataset - {country}
# Split: 80% Train / 10% Val / 10% Test (this country ONLY)
# Projet Fin d'Etude - Master 2

path: {(country_dir).absolute()}
train: train/images
val: val/images
test: test/images

nc: {len(CLASS_MAPPING)}

names:
  0: D00_Longitudinal_Crack
  1: D10_Transverse_Crack
  2: D20_Alligator_Crack
  3: D40_Pothole
"""
        yaml_path = country_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"  Saved: {yaml_path}")

    print("\n" + "=" * 60)
    print("  PER-COUNTRY INDIVIDUAL SPLITS COMPLETE!")
    print("=" * 60)
    print(f"\nLocation: {individual_dir}")
    for country in COUNTRIES:
        cdir = individual_dir / country
        if cdir.exists():
            train_count = len(list((cdir / "train" / "images").iterdir())) if (cdir / "train" / "images").exists() else 0
            val_count = len(list((cdir / "val" / "images").iterdir())) if (cdir / "val" / "images").exists() else 0
            test_count = len(list((cdir / "test" / "images").iterdir())) if (cdir / "test" / "images").exists() else 0
            print(f"  {country}: {train_count} train / {val_count} val / {test_count} test")

    print("\nYou can now use menu options 10-13 in main.py")
    return True

if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("  RDD2022 DATASET SETUP")
        print("=" * 60)
        print("\nChoose setup mode:")
        print("  [1] Combined (all countries together) - for menus 1-9, A, B")
        print("  [2] Per-Country Individual Splits - for menus 10-13")
        print("      (80/10/10 split for EACH country separately)")

        mode = input("\n  Your choice [1]: ").strip()

        if mode == "2":
            success = setup_individual_country_splits()
        else:
            success = setup_dataset()

        if not success:
            print("\nSetup failed. Please check the dataset path.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

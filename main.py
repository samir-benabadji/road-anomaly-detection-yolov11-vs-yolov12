import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MENU_CONFIG, DATASET_CONFIG
from src.utils import clear_screen, print_colored, check_dataset
from src.train import (
    train_model,
    train_model_percountry,
    train_model_individual_country,
    train_model_svrdd,
    train_model_svrdd_variant2_to_variant5,
    train_model_svrdd_post_learning_cv,
    train_model_rdd2022_india,
    train_model_rdd2022_india_variant2_to_variant5,
)
from src.dataset_info import show_dataset_info

def _parse_batch_input(batch_input: str, default_batch: int) -> int | None:
    text = batch_input.strip().lower()
    if text in ("", "auto", "-1"):
        return None
    if text.isdigit() and int(text) > 0:
        return int(text)
    return default_batch

def _format_batch_display(batch_size: int | None) -> str:
    if batch_size is None:
        return "auto (VRAM-based)"
    return str(batch_size)

def print_header():
    clear_screen()
    print(MENU_CONFIG["title"])

def print_system_info():
    print_colored("  SYSTEM INFORMATION", "bold")
    print("-" * 50)

    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")

    if torch.cuda.is_available():
        print_colored(f"  GPU     : {torch.cuda.get_device_name(0)}", "green")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM    : {mem:.1f} GB")
    else:
        print_colored("  GPU     : Not available (CPU only - SLOW!)", "red")

    if check_dataset():
        print_colored("  Dataset : Ready", "green")
    else:
        print_colored("  Dataset : Not found - run: python setup_dataset.py", "red")

    print("-" * 50)

def print_menu():
    print_colored("\n  MAIN MENU", "bold")
    print("-" * 50)
    print("  [1]  Train YOLO11s (Standard)")
    print("  [2]  Train YOLO12s (Standard)")
    print("")
    print("  [3]  Train YOLO11s (Freeze + Cosine LR)")
    print("  [4]  Train YOLO12s (Freeze + Cosine LR)")
    print("")
    print_colored("  --- Per-Country Testing ---", "cyan")
    print("  [6]  YOLO11 Medium - Per-Country Test")
    print("  [7]  YOLO11 Large  - Per-Country Test")
    print("  [8]  YOLO12 Medium - Per-Country Test")
    print("  [9]  YOLO12 Large  - Per-Country Test")
    print("")
    print_colored("  --- Optimized (1280 + XLarge + Strong Aug + TTA) ---", "yellow")
    print("  [A]  YOLO11x Optimized - Per-Country Test")
    print("  [B]  YOLO12x Optimized - Per-Country Test")
    print("")
    print_colored("  --- Individual Country Training (each country separate) ---", "magenta")
    print("  [10] YOLO11 Large  - Individual Country")
    print("  [11] YOLO11 XLarge - Individual Country")
    print("  [12] YOLO12 Large  - Individual Country")
    print("  [13] YOLO12 XLarge - Individual Country")
    print("")
    print_colored("  --- SVRDD Dataset (2nd dataset comparison) ---", "green")
    print("  [15] YOLO11 Medium - SVRDD Dataset")
    print("  [16] YOLO12 Medium - SVRDD Dataset")
    print("  [17] YOLO11 Large  - SVRDD Dataset")
    print("  [18] YOLO12 Large  - SVRDD Dataset")
    print("")
    print_colored("  --- SVRDD Ablation Study (YOLO12-L architecture mods) ---", "yellow")
    print("  [19] YOLO12-L Deep A2C2f  - SVRDD (backbone A2C2f 4->6 blocks)")
    print("  [20] YOLO12-L P2 Head     - SVRDD (4th detection head at stride 4)")
    print("  [21] YOLO12-L P2 + Deep A2C2f - SVRDD (combined: P2 head + A2C2f 4->6)")
    print("  [24] YOLO12-L Wider Neck     - SVRDD (FPN channels 256->384, 512->768)")
    print("  [25] YOLO12-L P2 + Wider Neck - SVRDD (combined: P2 head + wider FPN)")
    print("  [27] YOLO12-L Two-Stage Transfer - SVRDD (Variant 2 -> Variant 5)")
    print("")
    print_colored("  --- RDD2022 Per-Country + India-Only Variants ---", "yellow")
    print("  [26] YOLO12-L P2 Head - RDD2022 Per-Country Test")
    print("  [28] YOLO12-L P2 + Wider Neck - RDD2022 India-Only")
    print("  [29] YOLO12-L Two-Stage Transfer - RDD2022 India-Only (Variant 2 -> Variant 5)")
    print("")
    print_colored("  --- SVRDD Post-Learning (Extra 1024x1024 Images + Cross-Validation) ---", "magenta")
    print("  [30] Baseline + extraImagesForSVRDDPostLearning (CV)")
    print("  [31] Variant 2 + extraImagesForSVRDDPostLearning (CV)")
    print("  [32] Variant 4 + extraImagesForSVRDDPostLearning (CV)")
    print("  [33] Variant 5 + extraImagesForSVRDDPostLearning (CV)")
    print("  [34] Variant 6 + extraImagesForSVRDDPostLearning (CV)")
    print("")
    print("  [5]  Dataset Info")
    print("  [0]  Exit")
    print("-" * 50)

def training_flow(model_key: str, improved: bool):
    model_name = model_key.upper()
    mode = "Freeze + Cosine LR" if improved else "Standard"

    print_header()
    print_colored(f"\n  TRAIN {model_name} ({mode})", "bold")
    print("-" * 50)

    if not check_dataset():
        print_colored("\n  Dataset not found!", "red")
        print("  Please run: python setup_dataset.py")
        input("\nPress Enter to continue...")
        return

    if torch.cuda.is_available():
        print_colored(f"  GPU: {torch.cuda.get_device_name(0)}", "green")
    else:
        print_colored("  WARNING: No GPU! Training will be very slow.", "yellow")

    print(f"\n  Model    : {model_name}")
    print(f"  Mode     : {mode}")
    print(f"  Classes  : {DATASET_CONFIG['num_classes']}")

    if improved:
        print(f"\n  Improvement technique:")
        print(f"    - Freeze first 10 backbone layers")
        print(f"    - Cosine annealing LR schedule")
        print(f"    - AdamW optimizer, lr=0.001")

    default_epochs = 100
    epochs_input = input(f"\n  Number of epochs [{default_epochs}]: ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else default_epochs

    default_batch = 16
    default_batch_label = "auto" if torch.cuda.is_available() else str(default_batch)
    batch_input = input(f"  Batch size [{default_batch_label}]: ").strip()
    batch_size = _parse_batch_input(batch_input, default_batch)

    print_colored(f"\n  Ready to start {model_name} ({mode}) training?", "yellow")
    print(f"  Epochs: {epochs}, Batch: {_format_batch_display(batch_size)}")
    confirm = input("  Type 'yes' to start: ").strip().lower()

    if confirm != "yes":
        print("  Training cancelled.")
        input("\nPress Enter to continue...")
        return

    result = train_model(
        model_key=model_key,
        improved=improved,
        epochs=epochs,
        batch_size=batch_size,
    )

    if result:
        print_colored(f"\n  Training complete!", "green")
        print(f"  Results: {result['results_dir']}")
    else:
        print_colored(f"\n  Training failed. Check errors above.", "red")

    input("\nPress Enter to continue...")

def percountry_training_flow(model_key: str, size: str,
                             variant_key_override: str = None,
                             variant_description: str = None):
    model_name = model_key.upper()
    size_label = {"m": "Medium", "l": "Large", "x": "XLarge"}[size]
    variant = variant_key_override or f"{model_key}{size}"

    print_header()
    print_colored(f"\n  {model_name} {size_label} - PER-COUNTRY TEST", "bold")
    print("-" * 50)

    if not check_dataset():
        print_colored("\n  Dataset not found!", "red")
        print("  Please run: python setup_dataset.py")
        input("\nPress Enter to continue...")
        return

    if torch.cuda.is_available():
        print_colored(f"  GPU: {torch.cuda.get_device_name(0)}", "green")
    else:
        print_colored("  WARNING: No GPU! Training will be very slow.", "yellow")

    print(f"\n  Model    : {model_name} ({size_label})")
    print(f"  Variant  : {variant}")
    if variant_description:
        print(f"  Modif.   : {variant_description}")
    print(f"  Classes  : {DATASET_CONFIG['num_classes']}")
    print(f"\n  Strategy:")
    print(f"    Train : ALL 5 countries combined")
    print(f"    Val   : ALL 5 countries combined")
    print(f"    Test  : EACH country separately (5 evaluations)")

    default_epochs = 100
    epochs_input = input(f"\n  Number of epochs [{default_epochs}]: ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else default_epochs

    default_batch = 16
    default_batch_label = "auto" if torch.cuda.is_available() else str(default_batch)
    batch_input = input(f"  Batch size [{default_batch_label}]: ").strip()
    batch_size = _parse_batch_input(batch_input, default_batch)

    print_colored(f"\n  Ready to start {model_name} {size_label} (Per-Country Test)?", "yellow")
    print(f"  Epochs: {epochs}, Batch: {_format_batch_display(batch_size)}")
    confirm = input("  Type 'yes' to start: ").strip().lower()

    if confirm != "yes":
        print("  Training cancelled.")
        input("\nPress Enter to continue...")
        return

    result = train_model_percountry(
        model_key=model_key,
        size=size,
        epochs=epochs,
        batch_size=batch_size,
        variant_key=variant_key_override,
    )

    if result:
        print_colored(f"\n  Per-country training & evaluation complete!", "green")
        print(f"  Results: {result['results_dir']}")
    else:
        print_colored(f"\n  Training failed. Check errors above.", "red")

    input("\nPress Enter to continue...")

def optimized_percountry_flow(model_key: str):
    model_name = model_key.upper()
    variant = f"{model_key}x"

    print_header()
    print_colored(f"\n  {model_name} XLarge OPTIMIZED - PER-COUNTRY TEST", "bold")
    print("-" * 50)

    if not check_dataset():
        print_colored("\n  Dataset not found!", "red")
        print("  Please run: python setup_dataset.py")
        input("\nPress Enter to continue...")
        return

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_colored(f"  GPU: {gpu_name} ({vram:.0f} GB VRAM)", "green")
        if vram < 20:
            print_colored("  WARNING: XLarge + imgsz=1280 needs ~24GB+ VRAM!", "yellow")
            print_colored("  Recommended: RTX 5090 (32GB) or similar.", "yellow")
    else:
        print_colored("  WARNING: No GPU! This config requires a powerful GPU.", "red")

    print(f"\n  Model    : {model_name} XLarge")
    print(f"  Variant  : {variant}")
    print(f"  Classes  : {DATASET_CONFIG['num_classes']}")
    print(f"\n  Optimizations:")
    print(f"    - Image size: 1280 (high resolution)")
    print(f"    - XLarge model (~57M params)")
    print(f"    - 200 epochs, patience=50")
    print(f"    - Strong augmentation (mixup, copy-paste, rotation)")
    print(f"    - Close mosaic at epoch 175")
    print(f"    - TTA (Test Time Augmentation) at evaluation")
    print(f"\n  Strategy:")
    print(f"    Train : ALL 5 countries combined")
    print(f"    Val   : ALL 5 countries combined")
    print(f"    Test  : EACH country separately (5 evaluations)")

    default_epochs = 200
    epochs_input = input(f"\n  Number of epochs [{default_epochs}]: ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else default_epochs

    default_batch = 16
    default_batch_label = "auto" if torch.cuda.is_available() else str(default_batch)
    batch_input = input(f"  Batch size [{default_batch_label}]: ").strip()
    batch_size = _parse_batch_input(batch_input, default_batch)

    print_colored(f"\n  Ready to start {model_name} XLarge Optimized (Per-Country Test)?", "yellow")
    print(f"  Epochs: {epochs}, Batch: {_format_batch_display(batch_size)}, ImgSz: 1280")
    confirm = input("  Type 'yes' to start: ").strip().lower()

    if confirm != "yes":
        print("  Training cancelled.")
        input("\nPress Enter to continue...")
        return

    result = train_model_percountry(
        model_key=model_key,
        size="x",
        epochs=epochs,
        batch_size=batch_size,
        optimized=True,
    )

    if result:
        print_colored(f"\n  Optimized per-country training & evaluation complete!", "green")
        print(f"  Results: {result['results_dir']}")
    else:
        print_colored(f"\n  Training failed. Check errors above.", "red")

    input("\nPress Enter to continue...")

def individual_country_flow(model_key: str, size: str):
    model_name = model_key.upper()
    size_label = {"l": "Large", "x": "XLarge"}[size]

    print_header()
    print_colored(f"\n  {model_name} {size_label} - INDIVIDUAL COUNTRY TRAINING", "bold")
    print("-" * 50)

    individual_dir = Path("data") / "individual"
    if not individual_dir.exists() or not any(individual_dir.iterdir()):
        print_colored("\n  Per-country individual splits not found!", "red")
        print("  Run: python setup_dataset.py -> option 2")
        input("\nPress Enter to continue...")
        return

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_colored(f"  GPU: {gpu_name} ({vram:.0f} GB VRAM)", "green")
    else:
        print_colored("  WARNING: No GPU! Training will be very slow.", "yellow")

    print(f"\n  Model    : {model_name} {size_label}")
    print(f"  Variant  : {model_key}{size}")
    print(f"  Classes  : {DATASET_CONFIG['num_classes']}")
    print(f"\n  Strategy:")
    print(f"    Each country trained SEPARATELY (own 80/10/10 split)")
    print(f"    Epochs   : unlimited (patience-based stop)")
    print(f"    Patience : 50 (stop after 50 epochs w/o improvement)")

    default_batch = 16
    default_batch_label = "auto" if torch.cuda.is_available() else str(default_batch)
    batch_input = input(f"\n  Batch size [{default_batch_label}]: ").strip()
    batch_size = _parse_batch_input(batch_input, default_batch)

    print_colored(f"\n  Ready to start {model_name} {size_label} Individual Country Training?", "yellow")
    print(f"  Batch: {_format_batch_display(batch_size)}, Patience: 50")
    print(f"  This will train 5 separate models (one per country).")
    confirm = input("  Type 'yes' to start: ").strip().lower()

    if confirm != "yes":
        print("  Training cancelled.")
        input("\nPress Enter to continue...")
        return

    result = train_model_individual_country(
        model_key=model_key,
        size=size,
        batch_size=batch_size,
    )

    if result:
        print_colored(f"\n  Individual country training complete!", "green")
        print(f"  Results: {result['results_dir']}")
    else:
        print_colored(f"\n  Training failed. Check errors above.", "red")

    input("\nPress Enter to continue...")

def svrdd_training_flow(model_key: str, size: str = "m"):
    model_name = model_key.upper()
    size_label = {"m": "Medium", "l": "Large"}[size]

    print_header()
    print_colored(f"\n  {model_name} {size_label} - SVRDD DATASET TRAINING", "bold")
    print("-" * 50)

    from config import SVRDD_DIR
    svrdd_yaml = SVRDD_DIR / "data.yaml"
    if not svrdd_yaml.exists():
        print_colored("\n  SVRDD dataset not found!", "red")
        print(f"  Expected at: {SVRDD_DIR}")
        print("  Download SVRDD_YOLO.zip from Zenodo and extract to SVRDD_dataset/")
        input("\nPress Enter to continue...")
        return

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_colored(f"  GPU: {gpu_name} ({vram:.0f} GB VRAM)", "green")
    else:
        print_colored("  WARNING: No GPU! Training will be very slow.", "yellow")

    print(f"\n  Model    : {model_name} {size_label}")
    print(f"  Dataset  : SVRDD (8,000 images, 7 classes)")
    print(f"  Classes  : 7 (cracks, pothole, patches, manhole)")
    print(f"\n  Strategy:")
    print(f"    Train : 6,000 images")
    print(f"    Val   : 1,000 images")
    print(f"    Test  : 1,000 images")
    print(f"    Epochs: unlimited (patience=50 stops training)")

    default_batch = 16
    default_batch_label = "auto" if torch.cuda.is_available() else str(default_batch)
    batch_input = input(f"\n  Batch size [{default_batch_label}]: ").strip()
    batch_size = _parse_batch_input(batch_input, default_batch)

    print_colored(f"\n  Ready to start {model_name} {size_label} on SVRDD?", "yellow")
    print(f"  Batch: {_format_batch_display(batch_size)}, Patience: 50")
    confirm = input("  Type 'yes' to start: ").strip().lower()

    if confirm != "yes":
        print("  Training cancelled.")
        input("\nPress Enter to continue...")
        return

    result = train_model_svrdd(
        model_key=model_key,
        size=size,
        batch_size=batch_size,
    )

    if result:
        print_colored(f"\n  SVRDD training complete!", "green")
        print(f"  Results: {result['results_dir']}")
    else:
        print_colored(f"\n  Training failed. Check errors above.", "red")

    input("\nPress Enter to continue...")

def svrdd_ablation_flow(ablation_key: str, description: str):
    print_header()
    print_colored(f"\n  YOLO12-L ABLATION: {description}", "bold")
    print("-" * 50)

    from config import SVRDD_DIR
    svrdd_yaml = SVRDD_DIR / "data.yaml"
    if not svrdd_yaml.exists():
        print_colored("\n  SVRDD dataset not found!", "red")
        print(f"  Expected at: {SVRDD_DIR}")
        input("\nPress Enter to continue...")
        return

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_colored(f"  GPU: {gpu_name} ({vram:.0f} GB VRAM)", "green")
    else:
        print_colored("  WARNING: No GPU! Training will be very slow.", "yellow")

    print(f"\n  Ablation variant : {ablation_key}")
    print(f"  Modification     : {description}")
    print(f"  Base model       : YOLO12-L (pretrained weights transferred)")
    print(f"  Dataset          : SVRDD (8,000 images, 7 classes)")
    print(f"  Strategy         : patience=50 (same as baseline)")

    default_batch_label = "auto" if torch.cuda.is_available() else "16"
    batch_input = input(f"\n  Batch size [{default_batch_label}]: ").strip()
    batch_size = _parse_batch_input(batch_input, 16)

    print_colored(f"\n  Ready to start ablation training?", "yellow")
    print(f"  Batch: {_format_batch_display(batch_size)}, Patience: 50")
    confirm = input("  Type 'yes' to start: ").strip().lower()

    if confirm != "yes":
        print("  Training cancelled.")
        input("\nPress Enter to continue...")
        return

    result = train_model_svrdd(
        model_key="yolo12",
        size="l",
        batch_size=batch_size,
        ablation_key=ablation_key,
    )

    if result:
        print_colored(f"\n  Ablation training complete!", "green")
        print(f"  Results: {result['results_dir']}")
    else:
        print_colored(f"\n  Training failed. Check errors above.", "red")

    input("\nPress Enter to continue...")

def svrdd_variant2_to_variant5_flow():
    print_header()
    print_colored("\n  YOLO12-L TWO-STAGE TRANSFER: VARIANT 2 -> VARIANT 5", "bold")
    print("-" * 50)

    from config import SVRDD_DIR
    svrdd_yaml = SVRDD_DIR / "data.yaml"
    if not svrdd_yaml.exists():
        print_colored("\n  SVRDD dataset not found!", "red")
        print(f"  Expected at: {SVRDD_DIR}")
        input("\nPress Enter to continue...")
        return

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_colored(f"  GPU: {gpu_name} ({vram:.0f} GB VRAM)", "green")
    else:
        print_colored("  WARNING: No GPU! Training will be very slow.", "yellow")

    print(f"\n  Dataset          : SVRDD (8,000 images, 7 classes)")
    print(f"  Stage 1 (Variant 2): YOLO12-L P2 Head")
    print(f"  Stage 2 (Variant 5): YOLO12-L P2 Head + Wider Neck")
    print(f"  Transfer rule      : Stage 2 starts from Stage 1 best.pt")
    print(f"  Resume behavior    : YES (both stages resume after interruption)")

    default_batch_label = "auto" if torch.cuda.is_available() else "16"
    batch_input = input(f"\n  Batch size [{default_batch_label}]: ").strip()
    batch_size = _parse_batch_input(batch_input, 16)

    print_colored("\n  Ready to start two-stage transfer training?", "yellow")
    print(f"  Batch: {_format_batch_display(batch_size)}, Patience: 50")
    confirm = input("  Type 'yes' to start: ").strip().lower()

    if confirm != "yes":
        print("  Training cancelled.")
        input("\nPress Enter to continue...")
        return

    result = train_model_svrdd_variant2_to_variant5(batch_size=batch_size)

    if result:
        print_colored(f"\n  Menu 27 pipeline complete!", "green")
        print(f"  Stage 1 results: {result['stage1']['results_dir']}")
        print(f"  Stage 2 results: {result['stage2']['results_dir']}")
    else:
        print_colored(f"\n  Pipeline failed. Check errors above.", "red")

    input("\nPress Enter to continue...")

def rdd2022_india_ablation_flow(ablation_key: str, description: str):
    print_header()
    print_colored(f"\n  YOLO12-L INDIA-ONLY ABLATION: {description}", "bold")
    print("-" * 50)

    if not check_dataset():
        print_colored("\n  Dataset not found!", "red")
        print("  Please run: python setup_dataset.py")
        input("\nPress Enter to continue...")
        return

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_colored(f"  GPU: {gpu_name} ({vram:.0f} GB VRAM)", "green")
    else:
        print_colored("  WARNING: No GPU! Training will be very slow.", "yellow")

    print(f"\n  Dataset          : RDD2022 (India-only split)")
    print(f"  Ablation variant : {ablation_key}")
    print(f"  Modification     : {description}")
    print(f"  Train/Val/Test   : India images only")

    default_epochs = 100
    epochs_input = input(f"\n  Number of epochs [{default_epochs}]: ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else default_epochs

    default_batch_label = "auto" if torch.cuda.is_available() else "16"
    batch_input = input(f"  Batch size [{default_batch_label}]: ").strip()
    batch_size = _parse_batch_input(batch_input, 16)

    print_colored(f"\n  Ready to start India-only ablation training?", "yellow")
    print(f"  Epochs: {epochs}, Batch: {_format_batch_display(batch_size)}")
    confirm = input("  Type 'yes' to start: ").strip().lower()

    if confirm != "yes":
        print("  Training cancelled.")
        input("\nPress Enter to continue...")
        return

    result = train_model_rdd2022_india(
        model_key="yolo12",
        size="l",
        epochs=epochs,
        batch_size=batch_size,
        ablation_key=ablation_key,
    )

    if result:
        print_colored(f"\n  India-only ablation training complete!", "green")
        print(f"  Results: {result['results_dir']}")
    else:
        print_colored(f"\n  Training failed. Check errors above.", "red")

    input("\nPress Enter to continue...")

def rdd2022_india_variant2_to_variant5_flow():
    print_header()
    print_colored("\n  YOLO12-L INDIA-ONLY TWO-STAGE: VARIANT 2 -> VARIANT 5", "bold")
    print("-" * 50)

    if not check_dataset():
        print_colored("\n  Dataset not found!", "red")
        print("  Please run: python setup_dataset.py")
        input("\nPress Enter to continue...")
        return

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_colored(f"  GPU: {gpu_name} ({vram:.0f} GB VRAM)", "green")
    else:
        print_colored("  WARNING: No GPU! Training will be very slow.", "yellow")

    print(f"\n  Dataset            : RDD2022 (India-only split)")
    print(f"  Stage 1 (Variant2) : YOLO12-L P2 Head")
    print(f"  Stage 2 (Variant5) : YOLO12-L P2 Head + Wider Neck")
    print(f"  Transfer rule      : Stage 2 starts from Stage 1 best.pt")
    print(f"  Train/Val/Test     : India images only")

    default_epochs = 100
    epochs_input = input(f"\n  Number of epochs per stage [{default_epochs}]: ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else default_epochs

    default_batch_label = "auto" if torch.cuda.is_available() else "16"
    batch_input = input(f"  Batch size [{default_batch_label}]: ").strip()
    batch_size = _parse_batch_input(batch_input, 16)

    print_colored("\n  Ready to start two-stage India-only training?", "yellow")
    print(f"  Epochs/stage: {epochs}, Batch: {_format_batch_display(batch_size)}")
    confirm = input("  Type 'yes' to start: ").strip().lower()

    if confirm != "yes":
        print("  Training cancelled.")
        input("\nPress Enter to continue...")
        return

    result = train_model_rdd2022_india_variant2_to_variant5(
        epochs=epochs,
        batch_size=batch_size,
    )

    if result:
        print_colored(f"\n  India-only two-stage pipeline complete!", "green")
        print(f"  Stage 1 results: {result['stage1']['results_dir']}")
        print(f"  Stage 2 results: {result['stage2']['results_dir']}")
    else:
        print_colored(f"\n  Pipeline failed. Check errors above.", "red")

    input("\nPress Enter to continue...")

def svrdd_post_learning_cv_flow(post_variant_key: str, variant_title: str):
    print_header()
    print_colored(f"\n  SVRDD POST-LEARNING + CV: {variant_title}", "bold")
    print("-" * 50)

    extra_dir = PROJECT_ROOT / "extraImagesForSVRDDPostLearning"
    if not extra_dir.exists():
        print_colored("\n  Folder not found!", "red")
        print(f"  Expected: {extra_dir}")
        input("\nPress Enter to continue...")
        return

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_colored(f"  GPU: {gpu_name} ({vram:.0f} GB VRAM)", "green")
    else:
        print_colored("  WARNING: No GPU! Fine-tuning will be slower.", "yellow")

    print(f"\n  Mode             : Post-learning (continue from existing checkpoint)")
    print(f"  Extra images dir : {extra_dir}")
    print(f"  Resolution filter: 1024x1024 (teacher request: trier par resolution)")
    print(f"  Validation       : K-fold cross-validation")

    default_folds = 5
    folds_input = input(f"\n  Number of folds [{default_folds}]: ").strip()
    k_folds = int(folds_input) if folds_input.isdigit() and int(folds_input) >= 2 else default_folds

    default_epochs = 30
    epochs_input = input(f"  Epochs per fold [{default_epochs}]: ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() and int(epochs_input) > 0 else default_epochs

    default_patience = 12
    patience_input = input(f"  Patience [{default_patience}]: ").strip()
    patience = int(patience_input) if patience_input.isdigit() and int(patience_input) > 0 else default_patience

    default_batch_label = "auto" if torch.cuda.is_available() else "16"
    batch_input = input(f"  Batch size [{default_batch_label}]: ").strip()
    batch_size = _parse_batch_input(batch_input, 16)

    print_colored("\n  Ready to start post-learning CV?", "yellow")
    print(f"  Folds: {k_folds}, Epochs/fold: {epochs}, Patience: {patience}, Batch: {_format_batch_display(batch_size)}")
    confirm = input("  Type 'yes' to start: ").strip().lower()

    if confirm != "yes":
        print("  Training cancelled.")
        input("\nPress Enter to continue...")
        return

    result = train_model_svrdd_post_learning_cv(
        post_variant_key=post_variant_key,
        epochs=epochs,
        batch_size=batch_size,
        k_folds=k_folds,
        patience=patience,
        resolution=(1024, 1024),
    )

    if result:
        print_colored("\n  Post-learning CV complete!", "green")
        print(f"  Results: {result['results_dir']}")
    else:
        print_colored("\n  Post-learning CV failed. Check errors above.", "red")

    input("\nPress Enter to continue...")

def dataset_info_flow():
    print_header()
    show_dataset_info()
    input("\nPress Enter to continue...")

def main():
    while True:
        print_header()
        print_system_info()
        print_menu()

        choice = input("\n  Enter your choice: ").strip()

        if choice == "0":
            print_colored("\n  Goodbye! Good luck with your project!\n", "green")
            break
        elif choice == "1":
            training_flow("yolo11", improved=False)
        elif choice == "2":
            training_flow("yolo12", improved=False)
        elif choice == "3":
            training_flow("yolo11", improved=True)
        elif choice == "4":
            training_flow("yolo12", improved=True)
        elif choice == "5":
            dataset_info_flow()
        elif choice == "6":
            percountry_training_flow("yolo11", "m")
        elif choice == "7":
            percountry_training_flow("yolo11", "l")
        elif choice == "8":
            percountry_training_flow("yolo12", "m")
        elif choice == "9":
            percountry_training_flow("yolo12", "l")
        elif choice.lower() == "a":
            optimized_percountry_flow("yolo11")
        elif choice.lower() == "b":
            optimized_percountry_flow("yolo12")
        elif choice == "10":
            individual_country_flow("yolo11", "l")
        elif choice == "11":
            individual_country_flow("yolo11", "x")
        elif choice == "12":
            individual_country_flow("yolo12", "l")
        elif choice == "13":
            individual_country_flow("yolo12", "x")
        elif choice == "15":
            svrdd_training_flow("yolo11", "m")
        elif choice == "16":
            svrdd_training_flow("yolo12", "m")
        elif choice == "17":
            svrdd_training_flow("yolo11", "l")
        elif choice == "18":
            svrdd_training_flow("yolo12", "l")
        elif choice == "19":
            svrdd_ablation_flow("yolo12l-deepA2C2f", "Deep A2C2f (backbone 4->6 blocks)")
        elif choice == "20":
            svrdd_ablation_flow("yolo12l-p2head", "P2 Head (4th detection head at stride 4)")
        elif choice == "21":
            svrdd_ablation_flow("yolo12l-p2head-deepA2C2f", "P2 Head + Deep A2C2f (combined)")
        elif choice == "24":
            svrdd_ablation_flow("yolo12l-widerNeck", "Wider Neck (FPN channels 256->384, 512->768)")
        elif choice == "25":
            svrdd_ablation_flow("yolo12l-p2head-widerNeck", "P2 Head + Wider Neck (combined)")
        elif choice == "26":
            percountry_training_flow(
                "yolo12",
                "l",
                variant_key_override="yolo12l-p2head",
                variant_description="P2 Head (4th detection head at stride 4)",
            )
        elif choice == "27":
            svrdd_variant2_to_variant5_flow()
        elif choice == "28":
            rdd2022_india_ablation_flow(
                "yolo12l-p2head-widerNeck",
                "P2 Head + Wider Neck (India-only)",
            )
        elif choice == "29":
            rdd2022_india_variant2_to_variant5_flow()
        elif choice == "30":
            svrdd_post_learning_cv_flow("baseline", "Baseline (YOLO12-L) + Extra Images")
        elif choice == "31":
            svrdd_post_learning_cv_flow("v2", "Variant 2 (P2 Head) + Extra Images")
        elif choice == "32":
            svrdd_post_learning_cv_flow("v4", "Variant 4 (Wider Neck) + Extra Images")
        elif choice == "33":
            svrdd_post_learning_cv_flow("v5", "Variant 5 (P2 + Wider Neck) + Extra Images")
        elif choice == "34":
            svrdd_post_learning_cv_flow("v6", "Variant 6 (Two-stage Transfer) + Extra Images")
        else:
            print_colored("  Invalid choice! Please try again.", "red")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n\n  Interrupted. Goodbye!\n", "yellow")
        sys.exit(0)

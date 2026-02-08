import os
import sys
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, MENU_CONFIG

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_colored(text: str, color: str = "end"):
    colors = MENU_CONFIG["colors"]
    color_code = colors.get(color, colors["end"])
    print(f"{color_code}{text}{colors['end']}")

def check_gpu() -> bool:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_colored(f"  GPU detected: {gpu_name} ({gpu_memory:.1f} GB)", "green")
        return True
    else:
        print_colored("  No GPU detected! Training will be slow on CPU.", "yellow")
        return False

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def check_dataset() -> bool:
    data_yaml = DATA_DIR / "data.yaml"
    train_images = DATA_DIR / "train" / "images"

    if not data_yaml.exists():
        return False

    if train_images.exists():
        imgs = list(train_images.glob("*"))
        if len(imgs) > 0:
            return True

    return False

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

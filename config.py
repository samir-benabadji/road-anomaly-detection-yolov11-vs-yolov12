import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR,
                 MODELS_DIR / "yolo11", MODELS_DIR / "yolo12"]:
    dir_path.mkdir(parents=True, exist_ok=True)

DATASET_CONFIG = {
    "name": "RDD2022_road_anomaly",
    "countries": ["Norway", "Japan", "India", "Czech", "United_States"],
    "classes": ["D00_Longitudinal_Crack", "D10_Transverse_Crack",
                "D20_Alligator_Crack", "D40_Pothole"],
    "class_mapping": {"D00": 0, "D10": 1, "D20": 2, "D40": 3},
    "num_classes": 4,
    "image_size": 640,
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15,
}

TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "image_size": 640,
    "patience": 30,
    "workers": 4,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": False,
    "warmup_epochs": 3,
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "mosaic": 1.0,
    "mixup": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "flipud": 0.0,
    "fliplr": 0.5,
}

IMPROVED_TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "image_size": 640,
    "patience": 30,
    "workers": 4,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.1,
    "cos_lr": True,
    "warmup_epochs": 5,
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "freeze": 10,
    "mosaic": 1.0,
    "mixup": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "flipud": 0.0,
    "fliplr": 0.5,
}

YOLO11_CONFIG = {
    "model_variant": "yolo11s",
    "pretrained": True,
}

YOLO11_MEDIUM_CONFIG = {
    "model_variant": "yolo11m",
    "pretrained": True,
}

YOLO11_LARGE_CONFIG = {
    "model_variant": "yolo11l",
    "pretrained": True,
}

YOLO12_CONFIG = {
    "model_variant": "yolo12s",
    "pretrained": True,
}

YOLO12_MEDIUM_CONFIG = {
    "model_variant": "yolo12m",
    "pretrained": True,
}

YOLO12_LARGE_CONFIG = {
    "model_variant": "yolo12l",
    "pretrained": True,
}

YOLO12_LARGE_DEEP_A2C2F_CONFIG = {
    "model_variant": "yolo12l",
    "pretrained": True,
    "custom_yaml": "models/yolo12l-deepA2C2f.yaml",
    "ablation_tag": "DeepA2C2f",
}

YOLO12_LARGE_P2HEAD_CONFIG = {
    "model_variant": "yolo12l",
    "pretrained": True,
    "custom_yaml": "models/yolo12l-p2head.yaml",
    "ablation_tag": "P2Head",
}

YOLO12_LARGE_P2HEAD_DEEP_A2C2F_CONFIG = {
    "model_variant": "yolo12l",
    "pretrained": True,
    "custom_yaml": "models/yolo12l-p2head-deepA2C2f.yaml",
    "ablation_tag": "P2Head_DeepA2C2f",
}

YOLO12_LARGE_WIDER_NECK_CONFIG = {
    "model_variant": "yolo12l",
    "pretrained": True,
    "custom_yaml": "models/yolo12l-widerNeck.yaml",
    "ablation_tag": "WiderNeck",
}

YOLO12_LARGE_P2HEAD_WIDER_NECK_CONFIG = {
    "model_variant": "yolo12l",
    "pretrained": True,
    "custom_yaml": "models/yolo12l-p2head-widerNeck.yaml",
    "ablation_tag": "P2Head_WiderNeck",
}

YOLO11_XLARGE_CONFIG = {
    "model_variant": "yolo11x",
    "pretrained": True,
}

YOLO12_XLARGE_CONFIG = {
    "model_variant": "yolo12x",
    "pretrained": True,
}

OPTIMIZED_TRAINING_CONFIG = {
    "epochs": 200,
    "batch_size": 16,
    "image_size": 1280,
    "patience": 50,
    "workers": 4,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": False,
    "warmup_epochs": 5,
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "mosaic": 1.0,
    "close_mosaic": 25,
    "mixup": 0.15,
    "copy_paste": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.9,
    "flipud": 0.0,
    "fliplr": 0.5,
    "tta": True,
}

INDIVIDUAL_TRAINING_CONFIG = {
    "epochs": 9999,
    "batch_size": 16,
    "image_size": 640,
    "patience": 50,
    "workers": 4,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": False,
    "warmup_epochs": 3,
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "mosaic": 1.0,
    "mixup": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "flipud": 0.0,
    "fliplr": 0.5,
}

SVRDD_DIR = PROJECT_ROOT / "SVRDD_dataset"

SVRDD_DATASET_CONFIG = {
    "name": "SVRDD_road_damage",
    "classes": ["Longitudinal_Crack", "Transverse_Crack", "Alligator_Crack",
                "Pothole", "Longitudinal_Patch", "Transverse_Patch",
                "Manhole_Cover"],
    "num_classes": 7,
    "image_size": 640,
}

SVRDD_TRAINING_CONFIG = {
    "epochs": 9999,
    "batch_size": 16,
    "image_size": 640,
    "patience": 50,
    "workers": 4,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "cos_lr": False,
    "warmup_epochs": 3,
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "mosaic": 1.0,
    "mixup": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "flipud": 0.0,
    "fliplr": 0.5,
}

GPU_CONFIG = {
    "device": "0",
    "amp": True,
    "cache": "disk",
    "auto_batch": True,
    "workers": "auto",
    "max_workers": 16,
    "deterministic": False,
    "compile": True,
}

EVALUATION_CONFIG = {
    "iou_threshold": 0.5,
    "conf_threshold": 0.25,
    "metrics": [
        "mAP50", "mAP50-95", "precision", "recall",
        "f1_score", "inference_time",
    ]
}

MENU_CONFIG = {
    "title": """
=====================================================================
     ROAD ANOMALY DETECTION - YOLO11 vs YOLO12 COMPARISON
                 Projet Fin d'Etude - Master 2
=====================================================================""",
    "colors": {
        "header": "\033[95m",
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "end": "\033[0m",
        "bold": "\033[1m",
        "cyan": "\033[96m",
    }
}

def get_results_dir(model_name: str, mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{model_name}_{mode}_{timestamp}"
    results_path = RESULTS_DIR / folder_name
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path

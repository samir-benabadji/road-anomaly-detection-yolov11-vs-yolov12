import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
    TRAINING_CONFIG, YOLOV8_CONFIG, GPU_CONFIG, get_results_dir
)
from src.utils import print_colored, check_gpu, get_timestamp

def train_yolov8(epochs: int = None, batch_size: int = None,
                 model_variant: str = None, resume: bool = False):
    print_colored("\n" + "="*60, "blue")
    print_colored("🚀 STARTING YOLOv8 TRAINING", "bold")
    print_colored("="*60, "blue")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print_colored("❌ ultralytics not installed. Installing...", "yellow")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO
    
    epochs = epochs or TRAINING_CONFIG["epochs"]
    batch_size = batch_size or TRAINING_CONFIG["batch_size"]
    model_variant = model_variant or YOLOV8_CONFIG["model_variant"]
    
    results_dir = get_results_dir()
    run_name = f"yolov8_{model_variant}_{get_timestamp()}"
    
    data_yaml = DATA_DIR / "data.yaml"
    
    if not data_yaml.exists():
        print_colored("❌ data.yaml not found! Please setup dataset first.", "red")
        return None
    
    device = "0" if torch.cuda.is_available() else "cpu"
    
    print_colored("\n📋 Training Configuration:", "bold")
    print(f"  • Model: {model_variant}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Batch Size: {batch_size}")
    print(f"  • Image Size: {TRAINING_CONFIG['image_size']}")
    print(f"  • Device: {'GPU' if device == '0' else 'CPU'}")
    print(f"  • Data: {data_yaml}")
    print(f"  • Results: {results_dir}")
    
    try:
        print_colored(f"\n📥 Loading pretrained {model_variant}...", "blue")
        model = YOLO(f"{model_variant}.pt")
        
        print_colored("\n🏃 Starting training...\n", "green")
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=TRAINING_CONFIG["image_size"],
            device=device,
            project=str(results_dir),
            name=run_name,
            exist_ok=True,
            patience=TRAINING_CONFIG["patience"],
            workers=TRAINING_CONFIG["workers"],
            optimizer=TRAINING_CONFIG["optimizer"],
            lr0=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            momentum=TRAINING_CONFIG["momentum"],
            mosaic=TRAINING_CONFIG["mosaic"],
            mixup=TRAINING_CONFIG["mixup"],
            hsv_h=TRAINING_CONFIG["hsv_h"],
            hsv_s=TRAINING_CONFIG["hsv_s"],
            hsv_v=TRAINING_CONFIG["hsv_v"],
            degrees=TRAINING_CONFIG["degrees"],
            translate=TRAINING_CONFIG["translate"],
            scale=TRAINING_CONFIG["scale"],
            shear=TRAINING_CONFIG["shear"],
            flipud=TRAINING_CONFIG["flipud"],
            fliplr=TRAINING_CONFIG["fliplr"],
            amp=GPU_CONFIG["amp"],
            cache=GPU_CONFIG["cache_images"],
            verbose=True,
        )
        
        print_colored("\n✅ Training completed successfully!", "green")
        
        run_path = results_dir / run_name
        best_weights = run_path / "weights" / "best.pt"
        
        if best_weights.exists():
            models_yolov8_dir = MODELS_DIR / "yolov8"
            models_yolov8_dir.mkdir(parents=True, exist_ok=True)
            
            final_weights = models_yolov8_dir / f"{run_name}_best.pt"
            shutil.copy(best_weights, final_weights)
            
            print_colored(f"\n📁 Model saved to: {final_weights}", "green")
            
            copy_training_results(run_path, results_dir, "yolov8")
            
            return final_weights
            
    except Exception as e:
        print_colored(f"\n❌ Training error: {e}", "red")
        import traceback
        traceback.print_exc()
    
    return None

def copy_training_results(run_path: Path, results_dir: Path, model_name: str):
    print_colored("\n📊 Saving training results...", "blue")
    
    graphs_dir = results_dir / "graphs"
    metrics_dir = results_dir / "metrics"
    
    result_files = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]
    
    for file in result_files:
        src = run_path / file
        if src.exists():
            dst = graphs_dir / f"{model_name}_{file}"
            shutil.copy(src, dst)
            print(f"  Saved: {dst.name}")
    
    results_csv = run_path / "results.csv"
    if results_csv.exists():
        dst = metrics_dir / f"{model_name}_training_results.csv"
        shutil.copy(results_csv, dst)
        print(f"  Saved: {dst.name}")

def get_yolov8_model_info():
    return {
        "name": "YOLOv8",
        "version": "8.0",
        "backbone": "Modified CSPDarknet with C2f modules",
        "neck": "PANet with C2f modules",
        "head": "Decoupled Anchor-Free Head",
        "key_features": [
            "C2f module (improved C3 from YOLOv5)",
            "Anchor-free detection (no anchor boxes)",
            "Decoupled head (separate cls/reg branches)",
            "TaskAlignedAssigner for label assignment",
            "Distribution Focal Loss (DFL)",
            "CIoU loss for bounding boxes",
        ],
        "changes_from_yolov5": [
            "Replaced Focus layer with standard Conv",
            "Replaced C3 modules with C2f",
            "Removed anchor boxes (anchor-free)",
            "Added decoupled head",
            "New loss function combination",
            "Improved label assignment strategy",
        ],
        "variants": {
            "yolov8n": {"params": "3.2M", "flops": "8.7G", "map": "37.3"},
            "yolov8s": {"params": "11.2M", "flops": "28.6G", "map": "44.9"},
            "yolov8m": {"params": "25.9M", "flops": "78.9G", "map": "50.2"},
            "yolov8l": {"params": "43.7M", "flops": "165.2G", "map": "52.9"},
            "yolov8x": {"params": "68.2M", "flops": "257.8G", "map": "53.9"},
        },
        "architecture_details": """
YOLOv8 Architecture:
====================

1. BACKBONE (Modified CSPDarknet):
   - Input: 640x640x3
   - Standard Conv (replaces Focus from YOLOv5)
   - C2f blocks (Cross Stage Partial with 2 convs + flow)
   - SPPF (Spatial Pyramid Pooling - Fast)
   - Output: Multi-scale feature maps

2. NECK (PANet with C2f):
   - Feature Pyramid Network for top-down fusion
   - Path Aggregation for bottom-up fusion
   - C2f modules throughout for better gradient flow

3. HEAD (Decoupled Anchor-Free):
   - Anchor-free: predicts center offset directly
   - Decoupled branches:
     * Classification branch (BCE loss)
     * Regression branch (DFL + CIoU loss)
   - Three output scales (P3, P4, P5)

Key Architecture Changes from YOLOv5:
=====================================
┌─────────────────────────────────────────────────────────┐
│  Component      │  YOLOv5        │  YOLOv8              │
├─────────────────────────────────────────────────────────┤
│  First Layer    │  Focus         │  Conv 3x3            │
│  CSP Block      │  C3            │  C2f                 │
│  Detection      │  Anchor-based  │  Anchor-free         │
│  Head           │  Coupled       │  Decoupled           │
│  Loss (box)     │  CIoU          │  CIoU + DFL          │
│  Assignment     │  SimOTA        │  TaskAlignedAssigner │
└─────────────────────────────────────────────────────────┘
"""
    }

if __name__ == "__main__":
    train_yolov8(epochs=10, batch_size=16)

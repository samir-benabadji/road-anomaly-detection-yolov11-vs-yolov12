import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
    TRAINING_CONFIG, YOLOV5_CONFIG, GPU_CONFIG, get_results_dir
)
from src.utils import print_colored, check_gpu, get_timestamp

def setup_yolov5():
    yolov5_path = PROJECT_ROOT / "yolov5"
    
    if not yolov5_path.exists():
        print_colored("\n📥 Cloning YOLOv5 repository...", "blue")
        
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/ultralytics/yolov5.git",
                str(yolov5_path)
            ], check=True)
            
            print_colored("📦 Installing YOLOv5 requirements...", "blue")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r",
                str(yolov5_path / "requirements.txt")
            ], check=True)
            
            print_colored("✅ YOLOv5 setup complete!", "green")
            
        except subprocess.CalledProcessError as e:
            print_colored(f"❌ Error setting up YOLOv5: {e}", "red")
            return False
    
    return True

def train_yolov5(epochs: int = None, batch_size: int = None, 
                 model_variant: str = None, resume: bool = False):
    print_colored("\n" + "="*60, "blue")
    print_colored("🚀 STARTING YOLOv5 TRAINING", "bold")
    print_colored("="*60, "blue")
    
    if not setup_yolov5():
        return None
    
    epochs = epochs or TRAINING_CONFIG["epochs"]
    batch_size = batch_size or TRAINING_CONFIG["batch_size"]
    model_variant = model_variant or YOLOV5_CONFIG["model_variant"]
    
    results_dir = get_results_dir()
    run_name = f"yolov5_{model_variant}_{get_timestamp()}"
    
    yolov5_path = PROJECT_ROOT / "yolov5"
    data_yaml = DATA_DIR / "data.yaml"
    weights = f"{model_variant}.pt"
    
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
    
    train_script = yolov5_path / "train.py"
    
    cmd = [
        sys.executable, str(train_script),
        "--img", str(TRAINING_CONFIG["image_size"]),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", str(data_yaml),
        "--weights", weights,
        "--device", device,
        "--project", str(results_dir),
        "--name", run_name,
        "--exist-ok",
        "--patience", str(TRAINING_CONFIG["patience"]),
        "--workers", str(TRAINING_CONFIG["workers"]),
    ]
    
    if GPU_CONFIG["cache_images"]:
        cmd.append("--cache")
    
    if resume:
        cmd.append("--resume")
    
    print_colored("\n🏃 Starting training...\n", "green")
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(yolov5_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print_colored("\n✅ Training completed successfully!", "green")
            
            run_path = results_dir / run_name
            best_weights = run_path / "weights" / "best.pt"
            
            if best_weights.exists():
                models_yolov5_dir = MODELS_DIR / "yolov5"
                models_yolov5_dir.mkdir(parents=True, exist_ok=True)
                
                final_weights = models_yolov5_dir / f"{run_name}_best.pt"
                shutil.copy(best_weights, final_weights)
                
                print_colored(f"\n📁 Model saved to: {final_weights}", "green")
                
                copy_training_results(run_path, results_dir, "yolov5")
                
                return final_weights
        else:
            print_colored(f"\n❌ Training failed with return code: {process.returncode}", "red")
            
    except Exception as e:
        print_colored(f"\n❌ Training error: {e}", "red")
    
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

def get_yolov5_model_info():
    return {
        "name": "YOLOv5",
        "version": "7.0",
        "backbone": "CSPDarknet53",
        "neck": "PANet (Path Aggregation Network)",
        "head": "YOLOv5 Head (3 detection scales)",
        "key_features": [
            "Focus layer for efficient downsampling",
            "Cross Stage Partial (CSP) connections",
            "Spatial Pyramid Pooling (SPP) module",
            "Data augmentation (Mosaic, MixUp)",
            "Multi-scale training",
            "Auto-anchor calculation",
        ],
        "variants": {
            "yolov5n": {"params": "1.9M", "flops": "4.5G", "map": "28.0"},
            "yolov5s": {"params": "7.2M", "flops": "16.5G", "map": "37.4"},
            "yolov5m": {"params": "21.2M", "flops": "49.0G", "map": "45.4"},
            "yolov5l": {"params": "46.5M", "flops": "109.1G", "map": "49.0"},
            "yolov5x": {"params": "86.7M", "flops": "205.7G", "map": "50.7"},
        },
        "architecture_details": """
YOLOv5 Architecture:
====================

1. BACKBONE (CSPDarknet53):
   - Input: 640x640x3
   - Focus: Slices image into 4 parts (320x320x12)
   - CSP Bottleneck blocks with cross-stage connections
   - Output: Multi-scale feature maps (P3, P4, P5)

2. NECK (PANet):
   - Feature Pyramid Network (FPN) for top-down fusion
   - Path Aggregation for bottom-up augmentation
   - SPP module for multi-scale feature extraction

3. HEAD:
   - Three detection heads for different scales:
     * P3 (80x80): Small objects
     * P4 (40x40): Medium objects  
     * P5 (20x20): Large objects
   - Each predicts: x, y, w, h, objectness, class_probs
"""
    }

if __name__ == "__main__":
    train_yolov5(epochs=10, batch_size=16)

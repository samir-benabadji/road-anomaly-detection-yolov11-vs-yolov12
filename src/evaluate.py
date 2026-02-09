import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
    EVALUATION_CONFIG, DATASET_CONFIG, get_results_dir
)
from src.utils import print_colored, save_plot, save_metrics, get_timestamp

def evaluate_model(model_path: Path, model_type: str, 
                   results_dir: Optional[Path] = None) -> Dict:
    print_colored("\n" + "="*60, "blue")
    print_colored(f"📈 EVALUATING {model_type.upper()} MODEL", "bold")
    print_colored("="*60, "blue")
    
    if results_dir is None:
        results_dir = get_results_dir()
    
    print(f"\nModel: {model_path.name}")
    print(f"Results will be saved to: {results_dir}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if model_type == "yolov5":
        metrics = evaluate_yolov5(model_path, device, results_dir)
    else:
        metrics = evaluate_yolov8(model_path, device, results_dir)
    
    save_metrics(metrics, f"{model_type}_evaluation_{get_timestamp()}", results_dir)
    
    print_evaluation_summary(metrics, model_type)
    
    return metrics

def evaluate_yolov5(model_path: Path, device: str, results_dir: Path) -> Dict:
    import subprocess
    
    yolov5_path = PROJECT_ROOT / "yolov5"
    data_yaml = DATA_DIR / "data.yaml"
    
    if not yolov5_path.exists():
        print_colored("❌ YOLOv5 not found. Please train a YOLOv5 model first.", "red")
        return {}
    
    val_script = yolov5_path / "val.py"
    
    cmd = [
        sys.executable, str(val_script),
        "--weights", str(model_path),
        "--data", str(data_yaml),
        "--img", str(EVALUATION_CONFIG.get("image_size", 640)),
        "--device", device.replace("cuda:", ""),
        "--project", str(results_dir),
        "--name", f"yolov5_eval_{get_timestamp()}",
        "--save-txt",
        "--save-conf",
        "--verbose",
    ]
    
    print_colored("\n🔄 Running YOLOv5 validation...", "blue")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(yolov5_path),
            capture_output=True,
            text=True
        )
        
        metrics = parse_yolov5_output(result.stdout)
        
        metrics["inference_time_ms"] = measure_inference_time(model_path, "yolov5", device)
        
        metrics["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)
        
        return metrics
        
    except Exception as e:
        print_colored(f"❌ Evaluation error: {e}", "red")
        return {}

def evaluate_yolov8(model_path: Path, device: str, results_dir: Path) -> Dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        print_colored("❌ ultralytics not installed!", "red")
        return {}
    
    data_yaml = DATA_DIR / "data.yaml"
    
    print_colored("\n🔄 Running YOLOv8 validation...", "blue")
    
    try:
        model = YOLO(str(model_path))
        
        results = model.val(
            data=str(data_yaml),
            device=device,
            project=str(results_dir),
            name=f"yolov8_eval_{get_timestamp()}",
            save_txt=True,
            save_conf=True,
            verbose=True,
        )
        
        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "f1_score": 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6),
        }
        
        class_names = DATASET_CONFIG.get("classes", [])
        if hasattr(results.box, 'ap50') and len(class_names) > 0:
            for i, class_name in enumerate(class_names):
                if i < len(results.box.ap50):
                    metrics[f"AP50_{class_name}"] = float(results.box.ap50[i])
        
        metrics["inference_time_ms"] = measure_inference_time(model_path, "yolov8", device)
        
        metrics["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)
        
        return metrics
        
    except Exception as e:
        print_colored(f"❌ Evaluation error: {e}", "red")
        import traceback
        traceback.print_exc()
        return {}

def measure_inference_time(model_path: Path, model_type: str, device: str, 
                           num_runs: int = 100) -> float:
    print_colored("\n⏱️  Measuring inference time...", "blue")
    
    img_size = EVALUATION_CONFIG.get("image_size", 640)
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    if "cuda" in device:
        dummy_input = dummy_input.cuda()
    
    times = []
    
    if model_type == "yolov8":
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        
        for _ in range(10):
            model.predict(dummy_input, verbose=False)
        
        for _ in tqdm(range(num_runs), desc="Measuring inference"):
            if "cuda" in device:
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            model.predict(dummy_input, verbose=False)
            
            if "cuda" in device:
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
        
        if "cuda" in device:
            model = model.cuda()
        
        model.eval()
        
        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)
        
        for _ in tqdm(range(num_runs), desc="Measuring inference"):
            if "cuda" in device:
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                model(dummy_input)
            
            if "cuda" in device:
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    
    return avg_time

def parse_yolov5_output(output: str) -> Dict:
    metrics = {}
    
    lines = output.split('\n')
    for line in lines:
        if 'all' in line.lower() and 'images' in line.lower():
            parts = line.split()
            try:
                idx = parts.index('all') if 'all' in parts else 0
                metrics["precision"] = float(parts[idx + 3]) if len(parts) > idx + 3 else 0
                metrics["recall"] = float(parts[idx + 4]) if len(parts) > idx + 4 else 0
                metrics["mAP50"] = float(parts[idx + 5]) if len(parts) > idx + 5 else 0
                metrics["mAP50-95"] = float(parts[idx + 6]) if len(parts) > idx + 6 else 0
            except (ValueError, IndexError):
                pass
    
    if "precision" in metrics and "recall" in metrics:
        p, r = metrics["precision"], metrics["recall"]
        metrics["f1_score"] = 2 * (p * r) / (p + r + 1e-6)
    
    return metrics

def print_evaluation_summary(metrics: Dict, model_type: str):
    print_colored(f"\n📊 {model_type.upper()} EVALUATION SUMMARY", "bold")
    print("-" * 50)
    
    key_metrics = [
        ("mAP50", "mAP@0.5"),
        ("mAP50-95", "mAP@0.5:0.95"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 Score"),
        ("inference_time_ms", "Inference Time (ms)"),
        ("model_size_mb", "Model Size (MB)"),
    ]
    
    for key, display_name in key_metrics:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                if "time" in key.lower() or "size" in key.lower():
                    print(f"  {display_name}: {value:.2f}")
                else:
                    print(f"  {display_name}: {value:.4f}")
            else:
                print(f"  {display_name}: {value}")
    
    print("-" * 50)

def create_evaluation_plots(metrics: Dict, model_type: str, results_dir: Path):
    class_metrics = {k: v for k, v in metrics.items() if k.startswith("AP50_")}
    
    if class_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = [k.replace("AP50_", "") for k in class_metrics.keys()]
        values = list(class_metrics.values())
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))
        bars = ax.bar(classes, values, color=colors, edgecolor='black')
        
        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("AP@0.5", fontsize=12)
        ax.set_title(f"{model_type.upper()} - Per-Class Average Precision", fontsize=14)
        ax.set_ylim(0, 1)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_plot(fig, f"{model_type}_per_class_ap", results_dir)
        plt.close()

if __name__ == "__main__":
    print("Evaluation module loaded successfully!")

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
    DATASET_CONFIG, get_results_dir
)
from src.utils import print_colored, get_timestamp

def run_prediction(model_path: Path, image_source: Union[str, Path], 
                   model_type: str, conf_threshold: float = 0.25,
                   save_results: bool = True) -> List[Dict]:
    print_colored("\n" + "="*60, "blue")
    print_colored(f"🔍 RUNNING PREDICTIONS WITH {model_type.upper()}", "bold")
    print_colored("="*60, "blue")
    
    image_source = Path(image_source)
    
    if image_source.is_file():
        images = [image_source]
    elif image_source.is_dir():
        images = list(image_source.glob("*.jpg")) + \
                 list(image_source.glob("*.jpeg")) + \
                 list(image_source.glob("*.png"))
    else:
        print_colored(f"❌ Invalid image source: {image_source}", "red")
        return []
    
    print(f"\nFound {len(images)} image(s) to process")
    print(f"Model: {model_path.name}")
    print(f"Confidence threshold: {conf_threshold}")
    
    if save_results:
        results_dir = get_results_dir()
        predictions_dir = results_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
    
    if model_type == "yolov8":
        results = predict_yolov8(model_path, images, conf_threshold, 
                                 predictions_dir if save_results else None)
    else:
        results = predict_yolov5(model_path, images, conf_threshold,
                                 predictions_dir if save_results else None)
    
    print_prediction_summary(results)
    
    if save_results:
        print_colored(f"\n📁 Results saved to: {predictions_dir}", "green")
    
    return results

def predict_yolov8(model_path: Path, images: List[Path], 
                   conf_threshold: float, save_dir: Optional[Path]) -> List[Dict]:
    try:
        from ultralytics import YOLO
    except ImportError:
        print_colored("❌ ultralytics not installed!", "red")
        return []
    
    model = YOLO(str(model_path))
    
    results_list = []
    
    for img_path in images:
        print(f"\n📷 Processing: {img_path.name}")
        
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            save=save_dir is not None,
            project=str(save_dir) if save_dir else None,
            name="",
            exist_ok=True,
        )
        
        for r in results:
            detections = []
            for box in r.boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                }
                detections.append(detection)
                print(f"  ✓ Detected: {detection['class_name']} ({detection['confidence']:.2%})")
            
            results_list.append({
                "image": str(img_path),
                "detections": detections,
                "num_detections": len(detections)
            })
    
    return results_list

def predict_yolov5(model_path: Path, images: List[Path],
                   conf_threshold: float, save_dir: Optional[Path]) -> List[Dict]:
    import torch
    
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
    model.conf = conf_threshold
    
    results_list = []
    
    for img_path in images:
        print(f"\n📷 Processing: {img_path.name}")
        
        results = model(str(img_path))
        
        detections = []
        pred = results.pandas().xyxy[0]
        
        for _, row in pred.iterrows():
            detection = {
                "class_id": int(row['class']),
                "class_name": row['name'],
                "confidence": float(row['confidence']),
                "bbox": [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
            }
            detections.append(detection)
            print(f"  ✓ Detected: {detection['class_name']} ({detection['confidence']:.2%})")
        
        results_list.append({
            "image": str(img_path),
            "detections": detections,
            "num_detections": len(detections)
        })
        
        if save_dir:
            save_annotated_image(img_path, detections, save_dir)
    
    return results_list

def save_annotated_image(img_path: Path, detections: List[Dict], save_dir: Path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = (colors[:, :3] * 255).astype(int)
    
    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        conf = det["confidence"]
        class_id = det["class_id"] % len(colors)
        color = tuple(map(int, colors[class_id]))
        
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    save_path = save_dir / f"pred_{img_path.name}"
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), img_bgr)

def print_prediction_summary(results: List[Dict]):
    print_colored("\n" + "-"*50, "blue")
    print_colored("📊 PREDICTION SUMMARY", "bold")
    print("-"*50)
    
    total_images = len(results)
    total_detections = sum(r["num_detections"] for r in results)
    
    print(f"Total images processed: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/total_images:.1f}")
    
    class_counts = {}
    for r in results:
        for det in r["detections"]:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        print("\nDetections by class:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  • {class_name}: {count}")
    
    print("-"*50)

def visualize_detections(image_path: Union[str, Path], detections: List[Dict],
                         save_path: Optional[Path] = None, show: bool = True):
    img = Image.open(image_path)
    img_array = np.array(img)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_array)
    
    colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        conf = det["confidence"]
        class_id = det["class_id"] % len(colors)
        color = colors[class_id]
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        rect = plt.Rectangle((x1, y1), width, height,
                             fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        label = f"{class_name}: {conf:.2%}"
        ax.text(x1, y1 - 5, label, color='white', fontsize=10,
               bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1))
    
    ax.set_title(f"Detections: {len(detections)}", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def batch_predict(model_path: Path, input_dir: Path, output_dir: Path,
                  model_type: str, conf_threshold: float = 0.25):
    print_colored(f"\n🔄 Running batch prediction on {input_dir}", "blue")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in input_dir.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(images)} images")
    
    results = run_prediction(model_path, input_dir, model_type, 
                            conf_threshold, save_results=True)
    
    import json
    summary_path = output_dir / "batch_results.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_colored(f"\n✅ Batch prediction complete! Results in {output_dir}", "green")
    
    return results

if __name__ == "__main__":
    print("Prediction module loaded successfully!")

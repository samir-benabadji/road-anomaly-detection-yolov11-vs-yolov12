import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
    COMPARISON_CONFIG, DATASET_CONFIG, get_results_dir
)
from src.utils import print_colored, save_plot, save_metrics, get_timestamp
from src.evaluate import evaluate_model
from src.train_yolov5 import get_yolov5_model_info
from src.train_yolov8 import get_yolov8_model_info

def compare_models(yolov5_model_path: Path, yolov8_model_path: Path,
                   results_dir: Optional[Path] = None) -> Dict:
    print_colored("\n" + "="*70, "blue")
    print_colored("📊 COMPARATIVE ANALYSIS: YOLOv5 vs YOLOv8", "bold")
    print_colored("="*70, "blue")
    
    if results_dir is None:
        results_dir = get_results_dir()
    
    comparison_dir = results_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    print_colored("\n📈 Evaluating YOLOv5...", "blue")
    yolov5_metrics = evaluate_model(yolov5_model_path, "yolov5", results_dir)
    
    print_colored("\n📈 Evaluating YOLOv8...", "blue")
    yolov8_metrics = evaluate_model(yolov8_model_path, "yolov8", results_dir)
    
    comparison = {
        "yolov5": yolov5_metrics,
        "yolov8": yolov8_metrics,
        "timestamp": get_timestamp(),
        "yolov5_model": str(yolov5_model_path.name),
        "yolov8_model": str(yolov8_model_path.name),
    }
    
    print_colored("\n📊 Generating comparison visualizations...", "blue")
    
    create_metrics_comparison_bar(yolov5_metrics, yolov8_metrics, comparison_dir)
    create_radar_chart(yolov5_metrics, yolov8_metrics, comparison_dir)
    create_performance_table(yolov5_metrics, yolov8_metrics, comparison_dir)
    create_architecture_comparison(comparison_dir)
    create_tradeoff_analysis(yolov5_metrics, yolov8_metrics, comparison_dir)
    
    save_comparison_report(comparison, comparison_dir)
    
    print_comparison_summary(yolov5_metrics, yolov8_metrics)
    
    return comparison

def create_metrics_comparison_bar(v5_metrics: Dict, v8_metrics: Dict, 
                                   save_dir: Path):
    metrics_to_plot = ["mAP50", "mAP50-95", "precision", "recall", "f1_score"]
    
    v5_values = [v5_metrics.get(m, 0) for m in metrics_to_plot]
    v8_values = [v8_metrics.get(m, 0) for m in metrics_to_plot]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, v5_values, width, label='YOLOv5', 
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, v8_values, width, label='YOLOv8', 
                   color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('YOLOv5 vs YOLOv8: Detection Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1 Score'])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    for fmt in ["png", "pdf"]:
        fig.savefig(save_dir / f"metrics_comparison.{fmt}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_comparison.png/pdf")

def create_radar_chart(v5_metrics: Dict, v8_metrics: Dict, save_dir: Path):
    metrics = ["mAP50", "mAP50-95", "precision", "recall", "f1_score"]
    labels = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
    
    v5_values = [v5_metrics.get(m, 0) for m in metrics]
    v8_values = [v8_metrics.get(m, 0) for m in metrics]
    
    if "inference_time_ms" in v5_metrics and "inference_time_ms" in v8_metrics:
        max_time = max(v5_metrics["inference_time_ms"], v8_metrics["inference_time_ms"])
        v5_values.append(1 - v5_metrics["inference_time_ms"] / max_time)
        v8_values.append(1 - v8_metrics["inference_time_ms"] / max_time)
        labels.append("Speed")
    
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    v5_values += v5_values[:1]
    v8_values += v8_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.plot(angles, v5_values, 'o-', linewidth=2, label='YOLOv5', color='#3498db')
    ax.fill(angles, v5_values, alpha=0.25, color='#3498db')
    
    ax.plot(angles, v8_values, 'o-', linewidth=2, label='YOLOv8', color='#e74c3c')
    ax.fill(angles, v8_values, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    
    ax.set_title('YOLOv5 vs YOLOv8: Multi-dimensional Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    
    for fmt in ["png", "pdf"]:
        fig.savefig(save_dir / f"radar_comparison.{fmt}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: radar_comparison.png/pdf")

def create_performance_table(v5_metrics: Dict, v8_metrics: Dict, save_dir: Path):
    metrics_display = [
        ("mAP50", "mAP@0.5", "Higher is better"),
        ("mAP50-95", "mAP@0.5:0.95", "Higher is better"),
        ("precision", "Precision", "Higher is better"),
        ("recall", "Recall", "Higher is better"),
        ("f1_score", "F1 Score", "Higher is better"),
        ("inference_time_ms", "Inference Time (ms)", "Lower is better"),
        ("model_size_mb", "Model Size (MB)", "Lower is better"),
    ]
    
    data = []
    for metric_key, display_name, note in metrics_display:
        v5_val = v5_metrics.get(metric_key, "N/A")
        v8_val = v8_metrics.get(metric_key, "N/A")
        
        if isinstance(v5_val, (int, float)) and isinstance(v8_val, (int, float)):
            if "Lower" in note:
                winner = "YOLOv5" if v5_val < v8_val else "YOLOv8"
                diff = ((v8_val - v5_val) / v8_val * 100) if "YOLOv5" == winner else ((v5_val - v8_val) / v5_val * 100)
            else:
                winner = "YOLOv5" if v5_val > v8_val else "YOLOv8"
                diff = ((v5_val - v8_val) / v8_val * 100) if "YOLOv5" == winner else ((v8_val - v5_val) / v5_val * 100)
            diff_str = f"+{abs(diff):.1f}%"
        else:
            winner = "N/A"
            diff_str = "N/A"
        
        if isinstance(v5_val, float):
            v5_val = f"{v5_val:.4f}" if v5_val < 10 else f"{v5_val:.2f}"
        if isinstance(v8_val, float):
            v8_val = f"{v8_val:.4f}" if v8_val < 10 else f"{v8_val:.2f}"
        
        data.append({
            "Metric": display_name,
            "YOLOv5": v5_val,
            "YOLOv8": v8_val,
            "Winner": winner,
            "Improvement": diff_str,
            "Note": note
        })
    
    df = pd.DataFrame(data)
    
    df.to_csv(save_dir / "performance_comparison.csv", index=False)
    print(f"  Saved: performance_comparison.csv")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * len(df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for i in range(len(data)):
        winner = data[i]["Winner"]
        if winner == "YOLOv5":
            table[(i + 1, 1)].set_facecolor('#d5f4e6')
        elif winner == "YOLOv8":
            table[(i + 1, 2)].set_facecolor('#fce4d6')
    
    ax.set_title('Performance Comparison Table', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    for fmt in ["png", "pdf"]:
        fig.savefig(save_dir / f"performance_table.{fmt}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: performance_table.png/pdf")

def create_architecture_comparison(save_dir: Path):
    v5_info = get_yolov5_model_info()
    v8_info = get_yolov8_model_info()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.set_title('YOLOv5 Architecture', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    components_v5 = [
        (5, 11, 'Input\n640×640×3', '#ecf0f1'),
        (5, 9.5, 'Focus Layer', '#3498db'),
        (5, 8, 'CSPDarknet53\nBackbone', '#2ecc71'),
        (5, 6, 'SPP Module', '#9b59b6'),
        (5, 4.5, 'PANet\n(FPN + PAN)', '#e74c3c'),
        (5, 2.5, 'YOLOv5 Head\n(Anchor-based)', '#f39c12'),
        (5, 1, 'Detections', '#1abc9c'),
    ]
    
    for x, y, text, color in components_v5:
        ax1.add_patch(plt.Rectangle((x-2, y-0.6), 4, 1.2, 
                                    facecolor=color, edgecolor='black', linewidth=2))
        ax1.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    for i in range(len(components_v5) - 1):
        ax1.annotate('', xy=(5, components_v5[i+1][1] + 0.6),
                    xytext=(5, components_v5[i][1] - 0.6),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_title('YOLOv8 Architecture', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    components_v8 = [
        (5, 11, 'Input\n640×640×3', '#ecf0f1'),
        (5, 9.5, 'Conv 3×3\n(No Focus)', '#3498db'),
        (5, 8, 'C2f Backbone\n(Modified CSP)', '#2ecc71'),
        (5, 6, 'SPPF Module', '#9b59b6'),
        (5, 4.5, 'C2f PANet\n(FPN + PAN)', '#e74c3c'),
        (5, 2.5, 'Decoupled Head\n(Anchor-free)', '#f39c12'),
        (5, 1, 'Detections', '#1abc9c'),
    ]
    
    for x, y, text, color in components_v8:
        ax2.add_patch(plt.Rectangle((x-2, y-0.6), 4, 1.2,
                                    facecolor=color, edgecolor='black', linewidth=2))
        ax2.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    for i in range(len(components_v8) - 1):
        ax2.annotate('', xy=(5, components_v8[i+1][1] + 0.6),
                    xytext=(5, components_v8[i][1] - 0.6),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    plt.tight_layout()
    
    for fmt in ["png", "pdf"]:
        fig.savefig(save_dir / f"architecture_comparison.{fmt}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: architecture_comparison.png/pdf")

def create_tradeoff_analysis(v5_metrics: Dict, v8_metrics: Dict, save_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    v5_map = v5_metrics.get("mAP50", 0)
    v8_map = v8_metrics.get("mAP50", 0)
    v5_time = v5_metrics.get("inference_time_ms", 1)
    v8_time = v8_metrics.get("inference_time_ms", 1)
    v5_size = v5_metrics.get("model_size_mb", 10)
    v8_size = v8_metrics.get("model_size_mb", 10)
    
    ax.scatter(v5_time, v5_map, s=v5_size * 10, c='#3498db', alpha=0.7, 
               edgecolors='black', linewidth=2, label='YOLOv5')
    ax.scatter(v8_time, v8_map, s=v8_size * 10, c='#e74c3c', alpha=0.7,
               edgecolors='black', linewidth=2, label='YOLOv8')
    
    ax.annotate(f'YOLOv5\nmAP: {v5_map:.3f}\nTime: {v5_time:.1f}ms',
               (v5_time, v5_map), textcoords="offset points",
               xytext=(20, 10), ha='left', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.annotate(f'YOLOv8\nmAP: {v8_map:.3f}\nTime: {v8_time:.1f}ms',
               (v8_time, v8_map), textcoords="offset points",
               xytext=(20, -30), ha='left', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Inference Time (ms)', fontsize=12)
    ax.set_ylabel('mAP@0.5', fontsize=12)
    ax.set_title('Accuracy vs Speed Tradeoff\n(Bubble size = Model size)', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.axhline(y=max(v5_map, v8_map), color='green', linestyle='--', alpha=0.5, label='Best Accuracy')
    ax.axvline(x=min(v5_time, v8_time), color='orange', linestyle='--', alpha=0.5, label='Fastest')
    
    plt.tight_layout()
    
    for fmt in ["png", "pdf"]:
        fig.savefig(save_dir / f"tradeoff_analysis.{fmt}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: tradeoff_analysis.png/pdf")

def save_comparison_report(comparison: Dict, save_dir: Path):
    json_path = save_dir / "comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"  Saved: comparison_report.json")
    
    md_content = generate_markdown_report(comparison)
    md_path = save_dir / "comparison_report.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"  Saved: comparison_report.md")

def generate_markdown_report(comparison: Dict) -> str:
    v5 = comparison.get("yolov5", {})
    v8 = comparison.get("yolov8", {})
    
    report = f"""# Comparative Analysis: YOLOv5 vs YOLOv8
## Road Anomaly Detection

**Generated:** {comparison.get('timestamp', 'N/A')}

---

## 1. Executive Summary

This report presents a comprehensive comparison between YOLOv5 and YOLOv8 
for the task of road anomaly detection.

### Models Evaluated
- **YOLOv5:** {comparison.get('yolov5_model', 'N/A')}
- **YOLOv8:** {comparison.get('yolov8_model', 'N/A')}

---

## 2. Performance Metrics

| Metric | YOLOv5 | YOLOv8 | Winner |
|--------|--------|--------|--------|
| mAP@0.5 | {v5.get('mAP50', 'N/A'):.4f if isinstance(v5.get('mAP50'), float) else 'N/A'} | {v8.get('mAP50', 'N/A'):.4f if isinstance(v8.get('mAP50'), float) else 'N/A'} | {'YOLOv5' if v5.get('mAP50', 0) > v8.get('mAP50', 0) else 'YOLOv8'} |
| mAP@0.5:0.95 | {v5.get('mAP50-95', 'N/A'):.4f if isinstance(v5.get('mAP50-95'), float) else 'N/A'} | {v8.get('mAP50-95', 'N/A'):.4f if isinstance(v8.get('mAP50-95'), float) else 'N/A'} | {'YOLOv5' if v5.get('mAP50-95', 0) > v8.get('mAP50-95', 0) else 'YOLOv8'} |
| Precision | {v5.get('precision', 'N/A'):.4f if isinstance(v5.get('precision'), float) else 'N/A'} | {v8.get('precision', 'N/A'):.4f if isinstance(v8.get('precision'), float) else 'N/A'} | {'YOLOv5' if v5.get('precision', 0) > v8.get('precision', 0) else 'YOLOv8'} |
| Recall | {v5.get('recall', 'N/A'):.4f if isinstance(v5.get('recall'), float) else 'N/A'} | {v8.get('recall', 'N/A'):.4f if isinstance(v8.get('recall'), float) else 'N/A'} | {'YOLOv5' if v5.get('recall', 0) > v8.get('recall', 0) else 'YOLOv8'} |
| F1 Score | {v5.get('f1_score', 'N/A'):.4f if isinstance(v5.get('f1_score'), float) else 'N/A'} | {v8.get('f1_score', 'N/A'):.4f if isinstance(v8.get('f1_score'), float) else 'N/A'} | {'YOLOv5' if v5.get('f1_score', 0) > v8.get('f1_score', 0) else 'YOLOv8'} |

---

## 3. Efficiency Metrics

| Metric | YOLOv5 | YOLOv8 | Winner |
|--------|--------|--------|--------|
| Inference Time (ms) | {v5.get('inference_time_ms', 'N/A'):.2f if isinstance(v5.get('inference_time_ms'), float) else 'N/A'} | {v8.get('inference_time_ms', 'N/A'):.2f if isinstance(v8.get('inference_time_ms'), float) else 'N/A'} | {'YOLOv5' if v5.get('inference_time_ms', float('inf')) < v8.get('inference_time_ms', float('inf')) else 'YOLOv8'} |
| Model Size (MB) | {v5.get('model_size_mb', 'N/A'):.2f if isinstance(v5.get('model_size_mb'), float) else 'N/A'} | {v8.get('model_size_mb', 'N/A'):.2f if isinstance(v8.get('model_size_mb'), float) else 'N/A'} | {'YOLOv5' if v5.get('model_size_mb', float('inf')) < v8.get('model_size_mb', float('inf')) else 'YOLOv8'} |

---

## 4. Architecture Differences

### YOLOv5
- **Backbone:** CSPDarknet53
- **Neck:** PANet (Path Aggregation Network)
- **Head:** Coupled, Anchor-based
- **Key Feature:** Focus layer for efficient downsampling

### YOLOv8
- **Backbone:** Modified CSPDarknet with C2f modules
- **Neck:** PANet with C2f modules
- **Head:** Decoupled, Anchor-free
- **Key Feature:** C2f modules for better gradient flow

---

## 5. Conclusions

Based on this comparative analysis:

1. **For Accuracy:** {'YOLOv8' if v8.get('mAP50', 0) > v5.get('mAP50', 0) else 'YOLOv5'} shows better overall detection performance.

2. **For Speed:** {'YOLOv5' if v5.get('inference_time_ms', float('inf')) < v8.get('inference_time_ms', float('inf')) else 'YOLOv8'} offers faster inference times.

3. **For Deployment:** Consider the trade-off between accuracy and speed based on your specific use case.

---

## 6. Visualizations

The following visualizations have been generated:
- `metrics_comparison.png` - Bar chart of key metrics
- `radar_comparison.png` - Multi-dimensional radar chart
- `performance_table.png` - Detailed comparison table
- `architecture_comparison.png` - Visual architecture diagram
- `tradeoff_analysis.png` - Accuracy vs Speed plot

---

*This report was automatically generated by the Road Anomaly Detection comparison tool.*
"""
    return report

def print_comparison_summary(v5_metrics: Dict, v8_metrics: Dict):
    print_colored("\n" + "="*70, "green")
    print_colored("📊 COMPARISON SUMMARY", "bold")
    print_colored("="*70, "green")
    
    print("\n┌─────────────────────┬───────────┬───────────┬──────────┐")
    print("│       Metric        │  YOLOv5   │  YOLOv8   │  Winner  │")
    print("├─────────────────────┼───────────┼───────────┼──────────┤")
    
    comparisons = [
        ("mAP@0.5", "mAP50", True),
        ("mAP@0.5:0.95", "mAP50-95", True),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("F1 Score", "f1_score", True),
        ("Inference (ms)", "inference_time_ms", False),
        ("Size (MB)", "model_size_mb", False),
    ]
    
    v5_wins = 0
    v8_wins = 0
    
    for display, key, higher_better in comparisons:
        v5_val = v5_metrics.get(key, 0)
        v8_val = v8_metrics.get(key, 0)
        
        if higher_better:
            winner = "YOLOv5" if v5_val > v8_val else "YOLOv8"
        else:
            winner = "YOLOv5" if v5_val < v8_val else "YOLOv8"
        
        if winner == "YOLOv5":
            v5_wins += 1
        else:
            v8_wins += 1
        
        v5_str = f"{v5_val:.4f}" if isinstance(v5_val, float) and v5_val < 10 else f"{v5_val:.2f}"
        v8_str = f"{v8_val:.4f}" if isinstance(v8_val, float) and v8_val < 10 else f"{v8_val:.2f}"
        
        print(f"│ {display:<19} │ {v5_str:>9} │ {v8_str:>9} │ {winner:>8} │")
    
    print("└─────────────────────┴───────────┴───────────┴──────────┘")
    
    print(f"\n🏆 Overall: YOLOv5 wins {v5_wins} | YOLOv8 wins {v8_wins}")
    
    if v5_wins > v8_wins:
        print_colored("\n→ YOLOv5 shows better overall performance for this task.", "blue")
    elif v8_wins > v5_wins:
        print_colored("\n→ YOLOv8 shows better overall performance for this task.", "blue")
    else:
        print_colored("\n→ Both models show comparable performance.", "blue")

if __name__ == "__main__":
    print("Comparison module loaded successfully!")

import sys
import shutil
import time
import os
import json
import csv
import random
import statistics
import re
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR,
    TRAINING_CONFIG, IMPROVED_TRAINING_CONFIG, OPTIMIZED_TRAINING_CONFIG,
    INDIVIDUAL_TRAINING_CONFIG,
    YOLO11_CONFIG, YOLO12_CONFIG, GPU_CONFIG,
    YOLO11_MEDIUM_CONFIG, YOLO11_LARGE_CONFIG, YOLO11_XLARGE_CONFIG,
    YOLO12_MEDIUM_CONFIG, YOLO12_LARGE_CONFIG, YOLO12_XLARGE_CONFIG,
    YOLO12_LARGE_DEEP_A2C2F_CONFIG,
    YOLO12_LARGE_P2HEAD_CONFIG,
    YOLO12_LARGE_P2HEAD_DEEP_A2C2F_CONFIG,
    YOLO12_LARGE_WIDER_NECK_CONFIG,
    YOLO12_LARGE_P2HEAD_WIDER_NECK_CONFIG,
    DATASET_CONFIG,
    SVRDD_DIR, SVRDD_DATASET_CONFIG, SVRDD_TRAINING_CONFIG,
    get_results_dir,
)
from config import RESULTS_DIR
from src.utils import print_colored, get_timestamp

def _find_resumable_run(model_name: str, mode_label: str) -> Path:
    if not RESULTS_DIR.exists():
        return None

    prefix = f"{model_name}_{mode_label}_"
    matching = sorted(
        [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.name,
        reverse=True,
    )

    for run_dir in matching:
        last_pt = run_dir / "train" / "weights" / "last.pt"
        if last_pt.exists():
            return run_dir

    return None

def _resolve_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return str(GPU_CONFIG.get("device", "0"))

def _is_gpu_device(device: str) -> bool:
    return device != "cpu"

def _resolve_batch_size(
    requested_batch_size: Optional[int],
    default_batch_size: int,
    using_gpu: bool,
) -> int:
    if requested_batch_size is not None:
        return requested_batch_size
    if using_gpu and GPU_CONFIG.get("auto_batch", False):
        return -1
    return default_batch_size

def _resolve_workers(default_workers: int, using_gpu: bool) -> int:
    workers_cfg = GPU_CONFIG.get("workers", default_workers)

    if isinstance(workers_cfg, int) and workers_cfg > 0:
        workers = workers_cfg
    elif isinstance(workers_cfg, str) and workers_cfg.lower() == "auto":
        cpu_count = os.cpu_count() or 8
        target = max(2, cpu_count - 1)
        max_workers = int(GPU_CONFIG.get("max_workers", 16))
        workers = min(max_workers, target)
    else:
        workers = default_workers

    if not using_gpu:
        workers = min(workers, 8)
    return max(1, workers)

def _resolve_compile_mode(using_gpu: bool):
    compile_cfg = GPU_CONFIG.get("compile", False)
    if not using_gpu:
        return False
    return compile_cfg

def _apply_gpu_optimizations(using_gpu: bool):
    if not using_gpu:
        return

    torch.backends.cudnn.benchmark = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

def _batch_display(batch_size: int) -> str:
    if batch_size == -1:
        return "auto (-1, VRAM-based)"
    return str(batch_size)

def _print_vram_info(device: str):
    if not _is_gpu_device(device):
        return
    try:
        gpu_idx = int(str(device).split(",")[0].strip())
    except (ValueError, IndexError):
        gpu_idx = 0

    props = torch.cuda.get_device_properties(gpu_idx)
    total_vram_gb = props.total_memory / (1024 ** 3)
    print(f"  CUDA device     : {gpu_idx}")
    print(f"  GPU name        : {props.name}")
    print(f"  Total VRAM      : {total_vram_gb:.1f} GB")

def _train_with_compile_fallback(model, train_kwargs: dict):
    try:
        return model.train(**train_kwargs)
    except (RuntimeError, TypeError) as exc:
        if train_kwargs.get("compile", False):
            print_colored(
                f"  compile mode failed ({exc.__class__.__name__}). Retrying with compile=False...",
                "yellow",
            )
            fallback_kwargs = dict(train_kwargs)
            fallback_kwargs.pop("compile", None)
            return model.train(**fallback_kwargs)
        raise

def _is_eval_artifacts_complete(eval_dir: Path) -> bool:
    if not eval_dir.exists() or not eval_dir.is_dir():
        return False

    required_files = (
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
    )
    return all((eval_dir / name).exists() for name in required_files)

def _val_with_worker_fallback(model, val_kwargs: dict):
    try:
        return model.val(**val_kwargs)
    except RuntimeError as exc:
        if "DataLoader worker" not in str(exc):
            raise
        if val_kwargs.get("workers", 0) == 0:
            raise

        print_colored(
            "  Validation DataLoader workers crashed. Retrying with workers=0...",
            "yellow",
        )
        retry_kwargs = dict(val_kwargs)
        retry_kwargs["workers"] = 0
        return model.val(**retry_kwargs)

def train_model(model_key: str, improved: bool = False,
                epochs: int = None, batch_size: int = None):
    model_cfg = YOLO11_CONFIG if model_key == "yolo11" else YOLO12_CONFIG
    train_cfg = IMPROVED_TRAINING_CONFIG if improved else TRAINING_CONFIG
    mode_label = "FreezeCosLR" if improved else "Standard"
    model_name = model_key.upper()
    variant = model_cfg["model_variant"]

    print_colored("\n" + "=" * 65, "blue")
    print_colored(f"  STARTING {model_name} TRAINING ({mode_label})", "bold")
    print_colored("=" * 65, "blue")

    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        print_colored("\n  data.yaml not found! Run setup_dataset.py first.", "red")
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        print_colored("  ultralytics not installed. Installing...", "yellow")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics>=8.3.0"],
                       check=True)
        from ultralytics import YOLO

    epochs = epochs or train_cfg["epochs"]
    device = _resolve_device()
    using_gpu = _is_gpu_device(device)
    _apply_gpu_optimizations(using_gpu)

    batch_size = _resolve_batch_size(
        requested_batch_size=batch_size,
        default_batch_size=train_cfg["batch_size"],
        using_gpu=using_gpu,
    )
    workers = _resolve_workers(
        default_workers=train_cfg["workers"],
        using_gpu=using_gpu,
    )
    compile_mode = _resolve_compile_mode(using_gpu)

    resuming = False
    resume_dir = _find_resumable_run(model_name, mode_label)
    if resume_dir:
        last_pt = resume_dir / "train" / "weights" / "last.pt"
        print_colored(f"\n  Found incomplete training run: {resume_dir.name}", "yellow")
        print(f"  Resume file: {last_pt}")
        resume_choice = input("  Resume from where it stopped? (yes/no): ").strip().lower()
        if resume_choice == "yes":
            resuming = True
            results_dir = resume_dir
        else:
            results_dir = get_results_dir(model_name, mode_label)
    else:
        results_dir = get_results_dir(model_name, mode_label)

    print_colored("\n  Training Configuration:", "bold")
    print(f"  Model variant   : {variant}")
    print(f"  Mode            : {mode_label}")
    print(f"  Epochs          : {epochs}")
    print(f"  Batch size      : {_batch_display(batch_size)}")
    print(f"  Workers         : {workers}")
    print(f"  Image size      : {train_cfg['image_size']}")
    print(f"  Learning rate   : {train_cfg['lr0']}")
    print(f"  Optimizer       : {train_cfg['optimizer']}")
    print(f"  Cosine LR       : {train_cfg['cos_lr']}")
    print(f"  Cache mode      : {GPU_CONFIG['cache']}")
    print(f"  Deterministic   : {GPU_CONFIG.get('deterministic', False)}")
    print(f"  Torch compile   : {compile_mode if compile_mode else False}")
    print(f"  Resume          : {'YES' if resuming else 'No (fresh start)'}")
    if improved:
        print(f"  Frozen layers   : {train_cfg['freeze']}")
    print(f"  Device          : {'GPU' if using_gpu else 'CPU'}")
    _print_vram_info(device)
    print(f"  Results folder  : {results_dir.name}")

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": train_cfg["image_size"],
        "device": device,
        "project": str(results_dir),
        "name": "train",
        "exist_ok": True,
        "patience": train_cfg["patience"],
        "workers": workers,
        "optimizer": train_cfg["optimizer"],
        "lr0": train_cfg["lr0"],
        "lrf": train_cfg["lrf"],
        "cos_lr": train_cfg["cos_lr"],
        "warmup_epochs": train_cfg["warmup_epochs"],
        "weight_decay": train_cfg["weight_decay"],
        "momentum": train_cfg["momentum"],
        "mosaic": train_cfg["mosaic"],
        "mixup": train_cfg["mixup"],
        "hsv_h": train_cfg["hsv_h"],
        "hsv_s": train_cfg["hsv_s"],
        "hsv_v": train_cfg["hsv_v"],
        "degrees": train_cfg["degrees"],
        "translate": train_cfg["translate"],
        "scale": train_cfg["scale"],
        "flipud": train_cfg["flipud"],
        "fliplr": train_cfg["fliplr"],
        "amp": GPU_CONFIG["amp"],
        "cache": GPU_CONFIG["cache"],
        "deterministic": GPU_CONFIG.get("deterministic", False),
        "verbose": True,
    }

    if compile_mode:
        train_kwargs["compile"] = compile_mode

    if improved:
        train_kwargs["freeze"] = train_cfg["freeze"]

    try:
        if resuming:
            last_pt = results_dir / "train" / "weights" / "last.pt"
            print_colored(f"\n  Resuming training from {last_pt.name}...", "yellow")
            model = YOLO(str(last_pt))
            train_kwargs["resume"] = True
        else:
            print_colored(f"\n  Loading pretrained {variant}...", "blue")
            model = YOLO(f"{variant}.pt")

        print_colored("\n  Training started! Progress below:\n", "green")
        start_time = time.time()

        results = _train_with_compile_fallback(model, train_kwargs)

        training_time = time.time() - start_time
        training_time_str = _format_time(training_time)

        print_colored(f"\n  Training completed in {training_time_str}!", "green")

        print_colored("\n  Evaluating on TEST set...", "blue")
        train_run_dir = results_dir / "train"
        best_weights = train_run_dir / "weights" / "best.pt"

        test_metrics = {}
        if best_weights.exists():
            test_model = YOLO(str(best_weights))
            print_colored("  (TTA enabled for evaluation)", "cyan")
            test_results = test_model.val(
                data=str(data_yaml),
                split="test",
                device=device,
                project=str(results_dir),
                name="test_eval",
                exist_ok=True,
                verbose=True,
                augment=True,
            )

            test_metrics = {
                "mAP50": float(test_results.box.map50),
                "mAP50-95": float(test_results.box.map),
                "precision": float(test_results.box.mp),
                "recall": float(test_results.box.mr),
            }
            p = test_metrics["precision"]
            r = test_metrics["recall"]
            test_metrics["f1_score"] = 2 * (p * r) / (p + r + 1e-6)

            class_names = ["D00_Longitudinal_Crack", "D10_Transverse_Crack",
                           "D20_Alligator_Crack", "D40_Pothole"]
            if hasattr(test_results.box, 'ap50'):
                for i, cname in enumerate(class_names):
                    if i < len(test_results.box.ap50):
                        test_metrics[f"AP50_{cname}"] = float(test_results.box.ap50[i])

            test_metrics["model_size_mb"] = best_weights.stat().st_size / (1024 * 1024)

        model_save_dir = MODELS_DIR / model_key
        model_save_dir.mkdir(parents=True, exist_ok=True)

        if best_weights.exists():
            save_name = f"{model_name}_{mode_label}_{get_timestamp()}_best.pt"
            final_weights = model_save_dir / save_name
            shutil.copy2(best_weights, final_weights)
            print_colored(f"\n  Model saved: {final_weights}", "green")

        print_colored("\n  Generating report...", "blue")

        from src.report_generator import generate_report

        report_info = {
            "model_name": model_name,
            "model_variant": variant,
            "mode": mode_label,
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "lr0": train_cfg["lr0"],
            "lrf": train_cfg["lrf"],
            "optimizer": train_cfg["optimizer"],
            "cos_lr": train_cfg["cos_lr"],
            "image_size": train_cfg["image_size"],
            "patience": train_cfg["patience"],
            "cache": GPU_CONFIG["cache"],
            "deterministic": GPU_CONFIG.get("deterministic", False),
            "compile": compile_mode if compile_mode else False,
            "training_time": training_time_str,
            "training_time_seconds": training_time,
            "device": "GPU" if using_gpu else "CPU",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }
        if improved:
            report_info["freeze_layers"] = train_cfg["freeze"]

        report_paths = generate_report(
            report_info=report_info,
            test_metrics=test_metrics,
            results_dir=results_dir,
            train_run_dir=train_run_dir,
        )

        _print_summary(test_metrics, model_name, mode_label, training_time_str)

        print_colored(f"\n  All results saved in: {results_dir}", "green")
        if report_paths:
            for rp in report_paths:
                print(f"    -> {Path(rp).name}")

        return {
            "model_name": model_name,
            "mode": mode_label,
            "metrics": test_metrics,
            "results_dir": str(results_dir),
            "weights": str(final_weights) if best_weights.exists() else None,
        }

    except Exception as e:
        print_colored(f"\n  Training error: {e}", "red")
        import traceback
        traceback.print_exc()
        return None

def _format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

def _print_summary(metrics: dict, model_name: str, mode: str, train_time: str):
    print_colored(f"\n{'=' * 55}", "cyan")
    print_colored(f"  {model_name} ({mode}) - TEST SET RESULTS", "bold")
    print_colored(f"{'=' * 55}", "cyan")
    print(f"  Training Time      : {train_time}")

    if not metrics:
        print("  (No test metrics available)")
        return

    print(f"  mAP@0.5           : {metrics.get('mAP50', 0):.4f}")
    print(f"  mAP@0.5:0.95      : {metrics.get('mAP50-95', 0):.4f}")
    print(f"  Precision          : {metrics.get('precision', 0):.4f}")
    print(f"  Recall             : {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score           : {metrics.get('f1_score', 0):.4f}")
    print(f"  Model Size (MB)    : {metrics.get('model_size_mb', 0):.2f}")

    per_class = {k: v for k, v in metrics.items() if k.startswith("AP50_")}
    if per_class:
        print(f"\n  Per-Class AP@0.5:")
        for k, v in per_class.items():
            cls_name = k.replace("AP50_", "")
            print(f"    {cls_name:<30s}: {v:.4f}")

    print_colored(f"{'=' * 55}\n", "cyan")

MODEL_CONFIGS = {
    "yolo11s": YOLO11_CONFIG,
    "yolo11m": YOLO11_MEDIUM_CONFIG,
    "yolo11l": YOLO11_LARGE_CONFIG,
    "yolo11x": YOLO11_XLARGE_CONFIG,
    "yolo12s": YOLO12_CONFIG,
    "yolo12m": YOLO12_MEDIUM_CONFIG,
    "yolo12l": YOLO12_LARGE_CONFIG,
    "yolo12x": YOLO12_XLARGE_CONFIG,
    "yolo12l-deepA2C2f": YOLO12_LARGE_DEEP_A2C2F_CONFIG,
    "yolo12l-p2head": YOLO12_LARGE_P2HEAD_CONFIG,
    "yolo12l-p2head-deepA2C2f": YOLO12_LARGE_P2HEAD_DEEP_A2C2F_CONFIG,
    "yolo12l-widerNeck": YOLO12_LARGE_WIDER_NECK_CONFIG,
    "yolo12l-p2head-widerNeck": YOLO12_LARGE_P2HEAD_WIDER_NECK_CONFIG,
}

SVRDD_POST_LEARNING_VARIANTS = {
    "baseline": {
        "display": "Baseline YOLO12-L (SVRDD standard)",
        "mode_candidates": ["SVRDD_Large"],
        "mode_tag": "Baseline",
    },
    "v2": {
        "display": "Variant 2 (P2 Head)",
        "mode_candidates": ["SVRDD_Large_P2Head", "SVRDD_Large_P2Head_TransferBase"],
        "mode_tag": "V2_P2Head",
    },
    "v4": {
        "display": "Variant 4 (Wider Neck)",
        "mode_candidates": ["SVRDD_Large_WiderNeck"],
        "mode_tag": "V4_WiderNeck",
    },
    "v5": {
        "display": "Variant 5 (P2 + Wider Neck, direct)",
        "mode_candidates": ["SVRDD_Large_P2Head_WiderNeck"],
        "mode_tag": "V5_P2Head_WiderNeck",
    },
    "v6": {
        "display": "Variant 6 (Two-stage transfer result)",
        "mode_candidates": ["SVRDD_Large_P2Head_WiderNeck_FromV2"],
        "mode_tag": "V6_TwoStageTransfer",
    },
}

def _find_latest_best_checkpoint_from_modes(model_name: str, mode_candidates: list[str]) -> tuple[Optional[Path], Optional[str]]:
    if not RESULTS_DIR.exists():
        return None, None

    matches = []
    ts_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
    for mode in mode_candidates:
        prefix = f"{model_name}_{mode}_"
        for run_dir in RESULTS_DIR.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
                continue
            suffix = run_dir.name[len(prefix):]
            if not ts_pattern.match(suffix):
                continue
            best_pt = run_dir / "train" / "weights" / "best.pt"
            if best_pt.exists():
                matches.append((run_dir.name, best_pt, mode))

    if not matches:
        return None, None

    matches.sort(key=lambda x: x[0], reverse=True)
    _, best_path, matched_mode = matches[0]
    return best_path, matched_mode

def _prepare_svrdd_post_learning_cv_data(
    k_folds: int,
    resolution: tuple[int, int],
    seed: int = 42,
):
    extra_dir = PROJECT_ROOT / "extraImagesForSVRDDPostLearning"
    if not extra_dir.exists():
        print_colored(f"  Missing folder: {extra_dir}", "red")
        return None

    try:
        from PIL import Image
    except ImportError:
        print_colored("  Pillow not installed. Installing...", "yellow")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "Pillow>=10.0.0"], check=True)
        from PIL import Image

    try:
        import yaml
    except ImportError:
        print_colored("  pyyaml not installed. Installing...", "yellow")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml>=6.0"], check=True)
        import yaml

    cv_root = DATA_DIR / "svrdd_postlearning_cv"
    if cv_root.exists():
        shutil.rmtree(cv_root)
    pool_images_dir = cv_root / "pool" / "images"
    pool_labels_dir = cv_root / "pool" / "labels"
    pool_images_dir.mkdir(parents=True, exist_ok=True)
    pool_labels_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    used_image_paths = []
    skipped_wrong_resolution = 0
    skipped_missing_labels = 0
    scanned_images = 0

    def _find_label_for_stem(stem: str) -> Optional[Path]:
        for split in ("train", "val", "test"):
            cand = DATA_DIR / split / "labels" / f"{stem}.txt"
            if cand.exists():
                return cand
        return None

    for src_img in sorted(extra_dir.iterdir(), key=lambda p: p.name):
        if not src_img.is_file() or src_img.suffix.lower() not in image_exts:
            continue
        scanned_images += 1

        try:
            with Image.open(src_img) as im:
                w, h = im.size
        except Exception:
            skipped_wrong_resolution += 1
            continue

        if (w, h) != resolution:
            skipped_wrong_resolution += 1
            continue

        stem_for_lookup = src_img.stem.split("__", 1)[1] if "__" in src_img.stem else src_img.stem
        label_src = _find_label_for_stem(stem_for_lookup)
        if label_src is None:
            skipped_missing_labels += 1
            continue

        new_stem = src_img.stem
        dst_img = pool_images_dir / f"{new_stem}{src_img.suffix.lower()}"
        dst_lbl = pool_labels_dir / f"{new_stem}.txt"

        if dst_img.exists() or dst_lbl.exists():
            suffix_idx = 1
            while True:
                candidate_stem = f"{new_stem}_{suffix_idx}"
                cand_img = pool_images_dir / f"{candidate_stem}{src_img.suffix.lower()}"
                cand_lbl = pool_labels_dir / f"{candidate_stem}.txt"
                if not cand_img.exists() and not cand_lbl.exists():
                    dst_img, dst_lbl = cand_img, cand_lbl
                    break
                suffix_idx += 1

        shutil.copy2(src_img, dst_img)
        shutil.copy2(label_src, dst_lbl)
        used_image_paths.append(dst_img.resolve())

    n = len(used_image_paths)
    if n < 2:
        print_colored("  Not enough labeled images for cross-validation.", "red")
        return None

    if k_folds < 2:
        print_colored("  k_folds must be >= 2.", "red")
        return None
    if k_folds > n:
        print_colored(f"  k_folds ({k_folds}) > usable images ({n}).", "red")
        return None

    rng = random.Random(seed)
    shuffled = used_image_paths[:]
    rng.shuffle(shuffled)

    folds = [[] for _ in range(k_folds)]
    for idx, item in enumerate(shuffled):
        folds[idx % k_folds].append(item)

    fold_yamls = []
    for i in range(k_folds):
        fold_id = i + 1
        fold_dir = cv_root / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_imgs = folds[i]
        train_imgs = []
        for j in range(k_folds):
            if j == i:
                continue
            train_imgs.extend(folds[j])

        train_txt = fold_dir / "train.txt"
        val_txt = fold_dir / "val.txt"
        test_txt = fold_dir / "test.txt"

        with open(train_txt, "w", encoding="utf-8") as f:
            for p in train_imgs:
                f.write(str(p).replace("\\", "/") + "\n")
        with open(val_txt, "w", encoding="utf-8") as f:
            for p in val_imgs:
                f.write(str(p).replace("\\", "/") + "\n")
        with open(test_txt, "w", encoding="utf-8") as f:
            for p in val_imgs:
                f.write(str(p).replace("\\", "/") + "\n")

        yaml_path = fold_dir / "data.yaml"
        yaml_payload = {
            "path": str(fold_dir.resolve()).replace("\\", "/"),
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
            "nc": SVRDD_DATASET_CONFIG["num_classes"],
            "names": {i: n for i, n in enumerate(SVRDD_DATASET_CONFIG["classes"])},
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_payload, f, default_flow_style=False, sort_keys=False)

        fold_yamls.append({
            "fold": fold_id,
            "yaml": yaml_path,
            "train_count": len(train_imgs),
            "val_count": len(val_imgs),
        })

    return {
        "cv_root": cv_root,
        "folds": fold_yamls,
        "usable_count": n,
        "scanned_count": scanned_images,
        "skipped_wrong_resolution": skipped_wrong_resolution,
        "skipped_missing_labels": skipped_missing_labels,
    }

def train_model_svrdd_post_learning_cv(
    post_variant_key: str,
    epochs: int = 30,
    batch_size: int = None,
    k_folds: int = 5,
    patience: int = 12,
    resolution: tuple[int, int] = (1024, 1024),
):
    variant_cfg = SVRDD_POST_LEARNING_VARIANTS.get(post_variant_key)
    if not variant_cfg:
        print_colored(f"  Unknown post-learning key: {post_variant_key}", "red")
        return None

    print_colored("\n" + "=" * 78, "blue")
    print_colored(f"  SVRDD POST-LEARNING + CV: {variant_cfg['display']}", "bold")
    print_colored("=" * 78, "blue")

    base_ckpt, matched_mode = _find_latest_best_checkpoint_from_modes(
        model_name="YOLO12",
        mode_candidates=variant_cfg["mode_candidates"],
    )
    if base_ckpt is None:
        print_colored("  No compatible pre-trained SVRDD checkpoint found.", "red")
        print("  Expected one of these previous modes:")
        for m in variant_cfg["mode_candidates"]:
            print(f"    - {m}")
        return None

    print_colored(f"  Using checkpoint: {base_ckpt}", "green")
    print(f"  Matched mode     : {matched_mode}")

    prep = _prepare_svrdd_post_learning_cv_data(
        k_folds=k_folds,
        resolution=resolution,
        seed=42,
    )
    if not prep:
        return None

    if prep["usable_count"] < k_folds:
        print_colored("  Not enough usable images for the selected number of folds.", "red")
        return None

    print_colored("\n  CV Dataset Preparation:", "bold")
    print(f"  Source images scanned       : {prep['scanned_count']}")
    print(f"  Usable (labeled + resolution): {prep['usable_count']}")
    print(f"  Skipped (resolution mismatch): {prep['skipped_wrong_resolution']}")
    print(f"  Skipped (missing labels)     : {prep['skipped_missing_labels']}")
    print(f"  Resolution filter            : {resolution[0]}x{resolution[1]}")
    print(f"  Folds                        : {k_folds}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print_colored("  ultralytics not installed. Installing...", "yellow")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics>=8.3.0"], check=True)
        from ultralytics import YOLO

    device = _resolve_device()
    using_gpu = _is_gpu_device(device)
    _apply_gpu_optimizations(using_gpu)
    workers = _resolve_workers(default_workers=SVRDD_TRAINING_CONFIG["workers"], using_gpu=using_gpu)
    compile_mode = _resolve_compile_mode(using_gpu)
    resolved_batch = _resolve_batch_size(
        requested_batch_size=batch_size,
        default_batch_size=SVRDD_TRAINING_CONFIG["batch_size"],
        using_gpu=using_gpu,
    )

    print_colored("\n  Training Configuration:", "bold")
    print(f"  Epochs per fold   : {epochs}")
    print(f"  Patience          : {patience}")
    print(f"  Batch size        : {_batch_display(resolved_batch)}")
    print(f"  Workers           : {workers}")
    print(f"  Device            : {'GPU' if using_gpu else 'CPU'}")
    _print_vram_info(device)
    print(f"  Torch compile     : {compile_mode if compile_mode else False}")

    mode_label = f"SVRDD_PostLearningCV_{variant_cfg['mode_tag']}"
    results_dir = get_results_dir("YOLO12", mode_label)
    print(f"  Results folder    : {results_dir}")

    fold_metrics = []
    total_start = time.time()

    for fold_spec in prep["folds"]:
        fold_id = fold_spec["fold"]
        fold_yaml = fold_spec["yaml"]
        fold_dir = results_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = fold_dir / "metrics.json"

        print_colored(f"\n  ===== Fold {fold_id}/{k_folds} =====", "cyan")
        print(f"  Train images: {fold_spec['train_count']}")
        print(f"  Val/Test images: {fold_spec['val_count']}")

        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                fold_metrics.append(existing)
                print_colored("  Reusing existing fold metrics (already completed).", "green")
                continue
            except Exception:
                pass

        train_kwargs = {
            "data": str(fold_yaml),
            "epochs": epochs,
            "batch": resolved_batch,
            "imgsz": SVRDD_TRAINING_CONFIG["image_size"],
            "device": device,
            "project": str(fold_dir),
            "name": "train",
            "exist_ok": True,
            "patience": patience,
            "workers": workers,
            "optimizer": SVRDD_TRAINING_CONFIG["optimizer"],
            "lr0": SVRDD_TRAINING_CONFIG["lr0"],
            "lrf": SVRDD_TRAINING_CONFIG["lrf"],
            "cos_lr": SVRDD_TRAINING_CONFIG["cos_lr"],
            "warmup_epochs": SVRDD_TRAINING_CONFIG["warmup_epochs"],
            "weight_decay": SVRDD_TRAINING_CONFIG["weight_decay"],
            "momentum": SVRDD_TRAINING_CONFIG["momentum"],
            "mosaic": SVRDD_TRAINING_CONFIG["mosaic"],
            "mixup": SVRDD_TRAINING_CONFIG["mixup"],
            "hsv_h": SVRDD_TRAINING_CONFIG["hsv_h"],
            "hsv_s": SVRDD_TRAINING_CONFIG["hsv_s"],
            "hsv_v": SVRDD_TRAINING_CONFIG["hsv_v"],
            "degrees": SVRDD_TRAINING_CONFIG["degrees"],
            "translate": SVRDD_TRAINING_CONFIG["translate"],
            "scale": SVRDD_TRAINING_CONFIG["scale"],
            "flipud": SVRDD_TRAINING_CONFIG["flipud"],
            "fliplr": SVRDD_TRAINING_CONFIG["fliplr"],
            "amp": GPU_CONFIG["amp"],
            "cache": GPU_CONFIG["cache"],
            "deterministic": GPU_CONFIG.get("deterministic", False),
            "verbose": True,
        }
        if compile_mode:
            train_kwargs["compile"] = compile_mode

        last_pt = fold_dir / "train" / "weights" / "last.pt"
        best_pt = fold_dir / "train" / "weights" / "best.pt"
        resumed = False

        try:
            if best_pt.exists():
                print_colored("  Found best.pt for this fold, skipping train and evaluating.", "yellow")
            else:
                if last_pt.exists():
                    print_colored("  Resuming interrupted fold training...", "yellow")
                    model = YOLO(str(last_pt))
                    train_kwargs["resume"] = True
                    resumed = True
                else:
                    print_colored("  Continuing training from existing SVRDD checkpoint...", "blue")
                    model = YOLO(str(base_ckpt))

                if not best_pt.exists():
                    fold_start = time.time()
                    _train_with_compile_fallback(model, train_kwargs)
                    fold_train_time = _format_time(time.time() - fold_start)
                else:
                    fold_train_time = "N/A (reused existing fold train)"

            if not best_pt.exists():
                print_colored("  Fold best.pt not found after training.", "red")
                return None

            eval_model = YOLO(str(best_pt))
            val_kwargs = {
                "data": str(fold_yaml),
                "split": "test",
                "device": device,
                "project": str(fold_dir),
                "name": "test",
                "exist_ok": True,
                "verbose": True,
                "workers": workers,
            }
            fold_test = _val_with_worker_fallback(eval_model, val_kwargs)

            m = {
                "fold": fold_id,
                "mAP50": float(fold_test.box.map50),
                "mAP50-95": float(fold_test.box.map),
                "precision": float(fold_test.box.mp),
                "recall": float(fold_test.box.mr),
                "training_time": fold_train_time if not resumed else f"{fold_train_time} (resumed)",
            }
            p, r = m["precision"], m["recall"]
            m["f1_score"] = 2 * (p * r) / (p + r + 1e-6)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
            fold_metrics.append(m)

            print(f"    mAP50      : {m['mAP50']:.4f}")
            print(f"    mAP50-95   : {m['mAP50-95']:.4f}")
            print(f"    Precision  : {m['precision']:.4f}")
            print(f"    Recall     : {m['recall']:.4f}")
            print(f"    F1         : {m['f1_score']:.4f}")

        except Exception as exc:
            print_colored(f"  Fold {fold_id} failed: {exc}", "red")
            import traceback
            traceback.print_exc()
            return None

    if not fold_metrics:
        print_colored("  No fold metrics were produced.", "red")
        return None

    keys = ["mAP50", "mAP50-95", "precision", "recall", "f1_score"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        summary[f"{k}_mean"] = statistics.mean(vals) if vals else 0.0
        summary[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    total_time = _format_time(time.time() - total_start)
    summary["folds"] = k_folds
    summary["usable_images"] = prep["usable_count"]
    summary["source_scanned"] = prep["scanned_count"]
    summary["skipped_missing_labels"] = prep["skipped_missing_labels"]
    summary["skipped_wrong_resolution"] = prep["skipped_wrong_resolution"]
    summary["base_checkpoint"] = str(base_ckpt)
    summary["matched_mode"] = matched_mode
    summary["epochs_per_fold"] = epochs
    summary["patience"] = patience
    summary["batch_size"] = _batch_display(resolved_batch)
    summary["resolution"] = f"{resolution[0]}x{resolution[1]}"
    summary["total_runtime"] = total_time

    summary_json = results_dir / "cv_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "fold_metrics": fold_metrics}, f, indent=2)

    summary_csv = results_dir / "cv_folds_metrics.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "mAP50", "mAP50-95", "precision", "recall", "f1_score", "training_time"],
        )
        writer.writeheader()
        for row in sorted(fold_metrics, key=lambda x: x["fold"]):
            writer.writerow(row)

    print_colored(f"\n{'='*72}", "cyan")
    print_colored(f"  SVRDD POST-LEARNING CV RESULTS: {variant_cfg['display']}", "bold")
    print_colored(f"{'='*72}", "cyan")
    print(f"  Folds                : {k_folds}")
    print(f"  Usable images        : {prep['usable_count']} / {prep['scanned_count']}")
    print(f"  mAP@0.5 mean +/- std : {summary['mAP50_mean']:.4f} +/- {summary['mAP50_std']:.4f}")
    print(f"  mAP@0.5:0.95         : {summary['mAP50-95_mean']:.4f} +/- {summary['mAP50-95_std']:.4f}")
    print(f"  Precision            : {summary['precision_mean']:.4f} +/- {summary['precision_std']:.4f}")
    print(f"  Recall               : {summary['recall_mean']:.4f} +/- {summary['recall_std']:.4f}")
    print(f"  F1 score             : {summary['f1_score_mean']:.4f} +/- {summary['f1_score_std']:.4f}")
    print(f"  Total runtime        : {total_time}")
    print_colored(f"{'='*72}\n", "cyan")

    print_colored(f"  Summary JSON: {summary_json}", "green")
    print_colored(f"  Summary CSV : {summary_csv}", "green")

    return {
        "post_variant_key": post_variant_key,
        "base_checkpoint": str(base_ckpt),
        "matched_mode": matched_mode,
        "results_dir": str(results_dir),
        "summary": summary,
        "fold_metrics": fold_metrics,
    }

def train_model_percountry(model_key: str, size: str,
                           epochs: int = None, batch_size: int = None,
                           optimized: bool = False,
                           variant_key: Optional[str] = None):
    resolved_variant_key = variant_key or f"{model_key}{size}"
    model_cfg = MODEL_CONFIGS.get(resolved_variant_key)
    if not model_cfg:
        print_colored(f"  Unknown model variant: {resolved_variant_key}", "red")
        return None

    train_cfg = OPTIMIZED_TRAINING_CONFIG if optimized else TRAINING_CONFIG
    model_name = model_key.upper()
    variant = model_cfg["model_variant"]
    ablation_tag = model_cfg.get("ablation_tag", "")
    size_label = {"m": "Medium", "l": "Large", "x": "XLarge"}[size]
    opt_suffix = "_Optimized" if optimized else ""
    if ablation_tag:
        mode_label = f"PerCountry_{size_label}_{ablation_tag}{opt_suffix}"
    else:
        mode_label = f"PerCountry_{size_label}{opt_suffix}"

    print_colored("\n" + "=" * 65, "blue")
    print_colored(f"  STARTING {model_name} {size_label} TRAINING (Per-Country Test)", "bold")
    print_colored("=" * 65, "blue")

    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        print_colored("\n  data.yaml not found! Run setup_dataset.py first.", "red")
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        print_colored("  ultralytics not installed. Installing...", "yellow")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics>=8.3.0"],
                       check=True)
        from ultralytics import YOLO

    print_colored("\n  Setting up per-country test directories...", "blue")
    from src.percountry import setup_percountry_test_dirs, evaluate_percountry, generate_percountry_report
    if not setup_percountry_test_dirs():
        print_colored("  Failed to setup per-country test dirs!", "red")
        return None
    print_colored("  Per-country test directories ready.", "green")

    epochs = epochs or train_cfg["epochs"]
    device = _resolve_device()
    using_gpu = _is_gpu_device(device)
    _apply_gpu_optimizations(using_gpu)

    batch_size = _resolve_batch_size(
        requested_batch_size=batch_size,
        default_batch_size=train_cfg["batch_size"],
        using_gpu=using_gpu,
    )
    workers = _resolve_workers(
        default_workers=train_cfg["workers"],
        using_gpu=using_gpu,
    )
    compile_mode = _resolve_compile_mode(using_gpu)

    resuming = False
    resume_dir = _find_resumable_run(model_name, mode_label)
    if resume_dir:
        last_pt = resume_dir / "train" / "weights" / "last.pt"
        print_colored(f"\n  Found incomplete training run: {resume_dir.name}", "yellow")
        print(f"  Resume file: {last_pt}")
        resume_choice = input("  Resume from where it stopped? (yes/no): ").strip().lower()
        if resume_choice == "yes":
            resuming = True
            results_dir = resume_dir
        else:
            results_dir = get_results_dir(model_name, mode_label)
    else:
        results_dir = get_results_dir(model_name, mode_label)

    print_colored("\n  Training Configuration:", "bold")
    print(f"  Variant key     : {resolved_variant_key}")
    print(f"  Model variant   : {variant}")
    if ablation_tag:
        print(f"  Architecture mod: {ablation_tag}")
    if model_cfg.get("custom_yaml"):
        print(f"  Custom YAML     : {model_cfg['custom_yaml']}")
    print(f"  Size            : {size_label}")
    print(f"  Mode            : Per-Country Test")
    print(f"  Epochs          : {epochs}")
    print(f"  Batch size      : {_batch_display(batch_size)}")
    print(f"  Workers         : {workers}")
    print(f"  Image size      : {train_cfg['image_size']}")
    print(f"  Learning rate   : {train_cfg['lr0']}")
    print(f"  Optimizer       : {train_cfg['optimizer']}")
    print(f"  Patience        : {train_cfg['patience']}")
    print(f"  Cache mode      : {GPU_CONFIG['cache']}")
    print(f"  Deterministic   : {GPU_CONFIG.get('deterministic', False)}")
    print(f"  Torch compile   : {compile_mode if compile_mode else False}")
    print(f"  Resume          : {'YES' if resuming else 'No (fresh start)'}")
    print(f"  Device          : {'GPU' if using_gpu else 'CPU'}")
    _print_vram_info(device)
    print(f"  Results folder  : {results_dir.name}")
    if optimized:
        print_colored(f"\n  Optimized Mode:", "cyan")
        print(f"    Image size        : {train_cfg['image_size']} (high-res)")
        print(f"    Epochs            : {train_cfg['epochs']} (extended)")
        print(f"    Patience          : {train_cfg['patience']}")
        print(f"    Mixup             : {train_cfg['mixup']}")
        print(f"    Copy-paste        : {train_cfg['copy_paste']}")
        print(f"    Close mosaic      : {train_cfg['close_mosaic']}")
        print(f"    Rotation degrees  : {train_cfg['degrees']}")
        print(f"    Scale             : {train_cfg['scale']}")
        print(f"    TTA (eval)        : {train_cfg.get('tta', False)}")
    print(f"\n  Train/Val: ALL countries combined")
    print(f"  Test: EACH country separately (5 evaluations)")

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": train_cfg["image_size"],
        "device": device,
        "project": str(results_dir),
        "name": "train",
        "exist_ok": True,
        "patience": train_cfg["patience"],
        "workers": workers,
        "optimizer": train_cfg["optimizer"],
        "lr0": train_cfg["lr0"],
        "lrf": train_cfg["lrf"],
        "cos_lr": train_cfg["cos_lr"],
        "warmup_epochs": train_cfg["warmup_epochs"],
        "weight_decay": train_cfg["weight_decay"],
        "momentum": train_cfg["momentum"],
        "mosaic": train_cfg["mosaic"],
        "mixup": train_cfg["mixup"],
        "hsv_h": train_cfg["hsv_h"],
        "hsv_s": train_cfg["hsv_s"],
        "hsv_v": train_cfg["hsv_v"],
        "degrees": train_cfg["degrees"],
        "translate": train_cfg["translate"],
        "scale": train_cfg["scale"],
        "flipud": train_cfg["flipud"],
        "fliplr": train_cfg["fliplr"],
        "amp": GPU_CONFIG["amp"],
        "cache": GPU_CONFIG["cache"],
        "deterministic": GPU_CONFIG.get("deterministic", False),
        "verbose": True,
    }

    if "close_mosaic" in train_cfg:
        train_kwargs["close_mosaic"] = train_cfg["close_mosaic"]
    if "copy_paste" in train_cfg:
        train_kwargs["copy_paste"] = train_cfg["copy_paste"]

    use_tta = train_cfg.get("tta", False)

    if compile_mode:
        train_kwargs["compile"] = compile_mode

    try:
        if resuming:
            last_pt = results_dir / "train" / "weights" / "last.pt"
            print_colored(f"\n  Resuming training from {last_pt.name}...", "yellow")
            model = YOLO(str(last_pt))
            train_kwargs["resume"] = True
        else:
            custom_yaml = model_cfg.get("custom_yaml")
            if custom_yaml:
                yaml_path = PROJECT_ROOT / custom_yaml
                print_colored(f"\n  Building custom architecture from {custom_yaml}...", "blue")
                model = YOLO(str(yaml_path))
                print_colored(f"  Transferring pretrained {variant} weights...", "blue")
                model.load(f"{variant}.pt")
            else:
                print_colored(f"\n  Loading pretrained {variant}...", "blue")
                model = YOLO(f"{variant}.pt")

        print_colored("\n  PHASE 1: Training on ALL countries combined...\n", "green")
        start_time = time.time()
        results = _train_with_compile_fallback(model, train_kwargs)
        training_time = time.time() - start_time
        training_time_str = _format_time(training_time)
        print_colored(f"\n  Training completed in {training_time_str}!", "green")

        train_run_dir = results_dir / "train"
        best_weights = train_run_dir / "weights" / "best.pt"

        if not best_weights.exists():
            print_colored("  best.pt not found after training!", "red")
            return None

        print_colored("\n  PHASE 2: Evaluating on ALL countries combined...", "blue")
        test_model = YOLO(str(best_weights))
        val_kwargs = {
            "data": str(data_yaml),
            "split": "test",
            "device": device,
            "project": str(results_dir),
            "name": "test_all",
            "exist_ok": True,
            "verbose": True,
        }
        if use_tta:
            val_kwargs["augment"] = True
            print_colored("  (TTA enabled for evaluation)", "cyan")
        overall_results = test_model.val(**val_kwargs)

        overall_metrics = {
            "mAP50": float(overall_results.box.map50),
            "mAP50-95": float(overall_results.box.map),
            "precision": float(overall_results.box.mp),
            "recall": float(overall_results.box.mr),
        }
        p, r = overall_metrics["precision"], overall_metrics["recall"]
        overall_metrics["f1_score"] = 2 * (p * r) / (p + r + 1e-6)
        overall_metrics["model_size_mb"] = best_weights.stat().st_size / (1024 * 1024)

        class_names = ["D00_Longitudinal_Crack", "D10_Transverse_Crack",
                       "D20_Alligator_Crack", "D40_Pothole"]
        if hasattr(overall_results.box, 'ap50'):
            for i, cname in enumerate(class_names):
                if i < len(overall_results.box.ap50):
                    overall_metrics[f"AP50_{cname}"] = float(overall_results.box.ap50[i])

        print_colored("\n  PHASE 3: Evaluating per country...", "blue")
        country_metrics = evaluate_percountry(
            model_path=str(best_weights),
            results_dir=results_dir,
            device=device,
            use_tta=use_tta,
        )

        print_colored("\n  PHASE 4: Generating reports and graphs...", "blue")

        report_info = {
            "model_name": model_name,
            "model_variant": resolved_variant_key,
            "mode": mode_label,
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "lr0": train_cfg["lr0"],
            "lrf": train_cfg["lrf"],
            "optimizer": train_cfg["optimizer"],
            "cos_lr": train_cfg["cos_lr"],
            "image_size": train_cfg["image_size"],
            "patience": train_cfg["patience"],
            "cache": GPU_CONFIG["cache"],
            "deterministic": GPU_CONFIG.get("deterministic", False),
            "compile": compile_mode if compile_mode else False,
            "training_time": training_time_str,
            "training_time_seconds": training_time,
            "device": "GPU" if using_gpu else "CPU",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }
        if model_cfg.get("custom_yaml"):
            report_info["custom_yaml"] = model_cfg["custom_yaml"]
        if ablation_tag:
            report_info["ablation_tag"] = ablation_tag

        from src.report_generator import generate_report
        report_paths = generate_report(
            report_info=report_info,
            test_metrics=overall_metrics,
            results_dir=results_dir,
            train_run_dir=train_run_dir,
        )

        percountry_paths = generate_percountry_report(
            country_metrics=country_metrics,
            overall_metrics=overall_metrics,
            report_info=report_info,
            results_dir=results_dir,
        )

        model_save_dir = MODELS_DIR / model_key
        model_save_dir.mkdir(parents=True, exist_ok=True)
        save_name = f"{model_name}_{mode_label}_{get_timestamp()}_best.pt"
        final_weights = model_save_dir / save_name
        shutil.copy2(best_weights, final_weights)
        print_colored(f"\n  Model saved: {final_weights}", "green")

        print_colored(f"\n{'=' * 65}", "cyan")
        print_colored(f"  {model_name} {size_label} - PER-COUNTRY TEST RESULTS", "bold")
        print_colored(f"{'=' * 65}", "cyan")
        print(f"  Training Time: {training_time_str}")
        print(f"\n  {'Country':<18s} {'mAP50':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
        print(f"  {'-------':<18s} {'-----':>8s} {'---------':>10s} {'------':>8s} {'--':>8s}")
        for country, m in country_metrics.items():
            print(f"  {country:<18s} {m['mAP50']:>8.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>8.4f} {m['f1_score']:>8.4f}")
        print(f"\n  {'ALL COMBINED':<18s} {overall_metrics['mAP50']:>8.4f} "
              f"{overall_metrics['precision']:>10.4f} {overall_metrics['recall']:>8.4f} "
              f"{overall_metrics['f1_score']:>8.4f}")
        print_colored(f"{'=' * 65}\n", "cyan")

        print_colored(f"\n  All results saved in: {results_dir}", "green")
        all_paths = (report_paths or []) + (percountry_paths or [])
        for rp in all_paths:
            print(f"    -> {Path(rp).name}")

        return {
            "model_name": model_name,
            "mode": mode_label,
            "variant_key": resolved_variant_key,
            "overall_metrics": overall_metrics,
            "country_metrics": country_metrics,
            "results_dir": str(results_dir),
            "weights": str(final_weights),
        }

    except Exception as e:
        print_colored(f"\n  Training error: {e}", "red")
        import traceback
        traceback.print_exc()
        return None

def train_model_individual_country(model_key: str, size: str,
                                    epochs: int = None, batch_size: int = None):
    variant_key = f"{model_key}{size}"
    model_cfg = MODEL_CONFIGS.get(variant_key)
    if not model_cfg:
        print_colored(f"  Unknown model variant: {variant_key}", "red")
        return None

    train_cfg = INDIVIDUAL_TRAINING_CONFIG
    model_name = model_key.upper()
    variant = model_cfg["model_variant"]
    size_label = {"l": "Large", "x": "XLarge"}[size]
    mode_label = f"IndividualCountry_{size_label}"

    print_colored("\n" + "=" * 65, "blue")
    print_colored(f"  {model_name} {size_label} - INDIVIDUAL COUNTRY TRAINING", "bold")
    print_colored("=" * 65, "blue")

    individual_dir = DATA_DIR / "individual"
    countries = DATASET_CONFIG["countries"]
    available_countries = []

    for country in countries:
        yaml_path = individual_dir / country / "data.yaml"
        if yaml_path.exists():
            available_countries.append(country)

    if not available_countries:
        print_colored("\n  No per-country splits found!", "red")
        print_colored("  Run: python setup_dataset.py -> option 2", "yellow")
        return None

    print_colored(f"\n  Found {len(available_countries)}/{len(countries)} country splits", "green")
    for c in available_countries:
        print(f"    - {c}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print_colored("  ultralytics not installed. Installing...", "yellow")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics>=8.3.0"],
                       check=True)
        from ultralytics import YOLO

    epochs = epochs or train_cfg["epochs"]
    device = _resolve_device()
    using_gpu = _is_gpu_device(device)
    _apply_gpu_optimizations(using_gpu)

    batch_size = _resolve_batch_size(
        requested_batch_size=batch_size,
        default_batch_size=train_cfg["batch_size"],
        using_gpu=using_gpu,
    )
    workers = _resolve_workers(
        default_workers=train_cfg["workers"],
        using_gpu=using_gpu,
    )
    compile_mode = _resolve_compile_mode(using_gpu)

    results_dir = None
    prefix = f"{model_name}_{mode_label}_"
    if RESULTS_DIR.exists():
        matching = sorted(
            [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)],
            key=lambda d: d.name,
            reverse=True,
        )
        for candidate in matching:
            has_countries = any(
                (candidate / c).exists() for c in available_countries
            )
            if not has_countries:
                continue
            all_done = all(
                (candidate / c / "train" / "weights" / "best.pt").exists()
                and (candidate / c / "test").exists()
                for c in available_countries
            )
            if not all_done:
                done_list = [
                    c for c in available_countries
                    if (candidate / c / "train" / "weights" / "best.pt").exists()
                    and (candidate / c / "test").exists()
                ]
                pending_list = [c for c in available_countries if c not in done_list]
                print_colored(f"\n  Found incomplete run: {candidate.name}", "yellow")
                print(f"    Completed : {', '.join(done_list) if done_list else 'none'}")
                print(f"    Remaining : {', '.join(pending_list)}")
                print_colored("  Resuming from this folder.", "green")
                results_dir = candidate
                break

    if results_dir is None:
        results_dir = get_results_dir(model_name, mode_label)

    print_colored("\n  Training Configuration:", "bold")
    print(f"  Model variant   : {variant}")
    print(f"  Size            : {size_label}")
    print(f"  Mode            : Individual Country Training")
    print(f"  Epochs          : {epochs} (patience-based stop)")
    print(f"  Patience        : {train_cfg['patience']} (stopping criterion)")
    print(f"  Batch size      : {_batch_display(batch_size)}")
    print(f"  Workers         : {workers}")
    print(f"  Image size      : {train_cfg['image_size']}")
    print(f"  Learning rate   : {train_cfg['lr0']}")
    print(f"  Optimizer       : {train_cfg['optimizer']}")
    print(f"  Cache mode      : {GPU_CONFIG['cache']}")
    print(f"  Torch compile   : {compile_mode if compile_mode else False}")
    print(f"  Device          : {'GPU' if using_gpu else 'CPU'}")
    _print_vram_info(device)
    print(f"  Results folder  : {results_dir.name}")
    print(f"\n  Strategy: Train/Val/Test EACH country SEPARATELY (80/10/10)")

    all_country_metrics = {}
    all_country_times = {}
    total_start = time.time()

    for country in available_countries:
        country_yaml = individual_dir / country / "data.yaml"
        country_results_dir = results_dir / country

        best_pt = country_results_dir / "train" / "weights" / "best.pt"
        test_dir = country_results_dir / "test"
        if best_pt.exists() and test_dir.exists():
            print_colored(f"\n  {country}: already completed, skipping.", "cyan")
            try:
                from ultralytics import YOLO as _YOLO
                _m = _YOLO(str(best_pt))
                _r = _m.val(data=str(country_yaml), split="test", device=device,
                            project=str(country_results_dir), name="test",
                            exist_ok=True, verbose=False)
                m = {
                    "mAP50": float(_r.box.map50), "mAP50-95": float(_r.box.map),
                    "precision": float(_r.box.mp), "recall": float(_r.box.mr),
                }
                p, r = m["precision"], m["recall"]
                m["f1_score"] = 2 * (p * r) / (p + r + 1e-6)
                m["model_size_mb"] = best_pt.stat().st_size / (1024 * 1024)
                m["training_time"] = "completed"
                all_country_metrics[country] = m
                print(f"    mAP50={m['mAP50']:.4f}, F1={m['f1_score']:.4f}")
            except Exception:
                pass
            continue

        print_colored(f"\n{'='*65}", "green")
        print_colored(f"  TRAINING ON: {country.upper()}", "bold")
        print_colored(f"{'='*65}", "green")

        resuming = False
        last_pt = country_results_dir / "train" / "weights" / "last.pt"

        if last_pt.exists() and not best_pt.exists():
            print_colored(f"  Found interrupted training for {country}, resuming...", "yellow")
            resuming = True

        train_kwargs = {
            "data": str(country_yaml),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": train_cfg["image_size"],
            "device": device,
            "project": str(country_results_dir),
            "name": "train",
            "exist_ok": True,
            "patience": train_cfg["patience"],
            "workers": workers,
            "optimizer": train_cfg["optimizer"],
            "lr0": train_cfg["lr0"],
            "lrf": train_cfg["lrf"],
            "cos_lr": train_cfg["cos_lr"],
            "warmup_epochs": train_cfg["warmup_epochs"],
            "weight_decay": train_cfg["weight_decay"],
            "momentum": train_cfg["momentum"],
            "mosaic": train_cfg["mosaic"],
            "mixup": train_cfg["mixup"],
            "hsv_h": train_cfg["hsv_h"],
            "hsv_s": train_cfg["hsv_s"],
            "hsv_v": train_cfg["hsv_v"],
            "degrees": train_cfg["degrees"],
            "translate": train_cfg["translate"],
            "scale": train_cfg["scale"],
            "flipud": train_cfg["flipud"],
            "fliplr": train_cfg["fliplr"],
            "amp": GPU_CONFIG["amp"],
            "cache": GPU_CONFIG["cache"],
            "deterministic": GPU_CONFIG.get("deterministic", False),
            "verbose": True,
        }

        if compile_mode:
            train_kwargs["compile"] = compile_mode

        try:
            if resuming:
                print_colored(f"  Resuming from {last_pt.name}...", "yellow")
                model = YOLO(str(last_pt))
                train_kwargs["resume"] = True
            else:
                print_colored(f"  Loading pretrained {variant}...", "blue")
                model = YOLO(f"{variant}.pt")

            country_start = time.time()
            results = _train_with_compile_fallback(model, train_kwargs)
            country_time = time.time() - country_start
            country_time_str = _format_time(country_time)
            all_country_times[country] = country_time

            print_colored(f"\n  {country} training done in {country_time_str}", "green")

            best_weights = country_results_dir / "train" / "weights" / "best.pt"
            if not best_weights.exists():
                print_colored(f"  best.pt not found for {country}!", "red")
                continue

            print_colored(f"  Evaluating {country} test set...", "blue")
            test_model = YOLO(str(best_weights))
            test_results = test_model.val(
                data=str(country_yaml),
                split="test",
                device=device,
                project=str(country_results_dir),
                name="test",
                exist_ok=True,
                verbose=True,
            )

            metrics = {
                "mAP50": float(test_results.box.map50),
                "mAP50-95": float(test_results.box.map),
                "precision": float(test_results.box.mp),
                "recall": float(test_results.box.mr),
            }
            p, r = metrics["precision"], metrics["recall"]
            metrics["f1_score"] = 2 * (p * r) / (p + r + 1e-6)
            metrics["model_size_mb"] = best_weights.stat().st_size / (1024 * 1024)
            metrics["training_time"] = country_time_str

            class_names = DATASET_CONFIG["classes"]
            if hasattr(test_results.box, 'ap50'):
                for i, cname in enumerate(class_names):
                    if i < len(test_results.box.ap50):
                        metrics[f"AP50_{cname}"] = float(test_results.box.ap50[i])

            all_country_metrics[country] = metrics

            model_save_dir = MODELS_DIR / model_key
            model_save_dir.mkdir(parents=True, exist_ok=True)
            save_name = f"{model_name}_{mode_label}_{country}_{get_timestamp()}_best.pt"
            final_weights = model_save_dir / save_name
            shutil.copy2(best_weights, final_weights)

            print(f"  {country}: mAP50={metrics['mAP50']:.4f}, "
                  f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                  f"F1={metrics['f1_score']:.4f} ({country_time_str})")

        except Exception as e:
            print_colored(f"\n  Error training {country}: {e}", "red")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start
    total_time_str = _format_time(total_time)

    if not all_country_metrics:
        print_colored("\n  No countries trained successfully!", "red")
        return None

    avg_metrics = {}
    metric_keys = ["mAP50", "mAP50-95", "precision", "recall", "f1_score"]
    for key in metric_keys:
        vals = [m[key] for m in all_country_metrics.values() if key in m]
        avg_metrics[key] = sum(vals) / len(vals) if vals else 0

    print_colored("\n  Generating combined report...", "blue")
    report_info = {
        "model_name": model_name,
        "model_variant": variant,
        "mode": mode_label,
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "lr0": train_cfg["lr0"],
        "lrf": train_cfg["lrf"],
        "optimizer": train_cfg["optimizer"],
        "cos_lr": train_cfg["cos_lr"],
        "image_size": train_cfg["image_size"],
        "patience": train_cfg["patience"],
        "cache": GPU_CONFIG["cache"],
        "deterministic": GPU_CONFIG.get("deterministic", False),
        "compile": compile_mode if compile_mode else False,
        "training_time": total_time_str,
        "training_time_seconds": total_time,
        "device": "GPU" if using_gpu else "CPU",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }

    from src.percountry import generate_percountry_report
    percountry_paths = generate_percountry_report(
        country_metrics=all_country_metrics,
        overall_metrics=avg_metrics,
        report_info=report_info,
        results_dir=results_dir,
    )

    print_colored(f"\n{'='*65}", "cyan")
    print_colored(f"  {model_name} {size_label} - INDIVIDUAL COUNTRY RESULTS", "bold")
    print_colored(f"{'='*65}", "cyan")
    print(f"  Total Time: {total_time_str}")
    print(f"  Stopping criterion: patience={train_cfg['patience']}")
    print(f"\n  {'Country':<18s} {'mAP50':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'Time':>10s}")
    print(f"  {'-------':<18s} {'-----':>8s} {'---------':>10s} {'------':>8s} {'--':>8s} {'----':>10s}")
    for country, m in all_country_metrics.items():
        t = m.get('training_time', '?')
        print(f"  {country:<18s} {m['mAP50']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} {t:>10s}")
    print(f"\n  {'AVERAGE':<18s} {avg_metrics['mAP50']:>8.4f} {avg_metrics['precision']:>10.4f} "
          f"{avg_metrics['recall']:>8.4f} {avg_metrics['f1_score']:>8.4f}")
    print_colored(f"{'='*65}\n", "cyan")

    print_colored(f"\n  All results saved in: {results_dir}", "green")
    if percountry_paths:
        for rp in percountry_paths:
            print(f"    -> {Path(rp).name}")

    return {
        "model_name": model_name,
        "mode": mode_label,
        "country_metrics": all_country_metrics,
        "avg_metrics": avg_metrics,
        "results_dir": str(results_dir),
    }

def train_model_svrdd(model_key: str, size: str, batch_size: int = None,
                      ablation_key: str = None,
                      mode_label_override: str = None,
                      initial_weights: Optional[str] = None,
                      reuse_completed: bool = False):
    if ablation_key:
        variant_key = ablation_key
    else:
        variant_key = f"{model_key}{size}"
    model_cfg = MODEL_CONFIGS.get(variant_key)
    if not model_cfg:
        print_colored(f"  Unknown model variant: {variant_key}", "red")
        return None

    train_cfg = SVRDD_TRAINING_CONFIG
    dataset_cfg = SVRDD_DATASET_CONFIG
    model_name = model_key.upper()
    variant = model_cfg["model_variant"]
    size_label = {"s": "Small", "m": "Medium", "l": "Large", "x": "XLarge"}[size]
    ablation_tag = model_cfg.get("ablation_tag", "")
    mode_label = mode_label_override or (
        f"SVRDD_{size_label}_{ablation_tag}" if ablation_tag else f"SVRDD_{size_label}"
    )

    print_colored("\n" + "=" * 65, "blue")
    if ablation_tag:
        print_colored(f"  {model_name} {size_label} - SVRDD ABLATION: {ablation_tag}", "bold")
    else:
        print_colored(f"  {model_name} {size_label} - SVRDD DATASET TRAINING", "bold")
    print_colored("=" * 65, "blue")

    data_yaml = SVRDD_DIR / "data.yaml"
    if not data_yaml.exists():
        print_colored("\n  SVRDD dataset not found!", "red")
        print_colored(f"  Expected at: {SVRDD_DIR}", "yellow")
        print_colored("  Download SVRDD_YOLO.zip from Zenodo and extract to SVRDD_dataset/", "yellow")
        return None

    import yaml
    with open(data_yaml, "r") as f:
        yaml_content = yaml.safe_load(f)
    abs_svrdd = str(SVRDD_DIR.resolve()).replace("\\", "/")
    if yaml_content.get("path") != abs_svrdd:
        yaml_content["path"] = abs_svrdd
        with open(data_yaml, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        print_colored(f"  Updated data.yaml path to: {abs_svrdd}", "cyan")

    try:
        from ultralytics import YOLO
    except ImportError:
        print_colored("  ultralytics not installed. Installing...", "yellow")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics>=8.3.0"],
                       check=True)
        from ultralytics import YOLO

    epochs = train_cfg["epochs"]
    device = _resolve_device()
    using_gpu = _is_gpu_device(device)
    _apply_gpu_optimizations(using_gpu)

    batch_size = _resolve_batch_size(
        requested_batch_size=batch_size,
        default_batch_size=train_cfg["batch_size"],
        using_gpu=using_gpu,
    )
    workers = _resolve_workers(
        default_workers=train_cfg["workers"],
        using_gpu=using_gpu,
    )
    compile_mode = _resolve_compile_mode(using_gpu)

    results_dir = None
    prefix = f"{model_name}_{mode_label}_"
    if RESULTS_DIR.exists():
        matching = sorted(
            [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)],
            key=lambda d: d.name,
            reverse=True,
        )
        for candidate in matching:
            train_dir = candidate / "train"
            best_pt = train_dir / "weights" / "best.pt"
            test_dir = candidate / "test"
            if best_pt.exists() and _is_eval_artifacts_complete(test_dir):
                continue
            if train_dir.exists():
                print_colored(f"\n  Found incomplete run: {candidate.name}", "yellow")
                print_colored("  Resuming from this folder.", "green")
                results_dir = candidate
                break

    if results_dir is None:
        results_dir = get_results_dir(model_name, mode_label)

    print_colored("\n  Training Configuration:", "bold")
    print(f"  Dataset         : SVRDD (Street View Road Damage Detection)")
    print(f"  Images          : 6000 train / 1000 val / 1000 test")
    print(f"  Classes         : {dataset_cfg['num_classes']} ({', '.join(dataset_cfg['classes'])})")
    print(f"  Model variant   : {variant}")
    print(f"  Epochs          : {epochs} (patience-based stop)")
    print(f"  Patience        : {train_cfg['patience']}")
    print(f"  Batch size      : {_batch_display(batch_size)}")
    print(f"  Workers         : {workers}")
    print(f"  Image size      : {train_cfg['image_size']}")
    print(f"  Learning rate   : {train_cfg['lr0']}")
    print(f"  Optimizer       : {train_cfg['optimizer']}")
    print(f"  Cache mode      : {GPU_CONFIG['cache']}")
    print(f"  Torch compile   : {compile_mode if compile_mode else False}")
    if initial_weights:
        print(f"  Init weights    : {initial_weights}")
    print(f"  Device          : {'GPU' if using_gpu else 'CPU'}")
    _print_vram_info(device)
    print(f"  Results folder  : {results_dir.name}")

    resuming = False
    evaluation_only = False
    last_pt = results_dir / "train" / "weights" / "last.pt"
    best_pt = results_dir / "train" / "weights" / "best.pt"
    test_dir = results_dir / "test"
    test_complete = _is_eval_artifacts_complete(test_dir)

    if best_pt.exists() and test_complete:
        print_colored(f"\n  This run is already complete!", "green")
        print_colored("  Delete the folder to re-train, or check results.", "yellow")
        if reuse_completed:
            return {
                "model_name": model_name,
                "mode": mode_label,
                "metrics": None,
                "results_dir": str(results_dir),
                "weights": str(best_pt),
                "best_weights": str(best_pt),
                "reused_completed": True,
            }
        return None

    if best_pt.exists() and not test_complete:
        print_colored(
            "\n  Found trained weights but missing/incomplete test outputs.",
            "yellow",
        )
        print_colored("  Skipping training and running test evaluation only.", "green")
        evaluation_only = True
    elif last_pt.exists():
        print_colored(f"\n  Found interrupted training, resuming...", "yellow")
        resuming = True

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": train_cfg["image_size"],
        "device": device,
        "project": str(results_dir),
        "name": "train",
        "exist_ok": True,
        "patience": train_cfg["patience"],
        "workers": workers,
        "optimizer": train_cfg["optimizer"],
        "lr0": train_cfg["lr0"],
        "lrf": train_cfg["lrf"],
        "cos_lr": train_cfg["cos_lr"],
        "warmup_epochs": train_cfg["warmup_epochs"],
        "weight_decay": train_cfg["weight_decay"],
        "momentum": train_cfg["momentum"],
        "mosaic": train_cfg["mosaic"],
        "mixup": train_cfg["mixup"],
        "hsv_h": train_cfg["hsv_h"],
        "hsv_s": train_cfg["hsv_s"],
        "hsv_v": train_cfg["hsv_v"],
        "degrees": train_cfg["degrees"],
        "translate": train_cfg["translate"],
        "scale": train_cfg["scale"],
        "flipud": train_cfg["flipud"],
        "fliplr": train_cfg["fliplr"],
        "amp": GPU_CONFIG["amp"],
        "cache": GPU_CONFIG["cache"],
        "deterministic": GPU_CONFIG.get("deterministic", False),
        "verbose": True,
    }

    if compile_mode:
        train_kwargs["compile"] = compile_mode

    try:
        training_time_str = "N/A (evaluation only, reused existing best.pt)"

        if not evaluation_only:
            custom_yaml = model_cfg.get("custom_yaml")
            if resuming:
                print_colored(f"\n  Resuming from {last_pt.name}...", "yellow")
                model = YOLO(str(last_pt))
                train_kwargs["resume"] = True
            elif custom_yaml:
                from config import PROJECT_ROOT
                yaml_path = PROJECT_ROOT / custom_yaml
                print_colored(f"\n  Building custom architecture from {custom_yaml}...", "blue")
                model = YOLO(str(yaml_path))
                if initial_weights:
                    print_colored(f"  Transferring weights from: {initial_weights}", "blue")
                    model.load(str(initial_weights))
                else:
                    print_colored(f"  Transferring pretrained {variant} weights...", "blue")
                    model.load(f"{variant}.pt")
            else:
                if initial_weights:
                    print_colored(f"\n  Loading initial weights: {initial_weights}...", "blue")
                    model = YOLO(str(initial_weights))
                else:
                    print_colored(f"\n  Loading pretrained {variant}...", "blue")
                    model = YOLO(f"{variant}.pt")

            print_colored(f"\n  Phase 1: Training {model_name} on SVRDD...", "blue")
            train_start = time.time()
            _train_with_compile_fallback(model, train_kwargs)
            training_time = time.time() - train_start
            training_time_str = _format_time(training_time)

            print_colored(f"\n  Training done in {training_time_str}", "green")
        else:
            print_colored("  Reusing existing training outputs from this run.", "cyan")

        best_weights = results_dir / "train" / "weights" / "best.pt"
        if not best_weights.exists():
            print_colored("  best.pt not found!", "red")
            return None

        if test_dir.exists() and not test_complete:
            print_colored("  Cleaning incomplete test folder before re-evaluation...", "yellow")
            shutil.rmtree(test_dir)

        print_colored(f"\n  Phase 2: Evaluating on SVRDD test set...", "blue")
        test_model = YOLO(str(best_weights))
        val_kwargs = {
            "data": str(data_yaml),
            "split": "test",
            "device": device,
            "project": str(results_dir),
            "name": "test",
            "exist_ok": True,
            "verbose": True,
            "workers": workers,
        }
        test_results = _val_with_worker_fallback(test_model, val_kwargs)

        metrics = {
            "mAP50": float(test_results.box.map50),
            "mAP50-95": float(test_results.box.map),
            "precision": float(test_results.box.mp),
            "recall": float(test_results.box.mr),
        }
        p, r = metrics["precision"], metrics["recall"]
        metrics["f1_score"] = 2 * (p * r) / (p + r + 1e-6)
        metrics["model_size_mb"] = best_weights.stat().st_size / (1024 * 1024)
        metrics["training_time"] = training_time_str

        class_names = dataset_cfg["classes"]
        if hasattr(test_results.box, 'ap50'):
            for i, cname in enumerate(class_names):
                if i < len(test_results.box.ap50):
                    metrics[f"AP50_{cname}"] = float(test_results.box.ap50[i])

        model_save_dir = MODELS_DIR / model_key
        model_save_dir.mkdir(parents=True, exist_ok=True)
        save_name = f"{model_name}_{mode_label}_{get_timestamp()}_best.pt"
        final_weights = model_save_dir / save_name
        shutil.copy2(best_weights, final_weights)

        print_colored(f"\n{'='*65}", "cyan")
        print_colored(f"  {model_name} {size_label} - SVRDD TEST RESULTS", "bold")
        print_colored(f"{'='*65}", "cyan")
        print(f"  Training Time      : {training_time_str}")
        print(f"  Patience           : {train_cfg['patience']}")
        print(f"  mAP@0.5           : {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95      : {metrics['mAP50-95']:.4f}")
        print(f"  Precision          : {metrics['precision']:.4f}")
        print(f"  Recall             : {metrics['recall']:.4f}")
        print(f"  F1 Score           : {metrics['f1_score']:.4f}")
        print(f"  Model Size (MB)    : {metrics['model_size_mb']:.2f}")

        per_class = {k: v for k, v in metrics.items() if k.startswith("AP50_")}
        if per_class:
            print(f"\n  Per-Class AP@0.5:")
            for k, v in per_class.items():
                cls_name = k.replace("AP50_", "")
                print(f"    {cls_name:<25s}: {v:.4f}")

        print_colored(f"{'='*65}\n", "cyan")
        print_colored(f"\n  All results saved in: {results_dir}", "green")

        return {
            "model_name": model_name,
            "mode": mode_label,
            "metrics": metrics,
            "results_dir": str(results_dir),
            "weights": str(final_weights),
            "best_weights": str(best_weights),
        }

    except Exception as e:
        print_colored(f"\n  Training error: {e}", "red")
        import traceback
        traceback.print_exc()
        return None

def train_model_svrdd_variant2_to_variant5(batch_size: int = None):
    print_colored("\n" + "=" * 72, "blue")
    print_colored("  SVRDD TWO-STAGE PIPELINE: VARIANT 2 -> VARIANT 5 TRANSFER", "bold")
    print_colored("=" * 72, "blue")

    stage1_mode = "SVRDD_Large_P2Head_TransferBase"
    stage2_mode = "SVRDD_Large_P2Head_WiderNeck_FromV2"

    print_colored("\n  Stage 1/2: Train Variant 2 (P2 Head)", "cyan")
    stage1 = train_model_svrdd(
        model_key="yolo12",
        size="l",
        batch_size=batch_size,
        ablation_key="yolo12l-p2head",
        mode_label_override=stage1_mode,
        reuse_completed=True,
    )
    if not stage1:
        print_colored("  Stage 1 failed.", "red")
        return None

    stage1_best = stage1.get("best_weights")
    if not stage1_best:
        stage1_best = str(Path(stage1["results_dir"]) / "train" / "weights" / "best.pt")
    if not Path(stage1_best).exists():
        print_colored(f"  Stage 1 best.pt not found: {stage1_best}", "red")
        return None

    print_colored("\n  Stage 2/2: Train Variant 5 from Stage 1 best.pt", "cyan")
    stage2 = train_model_svrdd(
        model_key="yolo12",
        size="l",
        batch_size=batch_size,
        ablation_key="yolo12l-p2head-widerNeck",
        mode_label_override=stage2_mode,
        initial_weights=stage1_best,
        reuse_completed=True,
    )
    if not stage2:
        print_colored("  Stage 2 failed.", "red")
        return None

    print_colored("\n  Two-stage training complete!", "green")
    print(f"  Stage 1 results : {stage1['results_dir']}")
    print(f"  Stage 1 best.pt : {stage1_best}")
    print(f"  Stage 2 results : {stage2['results_dir']}")
    print(f"  Stage 2 best.pt : {stage2.get('best_weights', 'N/A')}")

    return {
        "pipeline": "SVRDD_V2_to_V5",
        "stage1": stage1,
        "stage2": stage2,
        "results_dir": stage2["results_dir"],
        "weights": stage2.get("weights"),
    }

def _prepare_rdd2022_india_only_yaml():
    try:
        import yaml
    except ImportError:
        print_colored("  pyyaml not installed. Installing...", "yellow")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"], check=True)
        import yaml

    india_only_dir = DATA_DIR / "india_only"
    india_only_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {}
    split_list_paths = {}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for split in ("train", "val", "test"):
        images_dir = DATA_DIR / split / "images"
        labels_dir = DATA_DIR / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            print_colored(f"  Missing split folders for '{split}' in data/.", "red")
            return None, None

        india_images = sorted(
            [
                p for p in images_dir.iterdir()
                if p.is_file()
                and p.name.startswith("India_")
                and p.suffix.lower() in image_exts
            ],
            key=lambda p: p.name,
        )

        filtered_images = []
        skipped_without_labels = 0
        for img_path in india_images:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                filtered_images.append(img_path)
            else:
                skipped_without_labels += 1

        if not filtered_images:
            print_colored(f"  No India images with labels found in split '{split}'.", "red")
            return None, None

        if skipped_without_labels:
            print_colored(
                f"  Warning: skipped {skipped_without_labels} '{split}' images without labels.",
                "yellow",
            )

        list_path = india_only_dir / f"{split}.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for img_path in filtered_images:
                f.write(str(img_path.resolve()).replace("\\", "/") + "\n")

        split_counts[split] = len(filtered_images)
        split_list_paths[split] = f"india_only/{split}.txt"

    data_yaml = india_only_dir / "data.yaml"
    yaml_content = {
        "path": str(DATA_DIR.resolve()).replace("\\", "/"),
        "train": split_list_paths["train"],
        "val": split_list_paths["val"],
        "test": split_list_paths["test"],
        "nc": DATASET_CONFIG["num_classes"],
        "names": {i: name for i, name in enumerate(DATASET_CONFIG["classes"])},
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    return data_yaml, split_counts

def train_model_rdd2022_india(model_key: str, size: str, epochs: int = None,
                              batch_size: int = None, ablation_key: str = None,
                              mode_label_override: str = None,
                              initial_weights: Optional[str] = None,
                              reuse_completed: bool = False):
    variant_key = ablation_key or f"{model_key}{size}"
    model_cfg = MODEL_CONFIGS.get(variant_key)
    if not model_cfg:
        print_colored(f"  Unknown model variant: {variant_key}", "red")
        return None

    size_label = {"s": "Small", "m": "Medium", "l": "Large", "x": "XLarge"}[size]
    train_cfg = TRAINING_CONFIG
    model_name = model_key.upper()
    variant = model_cfg["model_variant"]
    ablation_tag = model_cfg.get("ablation_tag", "")
    mode_label = mode_label_override or (
        f"RDD2022_IndiaOnly_{size_label}_{ablation_tag}" if ablation_tag
        else f"RDD2022_IndiaOnly_{size_label}"
    )

    print_colored("\n" + "=" * 72, "blue")
    if ablation_tag:
        print_colored(f"  {model_name} {size_label} - RDD2022 INDIA-ONLY ABLATION: {ablation_tag}", "bold")
    else:
        print_colored(f"  {model_name} {size_label} - RDD2022 INDIA-ONLY TRAINING", "bold")
    print_colored("=" * 72, "blue")

    data_yaml, split_counts = _prepare_rdd2022_india_only_yaml()
    if not data_yaml:
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        print_colored("  ultralytics not installed. Installing...", "yellow")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics>=8.3.0"], check=True)
        from ultralytics import YOLO

    epochs = epochs or train_cfg["epochs"]
    device = _resolve_device()
    using_gpu = _is_gpu_device(device)
    _apply_gpu_optimizations(using_gpu)

    batch_size = _resolve_batch_size(
        requested_batch_size=batch_size,
        default_batch_size=train_cfg["batch_size"],
        using_gpu=using_gpu,
    )
    workers = _resolve_workers(
        default_workers=train_cfg["workers"],
        using_gpu=using_gpu,
    )
    compile_mode = _resolve_compile_mode(using_gpu)

    results_dir = None
    prefix = f"{model_name}_{mode_label}_"
    if RESULTS_DIR.exists():
        matching = sorted(
            [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)],
            key=lambda d: d.name,
            reverse=True,
        )
        for candidate in matching:
            train_dir = candidate / "train"
            best_pt = train_dir / "weights" / "best.pt"
            test_dir = candidate / "test"
            if best_pt.exists() and _is_eval_artifacts_complete(test_dir):
                continue
            if train_dir.exists():
                print_colored(f"\n  Found incomplete run: {candidate.name}", "yellow")
                print_colored("  Resuming from this folder.", "green")
                results_dir = candidate
                break

    if results_dir is None:
        results_dir = get_results_dir(model_name, mode_label)

    print_colored("\n  Training Configuration:", "bold")
    print(f"  Dataset         : RDD2022 India-only")
    print(f"  Split sizes     : train={split_counts['train']} / val={split_counts['val']} / test={split_counts['test']}")
    print(f"  Classes         : {DATASET_CONFIG['num_classes']} ({', '.join(DATASET_CONFIG['classes'])})")
    print(f"  Model variant   : {variant_key}")
    print(f"  Epochs          : {epochs}")
    print(f"  Patience        : {train_cfg['patience']}")
    print(f"  Batch size      : {_batch_display(batch_size)}")
    print(f"  Workers         : {workers}")
    print(f"  Image size      : {train_cfg['image_size']}")
    print(f"  Learning rate   : {train_cfg['lr0']}")
    print(f"  Optimizer       : {train_cfg['optimizer']}")
    print(f"  Cache mode      : {GPU_CONFIG['cache']}")
    print(f"  Torch compile   : {compile_mode if compile_mode else False}")
    if initial_weights:
        print(f"  Init weights    : {initial_weights}")
    print(f"  Device          : {'GPU' if using_gpu else 'CPU'}")
    _print_vram_info(device)
    print(f"  Results folder  : {results_dir.name}")

    resuming = False
    evaluation_only = False
    last_pt = results_dir / "train" / "weights" / "last.pt"
    best_pt = results_dir / "train" / "weights" / "best.pt"
    test_dir = results_dir / "test"
    test_complete = _is_eval_artifacts_complete(test_dir)

    if best_pt.exists() and test_complete:
        print_colored(f"\n  This run is already complete!", "green")
        print_colored("  Delete the folder to re-train, or check results.", "yellow")
        if reuse_completed:
            return {
                "model_name": model_name,
                "mode": mode_label,
                "metrics": None,
                "results_dir": str(results_dir),
                "weights": str(best_pt),
                "best_weights": str(best_pt),
                "reused_completed": True,
            }
        return None

    if best_pt.exists() and not test_complete:
        print_colored("\n  Found trained weights but missing/incomplete test outputs.", "yellow")
        print_colored("  Skipping training and running test evaluation only.", "green")
        evaluation_only = True
    elif last_pt.exists():
        print_colored(f"\n  Found interrupted training, resuming...", "yellow")
        resuming = True

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": train_cfg["image_size"],
        "device": device,
        "project": str(results_dir),
        "name": "train",
        "exist_ok": True,
        "patience": train_cfg["patience"],
        "workers": workers,
        "optimizer": train_cfg["optimizer"],
        "lr0": train_cfg["lr0"],
        "lrf": train_cfg["lrf"],
        "cos_lr": train_cfg["cos_lr"],
        "warmup_epochs": train_cfg["warmup_epochs"],
        "weight_decay": train_cfg["weight_decay"],
        "momentum": train_cfg["momentum"],
        "mosaic": train_cfg["mosaic"],
        "mixup": train_cfg["mixup"],
        "hsv_h": train_cfg["hsv_h"],
        "hsv_s": train_cfg["hsv_s"],
        "hsv_v": train_cfg["hsv_v"],
        "degrees": train_cfg["degrees"],
        "translate": train_cfg["translate"],
        "scale": train_cfg["scale"],
        "flipud": train_cfg["flipud"],
        "fliplr": train_cfg["fliplr"],
        "amp": GPU_CONFIG["amp"],
        "cache": GPU_CONFIG["cache"],
        "deterministic": GPU_CONFIG.get("deterministic", False),
        "verbose": True,
    }
    if compile_mode:
        train_kwargs["compile"] = compile_mode

    try:
        training_time_str = "N/A (evaluation only, reused existing best.pt)"

        if not evaluation_only:
            custom_yaml = model_cfg.get("custom_yaml")
            if resuming:
                print_colored(f"\n  Resuming from {last_pt.name}...", "yellow")
                model = YOLO(str(last_pt))
                train_kwargs["resume"] = True
            elif custom_yaml:
                yaml_path = PROJECT_ROOT / custom_yaml
                print_colored(f"\n  Building custom architecture from {custom_yaml}...", "blue")
                model = YOLO(str(yaml_path))
                if initial_weights:
                    print_colored(f"  Transferring weights from: {initial_weights}", "blue")
                    model.load(str(initial_weights))
                else:
                    print_colored(f"  Transferring pretrained {variant} weights...", "blue")
                    model.load(f"{variant}.pt")
            else:
                if initial_weights:
                    print_colored(f"\n  Loading initial weights: {initial_weights}...", "blue")
                    model = YOLO(str(initial_weights))
                else:
                    print_colored(f"\n  Loading pretrained {variant}...", "blue")
                    model = YOLO(f"{variant}.pt")

            print_colored(f"\n  Phase 1: Training {model_name} on RDD2022 India-only...", "blue")
            train_start = time.time()
            _train_with_compile_fallback(model, train_kwargs)
            training_time = time.time() - train_start
            training_time_str = _format_time(training_time)
            print_colored(f"\n  Training done in {training_time_str}", "green")
        else:
            print_colored("  Reusing existing training outputs from this run.", "cyan")

        best_weights = results_dir / "train" / "weights" / "best.pt"
        if not best_weights.exists():
            print_colored("  best.pt not found!", "red")
            return None

        if test_dir.exists() and not test_complete:
            print_colored("  Cleaning incomplete test folder before re-evaluation...", "yellow")
            shutil.rmtree(test_dir)

        print_colored(f"\n  Phase 2: Evaluating on India-only test split...", "blue")
        test_model = YOLO(str(best_weights))
        val_kwargs = {
            "data": str(data_yaml),
            "split": "test",
            "device": device,
            "project": str(results_dir),
            "name": "test",
            "exist_ok": True,
            "verbose": True,
            "workers": workers,
        }
        test_results = _val_with_worker_fallback(test_model, val_kwargs)

        metrics = {
            "mAP50": float(test_results.box.map50),
            "mAP50-95": float(test_results.box.map),
            "precision": float(test_results.box.mp),
            "recall": float(test_results.box.mr),
        }
        p, r = metrics["precision"], metrics["recall"]
        metrics["f1_score"] = 2 * (p * r) / (p + r + 1e-6)
        metrics["model_size_mb"] = best_weights.stat().st_size / (1024 * 1024)
        metrics["training_time"] = training_time_str

        class_names = DATASET_CONFIG["classes"]
        if hasattr(test_results.box, "ap50"):
            for i, cname in enumerate(class_names):
                if i < len(test_results.box.ap50):
                    metrics[f"AP50_{cname}"] = float(test_results.box.ap50[i])

        from src.report_generator import generate_report
        report_info = {
            "model_name": model_name,
            "model_variant": variant_key,
            "mode": mode_label,
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "lr0": train_cfg["lr0"],
            "lrf": train_cfg["lrf"],
            "optimizer": train_cfg["optimizer"],
            "cos_lr": train_cfg["cos_lr"],
            "image_size": train_cfg["image_size"],
            "patience": train_cfg["patience"],
            "cache": GPU_CONFIG["cache"],
            "deterministic": GPU_CONFIG.get("deterministic", False),
            "compile": compile_mode if compile_mode else False,
            "training_time": training_time_str,
            "device": "GPU" if using_gpu else "CPU",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }
        if model_cfg.get("custom_yaml"):
            report_info["custom_yaml"] = model_cfg["custom_yaml"]
        if ablation_tag:
            report_info["ablation_tag"] = ablation_tag

        train_run_dir = results_dir / "train"
        report_paths = generate_report(
            report_info=report_info,
            test_metrics=metrics,
            results_dir=results_dir,
            train_run_dir=train_run_dir,
        )

        model_save_dir = MODELS_DIR / model_key
        model_save_dir.mkdir(parents=True, exist_ok=True)
        save_name = f"{model_name}_{mode_label}_{get_timestamp()}_best.pt"
        final_weights = model_save_dir / save_name
        shutil.copy2(best_weights, final_weights)

        print_colored(f"\n{'='*65}", "cyan")
        print_colored(f"  {model_name} {size_label} - RDD2022 INDIA-ONLY TEST RESULTS", "bold")
        print_colored(f"{'='*65}", "cyan")
        print(f"  Training Time      : {training_time_str}")
        print(f"  mAP@0.5            : {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95       : {metrics['mAP50-95']:.4f}")
        print(f"  Precision          : {metrics['precision']:.4f}")
        print(f"  Recall             : {metrics['recall']:.4f}")
        print(f"  F1 Score           : {metrics['f1_score']:.4f}")
        print(f"  Model Size (MB)    : {metrics['model_size_mb']:.2f}")
        print_colored(f"{'='*65}\n", "cyan")

        print_colored(f"\n  All results saved in: {results_dir}", "green")
        for rp in report_paths or []:
            print(f"    -> {Path(rp).name}")

        return {
            "model_name": model_name,
            "mode": mode_label,
            "metrics": metrics,
            "results_dir": str(results_dir),
            "weights": str(final_weights),
            "best_weights": str(best_weights),
        }

    except Exception as e:
        print_colored(f"\n  Training error: {e}", "red")
        import traceback
        traceback.print_exc()
        return None

def train_model_rdd2022_india_variant2_to_variant5(epochs: int = None, batch_size: int = None):
    print_colored("\n" + "=" * 76, "blue")
    print_colored("  RDD2022 INDIA-ONLY TWO-STAGE PIPELINE: VARIANT 2 -> VARIANT 5", "bold")
    print_colored("=" * 76, "blue")

    stage1_mode = "RDD2022_IndiaOnly_Large_P2Head_TransferBase"
    stage2_mode = "RDD2022_IndiaOnly_Large_P2Head_WiderNeck_FromV2"

    print_colored("\n  Stage 1/2: Train Variant 2 (P2 Head) on India-only", "cyan")
    stage1 = train_model_rdd2022_india(
        model_key="yolo12",
        size="l",
        epochs=epochs,
        batch_size=batch_size,
        ablation_key="yolo12l-p2head",
        mode_label_override=stage1_mode,
        reuse_completed=True,
    )
    if not stage1:
        print_colored("  Stage 1 failed.", "red")
        return None

    stage1_best = stage1.get("best_weights")
    if not stage1_best:
        stage1_best = str(Path(stage1["results_dir"]) / "train" / "weights" / "best.pt")
    if not Path(stage1_best).exists():
        print_colored(f"  Stage 1 best.pt not found: {stage1_best}", "red")
        return None

    print_colored("\n  Stage 2/2: Train Variant 5 from Stage 1 best.pt (India-only)", "cyan")
    stage2 = train_model_rdd2022_india(
        model_key="yolo12",
        size="l",
        epochs=epochs,
        batch_size=batch_size,
        ablation_key="yolo12l-p2head-widerNeck",
        mode_label_override=stage2_mode,
        initial_weights=stage1_best,
        reuse_completed=True,
    )
    if not stage2:
        print_colored("  Stage 2 failed.", "red")
        return None

    print_colored("\n  Two-stage training complete!", "green")
    print(f"  Stage 1 results : {stage1['results_dir']}")
    print(f"  Stage 1 best.pt : {stage1_best}")
    print(f"  Stage 2 results : {stage2['results_dir']}")
    print(f"  Stage 2 best.pt : {stage2.get('best_weights', 'N/A')}")

    return {
        "pipeline": "RDD2022_India_V2_to_V5",
        "stage1": stage1,
        "stage2": stage2,
        "results_dir": stage2["results_dir"],
        "weights": stage2.get("weights"),
    }

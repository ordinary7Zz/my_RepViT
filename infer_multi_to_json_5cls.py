"""
Multi-dataset inference to JSON for RepViT multi-class classification (5 classes).

This script extends infer_multi_to_json.py to support 5-class (or general n-class)
classification. For each sample, it outputs the full probability vector and the
argmax prediction. Per-dataset metrics and confidence intervals are also saved.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def resolve_output_dir(output_dir, checkpoint_path):
    if output_dir:
        return Path(output_dir)
    project_root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint_stem = Path(checkpoint_path).resolve().parent.name
    return project_root / "test_log" / f"infer_multi_to_json_5cls_{checkpoint_stem}_{timestamp}"


def sanitize_filename(name):
    import re
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return sanitized or "dataset"


def collect_prediction_records(model, dataloader, dataset, model_name,
                                num_classes, device="cuda"):
    import torch

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if all_probs:
        y_prob = np.concatenate(all_probs, axis=0).astype(np.float64)
        y_true = np.concatenate(all_labels, axis=0).astype(int)
    else:
        y_prob = np.empty((0, num_classes), dtype=np.float64)
        y_true = np.empty((0,), dtype=int)

    sample_entries = getattr(dataset, "samples", None)
    if sample_entries is None:
        sample_entries = getattr(dataset, "imgs", [])

    if len(sample_entries) != len(y_true):
        raise ValueError(
            f"Dataset samples ({len(sample_entries)}) do not match predictions ({len(y_true)})"
        )

    records = []
    for (image_path, dataset_label), true_label, prob_vec in zip(
            sample_entries, y_true, y_prob):
        if int(dataset_label) != int(true_label):
            raise ValueError(
                f"Label mismatch for {image_path}: dataset={dataset_label}, "
                f"predicted_batch_order={true_label}"
            )

        prob_list = [float(p) for p in prob_vec]
        predicted_class = int(np.argmax(prob_vec))
        confidence = float(prob_vec[predicted_class])

        record = {
            "record_type": "sample",
            "image_file": str(Path(image_path).resolve()),
            "image_name": Path(image_path).name,
            "selected_model": model_name,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "true_label": int(true_label),
            "num_classes": num_classes,
        }
        # 输出每个类的概率: prob_class_0, prob_class_1, ...
        for i in range(num_classes):
            record[f"prob_class_{i}"] = prob_list[i]

        records.append(record)

    return records, y_true, y_prob


def save_records(records, output_path):
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def save_metrics(metrics_summary, output_path):
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_summary, handle, ensure_ascii=False, indent=2)


def format_metric(name, value, ci):
    lower, upper = ci
    if np.isnan(value):
        return f"{name}: nan [nan, nan]"
    if np.isnan(lower) or np.isnan(upper):
        return f"{name}: {value:.4f} [nan, nan]"
    return f"{name}: {value:.4f} [{lower:.4f}, {upper:.4f}]"


def main():
    parser = argparse.ArgumentParser(
        "Multi-dataset inference to JSON for RepViT 5-class classification"
    )

    parser.add_argument("--model", type=str, default="repvit_m1_0",
                        help="RepViT model name, e.g. repvit_m0_9, repvit_m1_0")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint, e.g. checkpoints/.../checkpoint_best.pth")
    parser.add_argument("--test-dirs", type=str, nargs="+", required=True,
                        help="One or more test dataset roots (ImageFolder style)")
    parser.add_argument("--test-names", type=str, nargs="*", default=None,
                        help="Optional names for each test dataset (same order as test-dirs)")
    parser.add_argument("--num-classes", type=int, default=5,
                        help="Number of classes (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--ci-alpha", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save per-dataset JSON files; defaults to test_log directory")

    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    from eval_multi_test_5cls import (
        METRIC_ORDER,
        bootstrap_ci,
        build_test_transform,
        compute_metrics,
        compute_confusion_matrix,
        load_model,
        format_confusion_matrix,
    )
    import model  # noqa: F401  # 导入以注册 RepViT 模型到 timm

    output_dir = resolve_output_dir(args.output_dir, args.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = build_test_transform(args.input_size)
    inference_model = load_model(
        args.model, args.checkpoint,
        num_classes=args.num_classes, device=device
    )
    checkpoint_stem = Path(args.checkpoint).resolve().parent.name

    print(f"Num classes: {args.num_classes}")
    print(f"Saving per-dataset JSON outputs to: {output_dir}")
    print("\n========== JSON Export Inference (5-class) ==========")

    for idx, test_root in enumerate(args.test_dirs):
        if not os.path.isdir(test_root):
            print(f"[WARN] Test dir not found, skip: {test_root}")
            continue

        dataset = datasets.ImageFolder(root=test_root, transform=transform)

        detected_classes = len(dataset.classes)
        if detected_classes != args.num_classes:
            print(f"[WARN] Dataset {test_root} has {detected_classes} classes, "
                  f"but model expects {args.num_classes}. "
                  f"Labels present: {dataset.class_to_idx}")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if args.test_names is not None and len(args.test_names) == len(args.test_dirs):
            dataset_name = args.test_names[idx]
        else:
            dataset_name = Path(test_root).name

        print(f"\n--- Dataset: {dataset_name} ({test_root}), size={len(dataset)} ---")
        print(f"    Classes: {dataset.class_to_idx}")

        records, y_true, y_prob = collect_prediction_records(
            inference_model,
            dataloader,
            dataset,
            model_name=args.model,
            num_classes=args.num_classes,
            device=device,
        )

        filename = f"{sanitize_filename(dataset_name)}__{sanitize_filename(checkpoint_stem)}__predictions.json"
        output_path = output_dir / filename
        save_records(records, output_path)

        metrics_filename = f"{sanitize_filename(dataset_name)}__{sanitize_filename(checkpoint_stem)}__metrics.json"
        metrics_output_path = output_dir / metrics_filename

        if len(y_true) > 0:
            metrics = compute_metrics(
                y_true, y_prob,
                num_classes=args.num_classes,
                ece_bins=args.ece_bins,
            )
            if args.bootstrap_iters > 0:
                ci = bootstrap_ci(
                    y_true, y_prob,
                    num_classes=args.num_classes,
                    ece_bins=args.ece_bins,
                    n_boot=args.bootstrap_iters,
                    alpha=args.ci_alpha,
                    seed=args.seed,
                )
            else:
                ci = {name: (float("nan"), float("nan")) for name in METRIC_ORDER}
            label_set = np.unique(y_true).tolist()

            cm = compute_confusion_matrix(y_true, y_prob, args.num_classes)
            cm_list = cm.tolist()
        else:
            metrics = {name: float("nan") for name in METRIC_ORDER}
            ci = {name: (float("nan"), float("nan")) for name in METRIC_ORDER}
            label_set = []
            cm_list = []

        metrics_summary = {
            "record_type": "classification_metrics_summary",
            "dataset_name": dataset_name,
            "dataset_root": str(Path(test_root).resolve()),
            "model": args.model,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "num_classes": args.num_classes,
            "class_to_idx": dataset.class_to_idx,
            "ece_bins": args.ece_bins,
            "bootstrap_iters": args.bootstrap_iters,
            "ci_alpha": args.ci_alpha,
            "seed": args.seed,
            "n_samples": int(len(y_true)),
            "label_set": label_set,
            "metrics": metrics,
            "confidence_intervals": {
                name: {
                    "lower": ci[name][0],
                    "upper": ci[name][1],
                }
                for name in METRIC_ORDER
            },
            "confusion_matrix": cm_list,
        }
        save_metrics(metrics_summary, metrics_output_path)

        print(
            "  ".join(format_metric(m, metrics[m], ci[m]) for m in METRIC_ORDER)
        )
        if len(cm_list) > 0:
            print(format_confusion_matrix(cm))

        if len(records) > 0:
            max_probs = np.max(y_prob, axis=1)
            print(
                f"Saved {len(records)} records to {output_path} | "
                f"Saved metrics to {metrics_output_path} | "
                f"label_set={label_set} | "
                f"conf_range=[{float(np.min(max_probs)):.6f}, {float(np.max(max_probs)):.6f}]"
            )
        else:
            print(f"Saved 0 records to {output_path} | Saved metrics to {metrics_output_path}")


if __name__ == "__main__":
    main()

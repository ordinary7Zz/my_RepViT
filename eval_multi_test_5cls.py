"""
Multi-dataset evaluation for RepViT multi-class classification (5 classes).

This script extends eval_multi_test.py to support 5-class (or general n-class)
classification by computing multi-class metrics instead of binary ones.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from timm.models import create_model

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    top_k_accuracy_score,
)

import utils  # 来自项目根目录的 utils.py
import model  # 导入以注册 RepViT 模型到 timm


METRIC_ORDER = [
    "AUROC_macro_ovr",
    "ACC",
    "ACC_top2",
    "PREC_macro",
    "RECALL_macro",
    "F1_macro",
    "PREC_weighted",
    "RECALL_weighted",
    "F1_weighted",
    "ECE",
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def resolve_output_path(output_file, checkpoint_path):
    if output_file:
        return Path(output_file)
    project_root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint_stem = Path(checkpoint_path).resolve().parent.name
    return project_root / "test_log" / f"eval_multi_test_5cls_{checkpoint_stem}_{timestamp}.txt"


def build_test_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_model(model_name, ckpt_path, num_classes=5, device="cuda"):
    print(f"Loading model {model_name} from {ckpt_path} (num_classes={num_classes})")
    model = create_model(
        model_name,
        num_classes=num_classes,
        distillation=False,
        pretrained=False,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    model_state = model.state_dict()
    removed_keys = []
    for k in list(state_dict.keys()):
        if k in model_state and state_dict[k].shape != model_state[k].shape:
            removed_keys.append(k)
            del state_dict[k]
    if removed_keys:
        print(f"Removed incompatible keys from checkpoint (shape mismatch): {removed_keys}")

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded state_dict with msg: {msg}")

    model.to(device)
    model.eval()
    utils.replace_batchnorm(model)

    return model


def collect_predictions(model, dataloader, num_classes, device="cuda"):
    """返回 y_true (N,), y_prob (N, num_classes)"""
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

    y_prob = np.concatenate(all_probs, axis=0).astype(np.float64)
    y_true = np.concatenate(all_labels, axis=0).astype(int)
    return y_true, y_prob


def compute_ece(y_true, y_prob, n_bins=10):
    """基于最大预测概率的 Expected Calibration Error（多分类）"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges[1:-1], right=True)

    ece = 0.0
    n = len(y_true)
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.sum() / n
    return float(ece)


def compute_metrics(y_true, y_prob, num_classes, ece_bins=10):
    """计算多分类指标"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.argmax(y_prob, axis=1)

    # AUROC (macro, one-vs-rest)
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        auroc = float("nan")
    else:
        try:
            auroc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=list(range(num_classes)),
            )
        except ValueError:
            auroc = float("nan")

    acc = accuracy_score(y_true, y_pred)

    # top-2 accuracy（仅当类别数 >= 2 时有意义）
    if num_classes >= 2:
        acc_top2 = top_k_accuracy_score(y_true, y_prob, k=min(2, num_classes))
    else:
        acc_top2 = float("nan")

    prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=list(range(num_classes))
    )
    prec_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=list(range(num_classes))
    )

    ece = compute_ece(y_true, y_prob, n_bins=ece_bins)

    return {
        "AUROC_macro_ovr": float(auroc),
        "ACC": float(acc),
        "ACC_top2": float(acc_top2),
        "PREC_macro": float(prec_macro),
        "RECALL_macro": float(recall_macro),
        "F1_macro": float(f1_macro),
        "PREC_weighted": float(prec_w),
        "RECALL_weighted": float(recall_w),
        "F1_weighted": float(f1_w),
        "ECE": float(ece),
    }


def compute_confusion_matrix(y_true, y_prob, num_classes):
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm


def bootstrap_ci(y_true, y_prob, num_classes, ece_bins=10, n_boot=2000, alpha=0.95, seed=42):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    lower_q = (1.0 - alpha) / 2.0
    upper_q = 1.0 - lower_q
    rng = np.random.default_rng(seed)

    boot_metrics = {name: [] for name in METRIC_ORDER}
    n = len(y_true)

    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        metrics = compute_metrics(
            y_true[sample_idx],
            y_prob[sample_idx],
            num_classes=num_classes,
            ece_bins=ece_bins,
        )
        for name, value in metrics.items():
            if not np.isnan(value):
                boot_metrics[name].append(value)

    ci = {}
    for name in METRIC_ORDER:
        values = np.asarray(boot_metrics[name], dtype=np.float64)
        if values.size == 0:
            ci[name] = (float("nan"), float("nan"))
        else:
            ci[name] = (
                float(np.quantile(values, lower_q)),
                float(np.quantile(values, upper_q)),
            )
    return ci


def format_metric(name, value, ci):
    lower, upper = ci
    if np.isnan(value):
        return f"{name}: nan [nan, nan]"
    if np.isnan(lower) or np.isnan(upper):
        return f"{name}: {value:.4f} [nan, nan]"
    return f"{name}: {value:.4f} [{lower:.4f}, {upper:.4f}]"


def format_confusion_matrix(cm):
    lines = ["Confusion Matrix (rows=true, cols=pred):"]
    header = "      " + "  ".join(f"{i:>6d}" for i in range(cm.shape[1]))
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"  {i:>3d} " + "  ".join(f"{v:>6d}" for v in row))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser("Multi-dataset evaluation for RepViT 5-class classification")

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
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save evaluation output; defaults to test_log directory")

    args = parser.parse_args()

    output_path = resolve_output_path(args.output_file, args.checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with output_path.open("w", encoding="utf-8") as output_handle:
        sys.stdout = Tee(original_stdout, output_handle)
        try:
            print(f"Saving evaluation output to: {output_path}")
            print(f"Num classes: {args.num_classes}")

            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            transform = build_test_transform(args.input_size)

            model = load_model(args.model, args.checkpoint,
                               num_classes=args.num_classes, device=device)

            print("\n========== Evaluation (5-class) ==========")
            for idx, test_root in enumerate(args.test_dirs):
                if not os.path.isdir(test_root):
                    print(f"[WARN] Test dir not found, skip: {test_root}")
                    continue

                dataset = datasets.ImageFolder(root=test_root, transform=transform)

                # 检查测试集类别数是否与模型一致
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
                    name = args.test_names[idx]
                else:
                    name = Path(test_root).name

                print(f"\n--- Dataset: {name} ({test_root}), size={len(dataset)} ---")
                print(f"    Classes: {dataset.class_to_idx}")

                y_true, y_prob = collect_predictions(
                    model, dataloader, num_classes=args.num_classes, device=device
                )

                metrics = compute_metrics(
                    y_true, y_prob,
                    num_classes=args.num_classes,
                    ece_bins=args.ece_bins,
                )
                ci = bootstrap_ci(
                    y_true, y_prob,
                    num_classes=args.num_classes,
                    ece_bins=args.ece_bins,
                    n_boot=args.bootstrap_iters,
                    alpha=args.ci_alpha,
                    seed=args.seed,
                )

                print("  ".join(format_metric(m, metrics[m], ci[m]) for m in METRIC_ORDER))

                cm = compute_confusion_matrix(y_true, y_prob, args.num_classes)
                print(format_confusion_matrix(cm))
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()

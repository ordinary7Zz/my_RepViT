import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from timm.models import create_model

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import utils  # 来自项目根目录的 utils.py
import model  # 导入以注册 RepViT 模型到 timm


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


def load_model(model_name, ckpt_path, num_classes=2, device="cuda"):
    print(f"Loading model {model_name} from {ckpt_path}")
    model = create_model(
        model_name,
        num_classes=num_classes,
        distillation=False,
        pretrained=False,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # 处理分类头维度不一致（例如从 ImageNet 1000 类权重加载到 2 类模型）
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

    # 可选：把 BN fuse 掉，和 main.py 里 eval 时保持一致
    utils.replace_batchnorm(model)

    return model


def evaluate_on_dataset(model, dataloader, device="cuda"):
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            logits = model(images)
            # 假设是二分类：num_classes=2，取第 1 类的概率作为阳性概率
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_score = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # 如果某个测试集只有单一类别，AUROC / AUPRC 会报错，做一下保护
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 1:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    # 0.5 阈值转为标签
    y_pred = (y_score >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # confusion_matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = float("nan")

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "ACC": acc,
        "PREC": prec,
        "RECALL": recall,
        "F1": f1,
        "SPEC": specificity,
    }


def main():
    parser = argparse.ArgumentParser("Multi-dataset evaluation for RepViT binary classification")

    parser.add_argument("--model", type=str, default="repvit_m1_0",
                        help="RepViT model name, e.g. repvit_m0_9, repvit_m1_0")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint, e.g. checkpoints/.../checkpoint_best.pth")
    parser.add_argument("--test-dirs", type=str, nargs="+", required=True,
                        help="One or more test dataset roots (ImageFolder style)")
    parser.add_argument("--test-names", type=str, nargs="*", default=None,
                        help="Optional names for each test dataset (same order as test-dirs)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = build_test_transform(args.input_size)

    model = load_model(args.model, args.checkpoint, num_classes=2, device=device)

    print("\n========== Evaluation ==========")
    for idx, test_root in enumerate(args.test_dirs):
        if not os.path.isdir(test_root):
            print(f"[WARN] Test dir not found, skip: {test_root}")
            continue

        dataset = datasets.ImageFolder(root=test_root, transform=transform)
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
        metrics = evaluate_on_dataset(model, dataloader, device=device)

        print(
            f"AUROC: {metrics['AUROC']:.4f}  "
            f"AUPRC: {metrics['AUPRC']:.4f}  "
            f"ACC: {metrics['ACC']:.4f}  "
            f"PREC: {metrics['PREC']:.4f}  "
            f"RECALL: {metrics['RECALL']:.4f}  "
            f"F1: {metrics['F1']:.4f}  "
            f"SPEC: {metrics['SPEC']:.4f}"
        )


if __name__ == "__main__":
    main()
# my_run_5cls 使用说明

本文档说明五分类（5-class）模型的训练、评估、推理导出 JSON 的完整流程。

涉及文件：

| 用途 | Python 脚本 | Shell 脚本 |
|---|---|---|
| 训练 | `train_explicit_paths.py` | `scripts/mytrain_5cls.sh`（需自行创建，见下文） |
| 多测试集评估 | `eval_multi_test_5cls.py` | `scripts/myeval_5cls.sh` |
| 推理导出 JSON | `infer_multi_to_json_5cls.py` | `scripts/myeval_json_5cls.sh` |

> 训练代码无需修改，复用 `train_explicit_paths.py`，类别数从数据目录自动推断。

---

## 1. 数据集目录结构要求

训练和测试数据均需为 `ImageFolder` 格式，每个子目录代表一个类别，**子目录数必须为 5**。子目录名建议用 `0` ~ `4`，确保 `ImageFolder` 生成的类别索引与模型输出维度对应。

```bash
dataset_5_cls/
  train/
    0/
      img_001.png
      img_002.png
    1/
      img_003.png
    2/
      img_004.png
    3/
      img_005.png
    4/
      img_006.png
  test/
    0/
    1/
    2/
    3/
    4/
```

> `ImageFolder` 会按子目录名字母序排序生成类别索引。使用 `0/1/2/3/4` 可保证索引与类别名一致。

---

## 2. 训练五分类模型

### 命令

```bash
python train_explicit_paths.py \
  --model repvit_m1_0 \
  --train-data-path /path/to/dataset_5_cls/train \
  --test-data-path  /path/to/dataset_5_cls/test \
  --batch-size 16 \
  --epochs 40 \
  --dist-eval \
  --output_dir checkpoints/5cls \
  --finetune pretrain/repvit_m1_0_distill_300e.pth \
  --set_bn_eval \
  --distillation-type none \
  --device cuda:0
```

### 关键参数说明

| 参数 | 说明 |
|---|---|
| `--model` | 模型名称，默认 `repvit_m1_0` |
| `--train-data-path` | 训练集目录（ImageFolder，5 个子目录） |
| `--test-data-path` | 验证/测试集目录（ImageFolder，5 个子目录） |
| `--batch-size` | batch size |
| `--epochs` | 训练轮数 |
| `--output_dir` | checkpoint 输出目录 |
| `--finetune` | 预训练权重路径（ImageNet 权重，分类头会因 shape 不匹配被自动丢弃并重新初始化） |
| `--set_bn_eval` | finetune 时冻结 BN 层 |
| `--distillation-type` | 蒸馏类型，`none` 表示不用蒸馏 |
| `--device` | 训练设备 |

### 输出结果

checkpoint 保存到 `checkpoints/5cls/repvit_m1_0_224_<wd>_<lr>_<reprob>/` 下，包含：

- `checkpoint_best.pth`：验证集最高准确率对应的权重
- `checkpoint_<epoch>.pth`：每轮权重（保留最近 4 个）
- `log.txt`：训练日志
- `args.txt`：训练参数

### 可选：封装为 shell 脚本

新建 `scripts/mytrain_5cls.sh`：

```bash
#!/bin/bash

python train_explicit_paths.py \
  --model repvit_m1_0 \
  --train-data-path /path/to/dataset_5_cls/train \
  --test-data-path  /path/to/dataset_5_cls/test \
  --batch-size 16 \
  --epochs 40 \
  --dist-eval \
  --output_dir checkpoints/5cls \
  --finetune pretrain/repvit_m1_0_distill_300e.pth \
  --set_bn_eval \
  --distillation-type none \
  --device cuda:0
```

然后执行：

```bash
bash scripts/mytrain_5cls.sh
```

---

## 3. 五分类评估：`myeval_5cls.sh`

### 命令

```bash
bash scripts/myeval_5cls.sh
```

### 对应 Python 调用

```bash
python3 eval_multi_test_5cls.py \
  --model repvit_m1_0 \
  --checkpoint ./checkpoints/5cls/repvit_m1_0_224_0.025_0.001_0.25/checkpoint_best.pth \
  --test-dirs \
      /path/to/test_set_A \
      /path/to/test_set_B \
  --test-names TestA TestB \
  --num-classes 5 \
  --batch-size 16 \
  --num-workers 4 \
  --input-size 224 \
  --device cuda
```

### 关键参数说明

| 参数 | 说明 |
|---|---|
| `--model` | 模型名称 |
| `--checkpoint` | 五分类模型权重路径 |
| `--test-dirs` | 一个或多个测试集目录（ImageFolder，5 个子目录） |
| `--test-names` | 每个测试集对应的名字，顺序与 `--test-dirs` 一致 |
| `--num-classes` | 类别数，默认 `5` |
| `--batch-size` | 推理 batch size |
| `--num-workers` | 数据加载线程数 |
| `--input-size` | 输入尺寸 |
| `--device` | 推理设备 |
| `--ece-bins` | ECE 分箱数，默认 `10` |
| `--bootstrap-iters` | bootstrap 置信区间迭代次数，默认 `2000` |
| `--ci-alpha` | 置信区间置信水平，默认 `0.95` |

### 输出指标

每个测试集输出以下指标（附 95% bootstrap 置信区间）：

```
AUROC_macro_ovr  ACC  ACC_top2  PREC_macro  RECALL_macro  F1_macro
PREC_weighted  RECALL_weighted  F1_weighted  ECE
```

并输出 `5×5` 混淆矩阵（行=真实类别，列=预测类别）。

### 输出文件

文本日志保存到 `test_log/eval_multi_test_5cls_<checkpoint_stem>_<timestamp>.txt`。

---

## 4. 五分类推理导出 JSON：`myeval_json_5cls.sh`

### 命令

```bash
bash scripts/myeval_json_5cls.sh
```

### 对应 Python 调用

```bash
python3 infer_multi_to_json_5cls.py \
  --model repvit_m1_0 \
  --checkpoint ./checkpoints/5cls/repvit_m1_0_224_0.025_0.001_0.25/checkpoint_best.pth \
  --test-dirs /path/to/test_set \
  --test-names FiveCls \
  --num-classes 5 \
  --batch-size 16 \
  --num-workers 4 \
  --input-size 224 \
  --device cuda \
  --output-dir ./test_log/auroc_json_5cls
```

### 关键参数说明

| 参数 | 说明 |
|---|---|
| `--model` | 模型名称 |
| `--checkpoint` | 五分类模型权重路径 |
| `--test-dirs` | 测试集目录（ImageFolder，5 个子目录） |
| `--test-names` | 测试集名字 |
| `--num-classes` | 类别数，默认 `5` |
| `--output-dir` | JSON 输出目录 |

### 输出文件

每个测试集生成两个文件：

```
<output-dir>/
  <TestName>__<checkpoint_stem>__predictions.json   # 逐样本预测
  <TestName>__<checkpoint_stem>__metrics.json        # 数据集级指标
```

### predictions.json 每条记录字段

```json
{
  "record_type": "sample",
  "image_file": "/abs/path/to/image.png",
  "image_name": "image.png",
  "selected_model": "repvit_m1_0",
  "predicted_class": 2,
  "confidence": 0.8731,
  "true_label": 2,
  "num_classes": 5,
  "prob_class_0": 0.0123,
  "prob_class_1": 0.0456,
  "prob_class_2": 0.8731,
  "prob_class_3": 0.0321,
  "prob_class_4": 0.0369
}
```

- `predicted_class`：`argmax` 预测类别
- `confidence`：最大预测概率
- `prob_class_0` ~ `prob_class_4`：5 个类别的 softmax 概率

### metrics.json 内容

```json
{
  "record_type": "classification_metrics_summary",
  "dataset_name": "FiveCls",
  "model": "repvit_m1_0",
  "num_classes": 5,
  "class_to_idx": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
  "n_samples": 500,
  "label_set": [0, 1, 2, 3, 4],
  "metrics": { "AUROC_macro_ovr": ..., "ACC": ..., ... },
  "confidence_intervals": { "AUROC_macro_ovr": {"lower": ..., "upper": ...}, ... },
  "confusion_matrix": [[...], [...], ...]
}
```

---

## 5. 常见使用流程

```bash
# 1. 训练五分类模型
bash scripts/mytrain_5cls.sh

# 2. 多测试集评估（输出指标 + 混淆矩阵）
bash scripts/myeval_5cls.sh

# 3. 推理导出 JSON（用于后续绘图/多模型对比）
bash scripts/myeval_json_5cls.sh
```

---

## 6. 与二分类脚本的区别

| 项目 | 二分类 | 五分类 |
|---|---|---|
| Python 脚本 | `eval_multi_test.py` / `infer_multi_to_json.py` | `eval_multi_test_5cls.py` / `infer_multi_to_json_5cls.py` |
| Shell 脚本 | `scripts/myeval.sh` / `scripts/myeval_json.sh` | `scripts/myeval_5cls.sh` / `scripts/myeval_json_5cls.sh` |
| `num_classes` | 硬编码 `2` | `--num-classes` 可配（默认 `5`） |
| 概率输出 | `prob_class_0` / `prob_class_1` | `prob_class_0` ~ `prob_class_4` |
| AUROC | 二分类 | `multi_class="ovr", average="macro"` |
| P/R/F1 | `average="binary"` | `macro` + `weighted` |
| 混淆矩阵 | 无 | `5×5` 矩阵 |
| SPEC（特异度） | 有 | 无（多分类不适用单一 SPEC） |
| ACC_top2 | 无 | 有 |
| 训练代码 | 共用 `train_explicit_paths.py` | 共用 `train_explicit_paths.py` |

---

## 7. 说明

- 所有脚本默认在项目根目录执行。
- 训练代码无需修改，类别数从数据目录子目录数自动推断。
- 修改路径时，优先检查数据目录、checkpoint 路径、预训练权重路径是否存在。
- 测试集目录的子目录数必须与 `--num-classes` 一致，否则脚本会输出 `[WARN]` 但仍会继续运行。
- 五分类脚本与二分类脚本相互独立，不会互相影响。

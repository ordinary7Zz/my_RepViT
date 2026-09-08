# my_run 使用说明

本文档说明 `mytrain.sh`、`myeval.sh`、`myeval_json.sh` 三个脚本的用途与使用方式。

## 1. mytrain.sh

对应文件：`mytrain.sh`

### 作用

用于训练 `repvit_m1_0` 分类模型。

### 当前脚本内容

```bash
python main.py \
  --model repvit_m1_0 \
  --data-path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3_cls \
  --batch-size 16 \
  --epochs 40 \
  --dist-eval \
  --output_dir checkpoints \
  --finetune pretrain/repvit_m1_0_distill_300e.pth \
  --set_bn_eval \
  --distillation-type none \
  --device cuda:0
```

### 使用方法

在项目根目录执行：

```bash
bash mytrain.sh
```

### 运行前建议修改的参数

- `--data-path`：训练数据路径
- `--batch-size`：batch size
- `--epochs`：训练轮数
- `--output_dir`：checkpoint 输出目录
- `--finetune`：预训练权重路径
- `--device`：训练设备，例如 `cuda:0`

### 输出结果

训练结果会保存到 `checkpoints` 目录下，包含训练生成的模型权重和日志。

---

## 2. myeval.sh

对应文件：`myeval.sh`

### 作用

用于调用 `eval_multi_test.py`，对一个训练好的模型在多个测试集上进行评估，并输出文本日志。

### 当前脚本内容

```bash
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

MODEL="repvit_m1_0"
CKPT_PATH="./checkpoints/repvit_m1_0/2026_02_26_21_16_50/checkpoint_best.pth"

TEST_DIRS=(
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all_cls"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test_cls"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test_cls"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test_cls"
)

BATCH_SIZE=16
NUM_WORKERS=4
INPUT_SIZE=224
DEVICE="cuda"
```

### 使用方法

在项目根目录执行：

```bash
bash myeval.sh
```

### 运行前建议修改的参数

- `MODEL`：模型名称
- `CKPT_PATH`：待评估模型权重路径
- `TEST_DIRS`：一个或多个测试集目录，要求是 `ImageFolder` 格式
- `--test-names` 对应的测试集名字，顺序应与 `TEST_DIRS` 一致
- `BATCH_SIZE`：推理 batch size
- `NUM_WORKERS`：数据加载线程数
- `INPUT_SIZE`：输入尺寸
- `DEVICE`：推理设备，例如 `cuda` 或 `cpu`

### 输出结果

该脚本调用 `eval_multi_test.py` 后，会在默认的 `test_log` 目录下生成评估日志文件，内容包括各测试集上的指标结果。

### 数据集目录结构要求

`myeval.sh` 使用 `torchvision.datasets.ImageFolder` 读取测试集，因此不需要额外的标签 JSON 文件，但要求每个测试集目录必须是 `ImageFolder` 格式。

二分类任务推荐目录结构如下：

```bash
test_cls/
  0/
    xxx.png
    yyy.png
  1/
    aaa.png
    bbb.png
```

说明：

- 每个子目录名代表一个类别。
- `ImageFolder` 会自动根据子目录名生成类别索引。
- 导出的真实标签来自目录结构本身，而不是外部标注文件。
- 如果数据不是这种目录结构，而是标签保存在 CSV / JSON / TXT 中，则当前脚本不能直接使用，需要额外改读取逻辑。

---

## 3. myeval_json.sh

对应文件：`myeval_json.sh`

### 作用

用于调用 `infer_multi_to_json.py`，对一个训练好的模型在多个测试集上进行推理，并将每个测试集的结果分别保存成一个 JSON 文件。

这些 JSON 文件可直接用于后续 AUROC 绘图流程。

### 当前脚本内容

```bash
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

MODEL="repvit_m1_0"
CKPT_PATH="./checkpoints/repvit_m1_0/2026_02_26_21_16_50/checkpoint_best.pth"
OUTPUT_DIR="./test_log/auroc_json"

TEST_DIRS=(
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all_cls"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test_cls"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test_cls"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test_cls"
)

TEST_NAMES=(DDTI TN3K ThyroidXL TN5K)

BATCH_SIZE=16
NUM_WORKERS=4
INPUT_SIZE=224
DEVICE="cuda"
```

### 使用方法

在项目根目录执行：

```bash
bash myeval_json.sh
```

### 运行前建议修改的参数

- `MODEL`：模型名称
- `CKPT_PATH`：待推理模型权重路径
- `OUTPUT_DIR`：JSON 输出目录
- `TEST_DIRS`：一个或多个测试集目录，要求是 `ImageFolder` 格式
- `TEST_NAMES`：每个测试集对应的名字，顺序必须与 `TEST_DIRS` 一致
- `BATCH_SIZE`：推理 batch size
- `NUM_WORKERS`：数据加载线程数
- `INPUT_SIZE`：输入尺寸
- `DEVICE`：推理设备，例如 `cuda` 或 `cpu`

### 输出结果

运行后会在 `OUTPUT_DIR` 下为每个测试集分别生成一个 JSON 文件，文件名类似：

```bash
DDTI__2026_02_26_21_16_50__predictions.json
TN3K__2026_02_26_21_16_50__predictions.json
ThyroidXL__2026_02_26_21_16_50__predictions.json
TN5K__2026_02_26_21_16_50__predictions.json
```

每个 JSON 文件的顶层是一个列表，列表中的每条记录至少包含：

- `true_label`
- `prob_class_1`

同时还会包含：

- `record_type`
- `image_file`
- `image_name`
- `selected_model`
- `predicted_class`
- `confidence`
- `prob_class_0`

### 数据集目录结构要求

`myeval_json.sh` 调用的 `infer_multi_to_json.py` 同样使用 `torchvision.datasets.ImageFolder` 读取测试集，因此不需要额外的标签 JSON 文件，但要求每个测试集目录必须是 `ImageFolder` 格式。

二分类任务推荐目录结构如下：

```bash
test_cls/
  0/
    xxx.png
    yyy.png
  1/
    aaa.png
    bbb.png
```

说明：

- 每个子目录名代表一个类别。
- `true_label` 来自图片所在的类别目录。
- `prob_class_1` 是模型输出的正类概率。
- 如果你的标签保存在单独的 CSV / JSON / TXT 文件中，而不是按目录分好类，则当前脚本不能直接使用，需要改成“图片路径 + 外部标签文件”的读取方式。

---

## 4. 常见使用流程

### 训练模型

```bash
bash mytrain.sh
```

### 用训练好的模型做多测试集评估

```bash
bash myeval.sh
```

### 导出可用于 AUROC 绘图的 JSON

```bash
bash myeval_json.sh
```

---

## 5. 五分类模型训练与推理

上述脚本（`mytrain.sh`、`myeval.sh`、`myeval_json.sh`）面向**二分类**任务。本节说明如何训练和推理**五分类**模型。

### 5.1 训练五分类模型

训练代码（`main.py` / `train_explicit_paths.py`）天然支持多分类，**无需修改任何 Python 代码**。只需将训练/测试数据按 `ImageFolder` 格式组织为 5 个子目录：

```bash
dataset_5_cls/
  train/
    0/
      xxx.png
    1/
      xxx.png
    2/
      xxx.png
    3/
      xxx.png
    4/
      xxx.png
  test/
    0/
    1/
    2/
    3/
    4/
```

`build_dataset` 会自动从子目录数推断 `nb_classes=5`，模型分类头会按 5 类构建，预训练权重的分类头会因 shape 不匹配被自动丢弃（`main.py:298-302`）。

训练命令示例（复用现有训练脚本，修改路径即可）：

```bash
python train_explicit_paths.py \
  --model repvit_m1_0 \
  --train-data-path /path/to/dataset_5_cls/train \
  --test-data-path /path/to/dataset_5_cls/test \
  --batch-size 16 \
  --epochs 40 \
  --dist-eval \
  --output_dir checkpoints/5cls \
  --finetune pretrain/repvit_m1_0_distill_300e.pth \
  --set_bn_eval \
  --distillation-type none \
  --device cuda:0
```

### 5.2 五分类评估：`myeval_5cls.sh`

对应文件：`scripts/myeval_5cls.sh`

### 作用

调用 `eval_multi_test_5cls.py`，对一个训练好的五分类模型在多个测试集上进行评估，输出多分类指标和混淆矩阵。

### 与二分类版本的区别

| 项目 | 二分类 `eval_multi_test.py` | 五分类 `eval_multi_test_5cls.py` |
|---|---|---|
| `num_classes` | 硬编码 `2` | 可配置（`--num-classes`，默认 `5`） |
| 概率提取 | `softmax(logits)[:, 1]`（正类概率） | 完整 softmax 概率矩阵 `(N, C)` |
| AUROC | 二分类 `roc_auc_score` | 多分类 `roc_auc_score(multi_class="ovr", average="macro")` |
| Precision/Recall/F1 | `average="binary"` | `average="macro"` 和 `"weighted"` |
| 特异度 (SPEC) | 二分类计算 | 移除（多分类不适用单一 SPEC） |
| 混淆矩阵 | 无 | 输出完整 `C x C` 混淆矩阵 |
| ECE | 基于正类概率分箱 | 基于最大预测概率分箱 |
| 新增指标 | — | `ACC_top2`（top-2 准确率） |

### 输出指标列表

```
AUROC_macro_ovr  ACC  ACC_top2  PREC_macro  RECALL_macro  F1_macro
PREC_weighted  RECALL_weighted  F1_weighted  ECE
```

每个指标附带 bootstrap 95% 置信区间，并输出混淆矩阵。

### 使用方法

```bash
bash scripts/myeval_5cls.sh
```

### 运行前需修改的参数

- `MODEL`：模型名称
- `CKPT_PATH`：五分类模型权重路径
- `TEST_DIRS`：测试集目录列表（`ImageFolder` 格式，每个目录需有 5 个子目录）
- `TEST_NAMES`：对应测试集名字
- `NUM_CLASSES`：类别数（默认 `5`，可根据需要修改）
- `BATCH_SIZE` / `NUM_WORKERS` / `INPUT_SIZE` / `DEVICE`

### 输出结果

在 `test_log/` 目录下生成文本日志文件，包含各测试集的指标和混淆矩阵。

---

### 5.3 五分类推理导出 JSON：`myeval_json_5cls.sh`

对应文件：`scripts/myeval_json_5cls.sh`

### 作用

调用 `infer_multi_to_json_5cls.py`，对五分类模型在多个测试集上推理，为每个测试集分别生成 `predictions.json` 和 `metrics.json`。

### JSON 记录字段

每条 sample 记录包含：

- `record_type`：`"sample"`
- `image_file` / `image_name`
- `selected_model`
- `predicted_class`：`argmax` 预测类别
- `confidence`：最大预测概率
- `true_label`：真实标签
- `num_classes`：类别数
- `prob_class_0` ~ `prob_class_4`：每个类别的 softmax 概率

`metrics.json` 包含多分类指标、置信区间和混淆矩阵。

### 使用方法

```bash
bash scripts/myeval_json_5cls.sh
```

### 运行前需修改的参数

- `MODEL`：模型名称
- `CKPT_PATH`：checkpoint 路径数组（支持多个）
- `OUTPUT_DIR`：JSON 输出目录
- `TEST_DIRS`：与 `CKPT_PATH` 一一对应的测试集目录
- `TEST_NAMES`：与 `CKPT_PATH` 一一对应的测试集名字
- `NUM_CLASSES`：类别数（默认 `5`）

### 输出文件

```
<OUTPUT_DIR>/
  <TestName>__<checkpoint_stem>__predictions.json
  <TestName>__<checkpoint_stem>__metrics.json
```

---

## 6. 说明

- 所有脚本都默认在项目根目录执行。
- 修改路径时，建议优先检查数据目录、checkpoint 路径、预训练权重路径是否存在。
- `myeval.sh` / `myeval_5cls.sh` 主要输出文本评估结果。
- `myeval_json.sh` / `myeval_json_5cls.sh` 主要输出每个数据集独立的 JSON 结果文件，适合后续绘图和多模型对比。
- 二分类脚本（`eval_multi_test.py` / `infer_multi_to_json.py`）与五分类脚本（`*_5cls.py`）相互独立，不会互相影响。

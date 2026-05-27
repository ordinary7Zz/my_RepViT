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

## 5. 说明

- 这三个脚本都默认在项目根目录执行。
- 修改路径时，建议优先检查数据目录、checkpoint 路径、预训练权重路径是否存在。
- `myeval.sh` 主要输出文本评估结果。
- `myeval_json.sh` 主要输出每个数据集独立的 JSON 结果文件，适合后续绘图和多模型对比。

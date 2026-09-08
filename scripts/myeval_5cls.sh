#!/usr/bin/env bash

# 五分类模型多测试集评估脚本
# 调用 eval_multi_test_5cls.py，输出多分类指标（AUROC macro、ACC、F1 等）和混淆矩阵

export CUDA_VISIBLE_DEVICES=0

MODEL="repvit_m1_0"
CKPT_PATH="./checkpoints/repvit_m1_0/<timestamp>/checkpoint_best.pth"

# 测试集目录（ImageFolder 格式，每个子目录代表一个类别，子目录数应等于 NUM_CLASSES）
TEST_DIRS=(
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all_cls_5"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test_cls_5"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test_cls_5"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test_cls_5"
)

# 与 TEST_DIRS 一一对应的测试集名字
TEST_NAMES=(DDTI TN3K ThyroidXL TN5K)

NUM_CLASSES=5
BATCH_SIZE=16
NUM_WORKERS=4
INPUT_SIZE=224
DEVICE="cuda"

# 拼接 TEST_DIRS 为一行
TEST_DIRS_ARGS=""
for d in "${TEST_DIRS[@]}"; do
  TEST_DIRS_ARGS="${TEST_DIRS_ARGS} ${d}"
done

echo "Running eval_multi_test_5cls.py ..."
echo "Model: ${MODEL}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Num classes: ${NUM_CLASSES}"
echo "Test dirs:${TEST_DIRS_ARGS}"

python3 eval_multi_test_5cls.py \
  --model "${MODEL}" \
  --checkpoint "${CKPT_PATH}" \
  --test-dirs ${TEST_DIRS_ARGS} \
  --test-names ${TEST_NAMES[@]} \
  --num-classes ${NUM_CLASSES} \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --input-size ${INPUT_SIZE} \
  --device "${DEVICE}"

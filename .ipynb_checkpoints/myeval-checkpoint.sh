#!/usr/bin/env bash

# 可选：指定 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 必填参数：根据你实际情况修改
MODEL="repvit_m1_0"
CKPT_PATH="./checkpoints/repvit_m1_0/2026_02_26_21_16_50/checkpoint_best.pth"

# 可以传入多个测试集目录（ImageFolder 格式）
TEST_DIRS=(
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all_cls"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test_cls"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test_cls"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test_cls"
)

BATCH_SIZE=16
NUM_WORKERS=4
INPUT_SIZE=224
DEVICE="cuda"   # 如果只想用 CPU，可以改成 "cpu"

# 拼接 TEST_DIRS 为一行
TEST_DIRS_ARGS=""
for d in "${TEST_DIRS[@]}"; do
  TEST_DIRS_ARGS="${TEST_DIRS_ARGS} ${d}"
done

echo "Running eval_multi_test.py ..."
echo "Model: ${MODEL}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Test dirs:${TEST_DIRS_ARGS}"

python3 eval_multi_test.py \
  --model "${MODEL}" \
  --checkpoint "${CKPT_PATH}" \
  --test-dirs ${TEST_DIRS_ARGS} \
  --test-names DDTI TN3K ThyroidXL TN5K \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --input-size ${INPUT_SIZE} \
  --device "${DEVICE}"
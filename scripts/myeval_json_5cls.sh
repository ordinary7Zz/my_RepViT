#!/usr/bin/env bash

# 五分类模型多测试集推理导出 JSON 脚本
# 调用 infer_multi_to_json_5cls.py，为每个测试集生成 predictions.json 和 metrics.json
# 每个 sample 记录包含 prob_class_0 ~ prob_class_4 及 argmax 预测

export CUDA_VISIBLE_DEVICES=0

MODEL="repvit_m1_0"

# 可配置多个 checkpoint，与 TEST_DIRS / TEST_NAMES 按下标一一对应
CKPT_PATH=(
  "./checkpoints/repvit_m1_0/<timestamp>/checkpoint_best.pth"
)

OUTPUT_DIR="./test_log/auroc_json_5cls"

# 与 CKPT_PATH 按下标一一对应的测试集目录（ImageFolder 格式）
TEST_DIRS=(
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/500_TestData_Malignancy_Cls/by_5cls_test"
)

# 与 CKPT_PATH 按下标一一对应的测试集名字
TEST_NAMES=(FiveCls)

NUM_CLASSES=5
BATCH_SIZE=16
NUM_WORKERS=4
INPUT_SIZE=224
DEVICE="cuda"

if [ ${#CKPT_PATH[@]} -ne ${#TEST_DIRS[@]} ]; then
  echo "Error: CKPT_PATH and TEST_DIRS must have the same length."
  exit 1
fi

if [ ${#TEST_NAMES[@]} -ne 0 ] && [ ${#TEST_NAMES[@]} -ne ${#CKPT_PATH[@]} ]; then
  echo "Error: TEST_NAMES must be empty or have the same length as CKPT_PATH."
  exit 1
fi

echo "Running infer_multi_to_json_5cls.py ${#CKPT_PATH[@]} times ..."
echo "Model: ${MODEL}"
echo "Num classes: ${NUM_CLASSES}"
echo "Output dir: ${OUTPUT_DIR}"

for i in "${!CKPT_PATH[@]}"; do
  echo "[$((i + 1))/${#CKPT_PATH[@]}] Checkpoint: ${CKPT_PATH[$i]}"
  echo "[$((i + 1))/${#CKPT_PATH[@]}] Test dir: ${TEST_DIRS[$i]}"

  cmd=(
    python3 infer_multi_to_json_5cls.py
    --model "${MODEL}"
    --checkpoint "${CKPT_PATH[$i]}"
    --test-dirs "${TEST_DIRS[$i]}"
    --num-classes ${NUM_CLASSES}
    --batch-size ${BATCH_SIZE}
    --num-workers ${NUM_WORKERS}
    --input-size ${INPUT_SIZE}
    --device "${DEVICE}"
    --output-dir "${OUTPUT_DIR}"
  )

  if [ ${#TEST_NAMES[@]} -ne 0 ]; then
    echo "[$((i + 1))/${#CKPT_PATH[@]}] Test name: ${TEST_NAMES[$i]}"
    cmd+=(--test-names "${TEST_NAMES[$i]}")
  fi

  "${cmd[@]}"
done

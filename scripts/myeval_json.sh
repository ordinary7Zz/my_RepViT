#!/usr/bin/env bash

# 可选：指定 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 必填参数：根据你实际情况修改
MODEL="repvit_m1_0"
CKPT_PATH=(
  "./checkpoints/BM/repvit_m1_0_224_0.025_0.001_0.25/checkpoint_best.pth"
  "./checkpoints/LNMCN01/repvit_m1_0_224_0.025_0.001_0.25/checkpoint_best.pth"
  "./checkpoints/FTCPTC/repvit_m1_0_224_0.025_0.001_0.25/checkpoint_best.pth"
)
OUTPUT_DIR="./test_log/auroc_json"

# 与 CKPT_PATH 按下标一一对应
TEST_DIRS=(
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/500_TestData_Malignancy_Cls/by_malignancy_test"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/by_LNM_CN01_test"
  "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped/by_FTCPTC_test"
)

TEST_NAMES=(BM LNMCN01 FTCPTC)

BATCH_SIZE=16
NUM_WORKERS=4
INPUT_SIZE=224
DEVICE="cuda"   # 如果只想用 CPU，可以改成 "cpu"

if [ ${#CKPT_PATH[@]} -ne ${#TEST_DIRS[@]} ]; then
  echo "Error: CKPT_PATH and TEST_DIRS must have the same length."
  exit 1
fi

if [ ${#TEST_NAMES[@]} -ne 0 ] && [ ${#TEST_NAMES[@]} -ne ${#CKPT_PATH[@]} ]; then
  echo "Error: TEST_NAMES must be empty or have the same length as CKPT_PATH."
  exit 1
fi

echo "Running infer_multi_to_json.py ${#CKPT_PATH[@]} times ..."
echo "Model: ${MODEL}"
echo "Output dir: ${OUTPUT_DIR}"

for i in "${!CKPT_PATH[@]}"; do
  echo "[$((i + 1))/${#CKPT_PATH[@]}] Checkpoint: ${CKPT_PATH[$i]}"
  echo "[$((i + 1))/${#CKPT_PATH[@]}] Test dir: ${TEST_DIRS[$i]}"

  cmd=(
    python3 infer_multi_to_json.py
    --model "${MODEL}"
    --checkpoint "${CKPT_PATH[$i]}"
    --test-dirs "${TEST_DIRS[$i]}"
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

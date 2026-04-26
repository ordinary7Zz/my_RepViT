#!/bin/bash

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
#!/bin/bash

python train_explicit_paths.py \
  --model repvit_m1_0 \
  --train-data-path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3_cls/train \
  --test-data-path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3_cls/test \
  --batch-size 16 \
  --epochs 20 \
  --dist-eval \
  --output_dir checkpoints/BM \
  --finetune pretrain/repvit_m1_0_distill_300e.pth \
  --set_bn_eval \
  --distillation-type none \
  --device cuda:0
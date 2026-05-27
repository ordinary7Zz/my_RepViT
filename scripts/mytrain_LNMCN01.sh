#!/bin/bash

python train_explicit_paths.py \
  --model repvit_m1_0 \
  --train-data-path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/by_LNM_CN01_train \
  --test-data-path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Lymph_Node_Metastasis_fake/by_LNM_CN01_test \
  --batch-size 16 \
  --epochs 10 \
  --dist-eval \
  --output_dir checkpoints/LNMCN01 \
  --finetune pretrain/repvit_m1_0_distill_300e.pth \
  --set_bn_eval \
  --distillation-type none \
  --device cuda:0
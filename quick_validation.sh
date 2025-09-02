#!/bin/bash

# Navigation数据集快速验证脚本
echo "=== Navigation数据集快速验证 ==="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 创建验证目录
mkdir -p output/navigation_validation
mkdir -p log/navigation_validation

# 快速验证训练（使用较小的参数）
echo "开始快速验证训练..."

python main.py \
    --dataname navigation \
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \
    --state train \
    --batch_size 8 \
    --lr 3e-5 \
    --max_spanLen 6 \
    --bert_max_length 256 \
    --iteration 20 \
    --loss ce \
    --etrans_func softmax \
    --model_save_dir results/navigation_validation \
    --logger_dir log/navigation_validation \
    --results_dir output/navigation_validation \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --model_dropout 0.2 \
    --bert_dropout 0.1 \
    --early_stop 5 \
    --clip_grad True \
    --seed 42 \
    --gpu True \
    --optimizer adamw \
    --use_span_weight True \
    --neg_span_weight 0.5

echo "快速验证完成！"
echo "结果保存在: output/navigation_validation"
echo "日志保存在: log/navigation_validation"

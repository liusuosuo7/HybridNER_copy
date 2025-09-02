#!/bin/bash

# Navigation数据集优化训练脚本
echo "=== Navigation数据集优化训练脚本 ==="
echo "目标：将F1值提升到60%以上"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# 清理GPU内存
echo "清理GPU内存..."
nvidia-smi --gpu-reset
sleep 2

# 创建输出目录
mkdir -p output/navigation_optimized
mkdir -p log/navigation_optimized
mkdir -p results/navigation_optimized

# 核心优化参数 - 针对navigation数据集特点
echo "使用优化参数训练..."

python main.py \
    --dataname navigation \
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \
    --state train \
    --batch_size 4 \
    --lr 2e-5 \
    --max_spanLen 8 \
    --bert_max_length 512 \
    --iteration 200 \
    --loss ce \
    --etrans_func softmax \
    --model_save_dir results/navigation_optimized \
    --logger_dir log/navigation_optimized \
    --results_dir output/navigation_optimized \
    --warmup_steps 500 \
    --weight_decay 0.05 \
    --model_dropout 0.4 \
    --bert_dropout 0.3 \
    --early_stop 15 \
    --clip_grad True \
    --seed 42 \
    --gpu True \
    --optimizer adamw \
    --adam_epsilon 1e-8 \
    --final_div_factor 1e3 \
    --warmup_proportion 0.1 \
    --polydecay_ratio 3 \
    --use_span_weight True \
    --neg_span_weight 0.3 \
    --use_tokenLen True \
    --use_spanLen True \
    --use_morph True \
    --classifier_sign multi_nonlinear \
    --classifier_act_func gelu

echo "训练完成！"
echo "结果保存在: output/navigation_optimized"
echo "模型保存在: results/navigation_optimized"
echo "日志保存在: log/navigation_optimized"

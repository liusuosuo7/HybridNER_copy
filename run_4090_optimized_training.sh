#!/bin/bash

# 4090显卡优化训练脚本 - 解决过拟合问题
echo "=== 4090显卡优化训练脚本 ==="
echo "目标：解决过拟合，提高F1值到60%+"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# 清理GPU内存
echo "清理GPU内存..."
nvidia-smi --gpu-reset
sleep 2

# 创建输出目录
mkdir -p output/navigation_4090_optimized
mkdir -p log/navigation_4090_optimized
mkdir -p results/navigation_4090_optimized

# 核心优化参数 - 针对过拟合问题
echo "使用优化参数训练..."

python main.py \
    --dataname navigation \
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \
    --state train \
    --batch_size 8 \
    --lr 5e-6 \
    --max_spanLen 6 \
    --bert_max_length 256 \
    --iteration 150 \
    --loss ce \
    --etrans_func softmax \
    --model_save_dir results/navigation_4090_optimized \
    --logger_dir log/navigation_4090_optimized \
    --results_dir output/navigation_4090_optimized \
    --warmup_steps 200 \
    --weight_decay 0.1 \
    --model_dropout 0.3 \
    --bert_dropout 0.2 \
    --early_stop 10 \
    --clip_grad True \
    --seed 42 \
    --gpu True \
    --optimizer adamw \
    --adam_epsilon 1e-8 \
    --final_div_factor 1e3 \
    --warmup_proportion 0.15 \
    --polydecay_ratio 2 \
    --use_span_weight True \
    --neg_span_weight 0.4 \
    --use_tokenLen True \
    --use_spanLen True \
    --use_morph True \
    --classifier_sign multi_nonlinear \
    --classifier_act_func gelu

echo "训练完成！"
echo "结果保存在: output/navigation_4090_optimized"
echo "模型保存在: results/navigation_4090_optimized"
echo "日志保存在: log/navigation_4090_optimized"

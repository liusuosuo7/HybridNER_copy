#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跳过数据增强，直接使用原始数据
当数据增强遇到内存问题时使用此脚本
"""

import json
import os
import shutil

def skip_data_augmentation():
    """跳过数据增强，直接使用原始数据"""
    
    print("=== 跳过数据增强，使用原始数据 ===")
    
    # 原始数据文件
    original_file = '/root/autodl-tmp/HybridNER/dataset/navigation/navigation_train_span.json'
    augmented_file = '/root/autodl-tmp/HybridNER/dataset/navigation/navigation_train_augmented.json'
    
    if not os.path.exists(original_file):
        print(f"原始训练文件不存在: {original_file}")
        return None
    
    # 直接复制原始文件作为增强文件
    shutil.copy2(original_file, augmented_file)
    
    # 验证文件
    with open(augmented_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"成功复制原始数据作为增强数据")
    print(f"数据样本数: {len(data)}")
    print(f"输出文件: {augmented_file}")
    print(f"增强比例: 1.0x (无增强)")
    
    return augmented_file

def create_optimized_training_script():
    """创建优化的训练脚本，跳过数据增强"""
    
    print("\n=== 创建优化的训练脚本 ===")
    
    script_content = '''#!/bin/bash
# 跳过数据增强的优化训练
echo "=== 跳过数据增强的优化训练 ==="

# 创建输出目录
mkdir -p /root/autodl-tmp/HybridNER/output/navigation_no_augmentation
mkdir -p ./log/navigation_no_augmentation
mkdir -p ./results/navigation_no_augmentation

echo "开始训练（跳过数据增强）..."
CUDA_VISIBLE_DEVICES=0 python main.py \\
    --dataname navigation \\
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \\
    --state train \\
    --batch_size 16 \\
    --lr 2e-5 \\
    --max_spanLen 8 \\
    --bert_max_length 256 \\
    --iteration 80 \\
    --loss ce \\
    --etrans_func softmax \\
    --model_save_dir /root/autodl-tmp/HybridNER/output/navigation_no_augmentation \\
    --logger_dir ./log/navigation_no_augmentation \\
    --results_dir ./results/navigation_no_augmentation \\
    --warmup_steps 300 \\
    --weight_decay 0.01 \\
    --model_dropout 0.1 \\
    --bert_dropout 0.1 \\
    --early_stop 15 \\
    --clip_grad True

echo "训练完成！"
'''
    
    with open("run_no_augmentation.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    os.chmod("run_no_augmentation.sh", 0o755)
    print("已创建: run_no_augmentation.sh")

if __name__ == "__main__":
    print("=== 跳过数据增强方案 ===")
    
    # 跳过数据增强
    skip_data_augmentation()
    
    # 创建优化的训练脚本
    create_optimized_training_script()
    
    print("\n=== 完成 ===")
    print("现在可以直接运行训练:")
    print("bash run_no_augmentation.sh")
    print("\n或者继续其他优化策略:")
    print("bash run_large_model.sh")
    print("bash run_multitask.sh")
    print("bash run_adversarial.sh")

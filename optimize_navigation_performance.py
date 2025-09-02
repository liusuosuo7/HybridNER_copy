#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Navigation数据集性能优化脚本
包含数据增强、实体平衡和模型调优策略
"""

import json
import os
import random
import copy
from collections import Counter, defaultdict
from typing import List, Dict, Any
import re

def analyze_entity_distribution(data_dir: str) -> Dict[str, Any]:
    """分析实体分布情况"""
    
    print("=== 分析实体分布 ===")
    
    train_file = os.path.join(data_dir, "navigation_train_span.json")
    
    if not os.path.exists(train_file):
        print(f"训练文件不存在: {train_file}")
        return {}
    
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计实体类型
    entity_types = Counter()
    entity_lengths = []
    text_lengths = []
    
    for item in data:
        text = item.get('text', '')
        text_lengths.append(len(text))
        
        entities = item.get('entities', [])
        for ent in entities:
            label = ent.get('label', 'unknown')
            entity_types[label] += 1
            
            # 计算实体长度
            start = ent.get('start_offset', ent.get('start', 0))
            end = ent.get('end_offset', ent.get('end', 0))
            entity_lengths.append(end - start)
    
    print(f"总样本数: {len(data)}")
    print(f"总实体数: {sum(entity_types.values())}")
    print(f"实体类型数: {len(entity_types)}")
    print(f"平均文本长度: {sum(text_lengths)/len(text_lengths):.1f}")
    print(f"平均实体长度: {sum(entity_lengths)/len(entity_lengths):.1f}")
    
    print("\n实体类型分布:")
    for ent_type, count in entity_types.most_common():
        percentage = count / sum(entity_types.values()) * 100
        print(f"  {ent_type}: {count} ({percentage:.1f}%)")
    
    return {
        'entity_types': dict(entity_types),
        'avg_text_length': sum(text_lengths)/len(text_lengths),
        'avg_entity_length': sum(entity_lengths)/len(entity_lengths),
        'total_samples': len(data)
    }

def create_balanced_dataset(data_dir: str, target_ratio: float = 0.3) -> None:
    """创建平衡的数据集，减少高频实体类型的权重"""
    
    print("\n=== 创建平衡数据集 ===")
    
    train_file = os.path.join(data_dir, "navigation_train_span.json")
    
    if not os.path.exists(train_file):
        print(f"训练文件不存在: {train_file}")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计实体类型
    entity_types = Counter()
    for item in data:
        entities = item.get('entities', [])
        for ent in entities:
            label = ent.get('label', 'unknown')
            entity_types[label] += 1
    
    # 计算每个实体类型的权重
    total_entities = sum(entity_types.values())
    entity_weights = {}
    
    for ent_type, count in entity_types.items():
        # 使用反比例权重，高频实体权重降低
        weight = (total_entities / count) ** target_ratio
        entity_weights[ent_type] = weight
    
    print("实体类型权重:")
    for ent_type, weight in sorted(entity_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ent_type}: {weight:.3f}")
    
    # 创建平衡后的数据
    balanced_data = []
    
    for item in data:
        entities = item.get('entities', [])
        if not entities:
            # 保留无实体的样本
            balanced_data.append(item)
            continue
        
        # 计算样本权重（基于包含的实体类型）
        sample_weight = 0
        for ent in entities:
            label = ent.get('label', 'unknown')
            sample_weight += entity_weights.get(label, 1.0)
        
        # 根据权重决定是否保留样本
        if random.random() < min(sample_weight / len(entities), 1.0):
            balanced_data.append(item)
    
    print(f"原始样本数: {len(data)}")
    print(f"平衡后样本数: {len(balanced_data)}")
    print(f"保留率: {len(balanced_data)/len(data)*100:.1f}%")
    
    # 保存平衡后的数据
    balanced_file = os.path.join(data_dir, "navigation_train_balanced.json")
    with open(balanced_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"平衡数据集已保存到: {balanced_file}")

def create_data_augmentation(data_dir: str) -> None:
    """创建数据增强版本"""
    
    print("\n=== 创建数据增强版本 ===")
    
    train_file = os.path.join(data_dir, "navigation_train_span.json")
    
    if not os.path.exists(train_file):
        print(f"训练文件不存在: {train_file}")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    augmented_data = []
    
    for item in data:
        # 添加原始样本
        augmented_data.append(item)
        
        entities = item.get('entities', [])
        if len(entities) >= 2:
            # 对于包含多个实体的样本，创建部分实体版本
            for i in range(len(entities)):
                new_item = copy.deepcopy(item)
                new_item['entities'] = [entities[i]]
                new_item['id'] = f"{item['id']}_aug_{i}"
                augmented_data.append(new_item)
    
    print(f"原始样本数: {len(data)}")
    print(f"增强后样本数: {len(augmented_data)}")
    print(f"增强倍数: {len(augmented_data)/len(data):.1f}x")
    
    # 保存增强后的数据
    augmented_file = os.path.join(data_dir, "navigation_train_augmented.json")
    with open(augmented_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    print(f"增强数据集已保存到: {augmented_file}")

def create_optimized_training_config() -> None:
    """创建优化的训练配置"""
    
    print("\n=== 创建优化训练配置 ===")
    
    config_content = '''#!/bin/bash

# Navigation数据集超优化训练脚本
echo "=== Navigation数据集超优化训练 ==="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 创建输出目录
mkdir -p output/navigation_super_optimized
mkdir -p log/navigation_super_optimized
mkdir -p results/navigation_super_optimized

# 超优化参数配置
python main.py \\
    --dataname navigation \\
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \\
    --state train \\
    --batch_size 2 \\
    --lr 1e-5 \\
    --max_spanLen 10 \\
    --bert_max_length 512 \\
    --iteration 300 \\
    --loss ce \\
    --etrans_func softmax \\
    --model_save_dir results/navigation_super_optimized \\
    --logger_dir log/navigation_super_optimized \\
    --results_dir output/navigation_super_optimized \\
    --warmup_steps 1000 \\
    --weight_decay 0.1 \\
    --model_dropout 0.5 \\
    --bert_dropout 0.4 \\
    --early_stop 20 \\
    --clip_grad True \\
    --seed 42 \\
    --gpu True \\
    --optimizer adamw \\
    --adam_epsilon 1e-8 \\
    --final_div_factor 1e4 \\
    --warmup_proportion 0.15 \\
    --polydecay_ratio 2 \\
    --use_span_weight True \\
    --neg_span_weight 0.2 \\
    --use_tokenLen True \\
    --use_spanLen True \\
    --use_morph True \\
    --classifier_sign multi_nonlinear \\
    --classifier_act_func gelu

echo "超优化训练完成！"
'''
    
    with open("run_navigation_super_optimized.sh", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("已创建超优化训练脚本: run_navigation_super_optimized.sh")

def create_ensemble_training_script() -> None:
    """创建集成训练脚本"""
    
    print("\n=== 创建集成训练脚本 ===")
    
    ensemble_content = '''#!/bin/bash

# Navigation数据集集成训练脚本
echo "=== Navigation数据集集成训练 ==="

# 训练多个模型进行集成

# 模型1: 基础配置
echo "训练模型1..."
python main.py \\
    --dataname navigation \\
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \\
    --state train \\
    --batch_size 4 \\
    --lr 2e-5 \\
    --max_spanLen 8 \\
    --bert_max_length 512 \\
    --iteration 150 \\
    --model_save_dir results/navigation_ensemble_1 \\
    --logger_dir log/navigation_ensemble_1 \\
    --results_dir output/navigation_ensemble_1 \\
    --seed 42

# 模型2: 高dropout配置
echo "训练模型2..."
python main.py \\
        --dataname navigation \\
        --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
        --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \\
        --state train \\
    --batch_size 4 \\
    --lr 1.5e-5 \\
    --max_spanLen 10 \\
    --bert_max_length 512 \\
    --iteration 150 \\
    --model_dropout 0.6 \\
    --bert_dropout 0.5 \\
    --model_save_dir results/navigation_ensemble_2 \\
    --logger_dir log/navigation_ensemble_2 \\
    --results_dir output/navigation_ensemble_2 \\
    --seed 123

# 模型3: 低学习率配置
echo "训练模型3..."
python main.py \\
    --dataname navigation \\
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \\
    --state train \\
    --batch_size 4 \\
    --lr 1e-5 \\
    --max_spanLen 6 \\
    --bert_max_length 512 \\
    --iteration 200 \\
    --model_save_dir results/navigation_ensemble_3 \\
    --logger_dir log/navigation_ensemble_3 \\
    --results_dir output/navigation_ensemble_3 \\
    --seed 456

echo "集成训练完成！"
'''
    
    with open("run_navigation_ensemble.sh", "w", encoding="utf-8") as f:
        f.write(ensemble_content)
    
    print("已创建集成训练脚本: run_navigation_ensemble.sh")

def main():
    """主函数"""
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    
    print("=== Navigation数据集性能优化 ===")
    
    # 1. 分析实体分布
    stats = analyze_entity_distribution(data_dir)
    
    # 2. 创建平衡数据集
    create_balanced_dataset(data_dir)
    
    # 3. 创建数据增强版本
    create_data_augmentation(data_dir)
    
    # 4. 创建优化训练配置
    create_optimized_training_config()
    
    # 5. 创建集成训练脚本
    create_ensemble_training_script()
    
    print("\n=== 优化完成 ===")
    print("已创建以下文件:")
    print("1. navigation_train_balanced.json - 平衡数据集")
    print("2. navigation_train_augmented.json - 增强数据集")
    print("3. run_navigation_super_optimized.sh - 超优化训练脚本")
    print("4. run_navigation_ensemble.sh - 集成训练脚本")
    print("\n建议按以下顺序尝试:")
    print("1. 先运行: bash run_navigation_optimized.sh")
    print("2. 如果效果不理想，运行: bash run_navigation_super_optimized.sh")
    print("3. 最后尝试集成: bash run_navigation_ensemble.sh")

if __name__ == "__main__":
    main()

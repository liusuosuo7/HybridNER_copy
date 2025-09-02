#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门解决过拟合问题的训练脚本
使用更强的正则化和数据增强技术
"""

import os
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import logging
from datetime import datetime

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(f'log/anti_overfitting_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_navigation_data(data_dir):
    """加载navigation数据集"""
    logger = logging.getLogger(__name__)
    
    train_file = os.path.join(data_dir, "navigation_train_span.json")
    dev_file = os.path.join(data_dir, "dev", "navigation_dev_span.json")
    test_file = os.path.join(data_dir, "test", "navigation_test_span.json")
    
    datasets = {}
    
    # 加载训练集
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            datasets['train'] = json.load(f)
        logger.info(f"训练集加载成功: {len(datasets['train'])} 样本")
    
    # 加载验证集
    if os.path.exists(dev_file):
        with open(dev_file, 'r', encoding='utf-8') as f:
            datasets['dev'] = json.load(f)
        logger.info(f"验证集加载成功: {len(datasets['dev'])} 样本")
    
    # 加载测试集
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            datasets['test'] = json.load(f)
        logger.info(f"测试集加载成功: {len(datasets['test'])} 样本")
    
    return datasets

def analyze_entity_distribution(datasets):
    """分析实体分布"""
    logger = logging.getLogger(__name__)
    
    entity_counts = {}
    total_entities = 0
    
    for split_name, data in datasets.items():
        entity_counts[split_name] = {}
        for item in data:
            if 'entities' in item:
                for entity in item['entities']:
                    if 'type' in entity:
                        entity_type = entity['type']
                        if entity_type not in entity_counts[split_name]:
                            entity_counts[split_name][entity_type] = 0
                        entity_counts[split_name][entity_type] += 1
                        total_entities += 1
    
    logger.info(f"总实体数: {total_entities}")
    for split_name, counts in entity_counts.items():
        logger.info(f"{split_name}集实体分布:")
        for entity_type, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {entity_type}: {count}")
    
    return entity_counts

def create_balanced_training_data(datasets, entity_counts):
    """创建平衡的训练数据"""
    logger = logging.getLogger(__name__)
    
    # 计算每个实体类型的目标数量
    target_count = 100  # 每个实体类型至少100个样本
    
    balanced_data = []
    entity_samples = {}
    
    # 初始化每个实体类型的样本列表
    for entity_type in entity_counts['train'].keys():
        entity_samples[entity_type] = []
    
    # 按实体类型分组样本
    for item in datasets['train']:
        if 'entities' in item:
            for entity in item['entities']:
                if 'type' in entity:
                    entity_type = entity['type']
                    if entity_type in entity_samples:
                        entity_samples[entity_type].append(item)
    
    # 平衡采样
    for entity_type, samples in entity_samples.items():
        if len(samples) > target_count:
            # 随机采样
            selected_samples = random.sample(samples, target_count)
        else:
            # 如果样本不足，重复采样
            selected_samples = samples * (target_count // len(samples) + 1)
            selected_samples = selected_samples[:target_count]
        
        balanced_data.extend(selected_samples)
        logger.info(f"{entity_type}: 从 {len(samples)} 个样本中选择 {len(selected_samples)} 个")
    
    # 打乱数据
    random.shuffle(balanced_data)
    
    logger.info(f"平衡后的训练数据: {len(balanced_data)} 样本")
    return balanced_data

def apply_data_augmentation(data, augmentation_factor=2):
    """应用数据增强"""
    logger = logging.getLogger(__name__)
    
    augmented_data = []
    
    for item in data:
        # 原始样本
        augmented_data.append(item)
        
        # 增强样本
        for _ in range(augmentation_factor - 1):
            augmented_item = item.copy()
            
            # 简单的文本增强：随机替换一些字符
            if 'context' in augmented_item:
                text = augmented_item['context']
                if len(text) > 10:
                    # 随机替换1-3个字符
                    num_replacements = random.randint(1, 3)
                    for _ in range(num_replacements):
                        if len(text) > 1:
                            pos = random.randint(0, len(text) - 1)
                            # 保持中文字符不变，只替换标点符号
                            if text[pos] in '，。！？；：""''（）【】':
                                new_chars = '，。！？；：""''（）【】'
                                text = text[:pos] + random.choice(new_chars) + text[pos+1:]
                
                augmented_item['context'] = text
            
            augmented_data.append(augmented_item)
    
    logger.info(f"数据增强完成: {len(data)} -> {len(augmented_data)} 样本")
    return augmented_data

def save_enhanced_data(data, output_dir, filename):
    """保存增强后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logging.getLogger(__name__).info(f"数据已保存到: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="反过拟合训练数据准备")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/HybridNER/dataset/navigation", 
                       help="数据集目录")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/HybridNER/dataset/navigation_enhanced", 
                       help="增强数据输出目录")
    parser.add_argument("--augmentation_factor", type=int, default=2, 
                       help="数据增强倍数")
    parser.add_argument("--seed", type=int, default=42, 
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设置日志
    logger = setup_logging()
    logger.info("=== 反过拟合训练数据准备 ===")
    
    # 加载数据
    datasets = load_navigation_data(args.data_dir)
    if not datasets:
        logger.error("无法加载数据集")
        return
    
    # 分析实体分布
    entity_counts = analyze_entity_distribution(datasets)
    
    # 创建平衡的训练数据
    balanced_train_data = create_balanced_training_data(datasets, entity_counts)
    
    # 应用数据增强
    enhanced_train_data = apply_data_augmentation(balanced_train_data, args.augmentation_factor)
    
    # 保存增强后的数据
    save_enhanced_data(enhanced_train_data, args.output_dir, "navigation_train_enhanced.json")
    
    # 保存验证集和测试集
    if 'dev' in datasets:
        save_enhanced_data(datasets['dev'], args.output_dir, "navigation_dev_enhanced.json")
    if 'test' in datasets:
        save_enhanced_data(datasets['test'], args.output_dir, "navigation_test_enhanced.json")
    
    logger.info("=== 数据准备完成 ===")
    logger.info(f"增强后的训练数据: {len(enhanced_train_data)} 样本")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 创建训练脚本
    create_training_script(args.output_dir)

def create_training_script(output_dir):
    """创建训练脚本"""
    training_script = f"""#!/bin/bash

# 反过拟合训练脚本
echo "=== 反过拟合训练脚本 ==="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# 清理GPU内存
echo "清理GPU内存..."
nvidia-smi --gpu-reset
sleep 2

# 创建输出目录
mkdir -p output/navigation_anti_overfitting
mkdir -p log/navigation_anti_overfitting
mkdir -p results/navigation_anti_overfitting

# 反过拟合训练参数
echo "使用反过拟合参数训练..."

python main.py \\
    --dataname navigation \\
    --data_dir {output_dir} \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \\
    --state train \\
    --batch_size 6 \\
    --lr 3e-6 \\
    --max_spanLen 6 \\
    --bert_max_length 256 \\
    --iteration 200 \\
    --loss ce \\
    --etrans_func softmax \\
    --model_save_dir results/navigation_anti_overfitting \\
    --logger_dir log/navigation_anti_overfitting \\
    --results_dir output/navigation_anti_overfitting \\
    --warmup_steps 300 \\
    --weight_decay 0.15 \\
    --model_dropout 0.4 \\
    --bert_dropout 0.3 \\
    --early_stop 15 \\
    --clip_grad True \\
    --seed 42 \\
    --gpu True \\
    --optimizer adamw \\
    --adam_epsilon 1e-8 \\
    --final_div_factor 1e2 \\
    --warmup_proportion 0.2 \\
    --polydecay_ratio 1.5 \\
    --use_span_weight True \\
    --neg_span_weight 0.3 \\
    --use_tokenLen True \\
    --use_spanLen True \\
    --use_morph True \\
    --classifier_sign multi_nonlinear \\
    --classifier_act_func gelu

echo "训练完成！"
echo "结果保存在: output/navigation_anti_overfitting"
echo "模型保存在: results/navigation_anti_overfitting"
echo "日志保存在: log/navigation_anti_overfitting"
"""
    
    script_path = "run_anti_overfitting_training.sh"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    os.chmod(script_path, 0o755)
    logging.getLogger(__name__).info(f"训练脚本已创建: {script_path}")

if __name__ == "__main__":
    main()

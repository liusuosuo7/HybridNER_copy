#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复Navigation数据集问题
使用所有实体类型，确保标签映射正确
"""

import json
import os
import shutil
from collections import Counter
from datetime import datetime

def analyze_and_fix_dataset(data_dir):
    """分析并修复数据集"""
    
    print("=== 分析并修复Navigation数据集 ===")
    
    # 1. 分析训练集获取所有实体类型
    train_file = os.path.join(data_dir, "navigation_train_span.json")
    
    if not os.path.exists(train_file):
        print(f"训练文件不存在: {train_file}")
        return
    
    print("分析训练集中的实体分布...")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 统计实体类型
    entity_types = Counter()
    total_entities = 0
    
    for item in train_data:
        entities = item.get('entities', [])
        total_entities += len(entities)
        for ent in entities:
            label = ent.get('label', 'unknown')
            entity_types[label] += 1
    
    print(f"总实体数量: {total_entities}")
    print(f"实体类型数量: {len(entity_types)}")
    
    # 获取所有实体类型（按数量排序）
    all_entities = [ent_type for ent_type, count in entity_types.most_common()]
    
    print("\n所有实体类型分布:")
    for i, (ent_type, count) in enumerate(entity_types.most_common(), 1):
        percentage = count / total_entities * 100
        print(f"  {i}. {ent_type}: {count} ({percentage:.1f}%)")
    
    return all_entities

def backup_original_dataset(data_dir):
    """备份原有数据集"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{data_dir}_backup_{timestamp}"
    
    print(f"\n备份原有数据集到: {backup_dir}")
    
    # 创建备份目录
    os.makedirs(backup_dir, exist_ok=True)
    
    # 复制所有文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            src_file = os.path.join(data_dir, filename)
            dst_file = os.path.join(backup_dir, filename)
            shutil.copy2(src_file, dst_file)
            print(f"  备份: {filename}")
    
    return backup_dir

def create_entity_mapping(all_entities):
    """创建实体类型映射"""
    
    # 创建标签映射
    label2idx = {"O": 0}
    for i, entity_type in enumerate(all_entities, 1):
        label2idx[entity_type] = i
    
    print(f"\n新的实体类型映射 (共{len(label2idx)}个标签):")
    for label, idx in label2idx.items():
        print(f"  {label}: {idx}")
    
    return label2idx

def update_model_config(label2idx, all_entities):
    """更新模型配置文件中的标签映射"""
    
    print(f"\n更新模型配置文件...")
    
    # 更新 dataloaders/spanner_dataset.py 中的标签映射
    spanner_file = "dataloaders/spanner_dataset.py"
    
    if os.path.exists(spanner_file):
        with open(spanner_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 构建新的标签映射字符串
        new_mapping_lines = ['        label2idx = {']
        new_mapping_lines.append('            "O": 0,')
        for entity_type in label2idx.keys():
            if entity_type != "O":
                new_mapping_lines.append(f'            "{entity_type}": {label2idx[entity_type]},')
        new_mapping_lines.append('        }')
        new_mapping = '\n'.join(new_mapping_lines)
        
        # 替换现有的navigation标签映射
        import re
        pattern = r'elif args\.dataname == \'navigation\':\s*\n\s*#.*?\n\s*label2idx = \{[\s\S]*?\n\s*\}'
        replacement = f"elif args.dataname == 'navigation':\n        # Navigation数据集标签映射（完整实体类型）\n{new_mapping}"
        
        new_content = re.sub(pattern, replacement, content)
        
        # 备份原文件
        backup_file = f"{spanner_file}.backup"
        shutil.copy2(spanner_file, backup_file)
        print(f"  备份原文件到: {backup_file}")
        
        # 写入新内容
        with open(spanner_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  已更新 {spanner_file}")
    else:
        print(f"  文件不存在: {spanner_file}")

def verify_dataset_integrity(data_dir, all_entities):
    """验证数据集完整性"""
    
    print(f"\n验证数据集完整性...")
    
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        filename = f"navigation_{split}_span.json"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  跳过: {filepath} (不存在)")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计实体
        entity_count = 0
        entity_types_in_split = Counter()
        
        for item in data:
            entities = item.get('entities', [])
            entity_count += len(entities)
            for ent in entities:
                label = ent.get('label', 'unknown')
                entity_types_in_split[label] += 1
        
        print(f"  {split}集: {len(data)}个样本, {entity_count}个实体, {len(entity_types_in_split)}种类型")
        
        # 检查是否有不在映射中的实体类型
        unknown_types = set(entity_types_in_split.keys()) - set(all_entities)
        if unknown_types:
            print(f"    警告: 发现未知实体类型: {unknown_types}")

def create_optimized_training_script():
    """创建优化的训练脚本"""
    
    print(f"\n创建优化的训练脚本...")
    
    script_content = '''#!/bin/bash
# Navigation数据集优化训练脚本（修复版）
echo "=== Navigation数据集优化训练（修复版） ==="
echo "使用完整实体类型映射"

# 创建输出目录
mkdir -p /root/autodl-tmp/HybridNER/output/navigation_fixed
mkdir -p ./log/navigation_fixed
mkdir -p ./results/navigation_fixed

echo "开始训练..."
CUDA_VISIBLE_DEVICES=0 python main.py \\
    --dataname navigation \\
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-base-chinese \\
    --state train \\
    --batch_size 16 \\
    --lr 2e-5 \\
    --max_spanLen 8 \\
    --bert_max_length 256 \\
    --iteration 50 \\
    --loss ce \\
    --etrans_func softmax \\
    --model_save_dir /root/autodl-tmp/HybridNER/output/navigation_fixed \\
    --logger_dir ./log/navigation_fixed \\
    --results_dir ./results/navigation_fixed \\
    --warmup_steps 200 \\
    --weight_decay 0.01 \\
    --model_dropout 0.1 \\
    --bert_dropout 0.1 \\
    --early_stop 10 \\
    --clip_grad True

echo "训练完成！"
'''
    
    with open("run_navigation_fixed.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod("run_navigation_fixed.sh", 0o755)
    
    print("  已创建: run_navigation_fixed.sh")

def main():
    """主函数"""
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    
    print("=== 修复Navigation数据集问题 ===")
    
    # 1. 分析数据集并获取所有实体类型
    all_entities = analyze_and_fix_dataset(data_dir)
    
    if not all_entities:
        print("无法获取实体类型信息，退出")
        return
    
    # 2. 备份原有数据集
    backup_dir = backup_original_dataset(data_dir)
    
    # 3. 创建新的实体映射
    label2idx = create_entity_mapping(all_entities)
    
    # 4. 更新模型配置
    update_model_config(label2idx, all_entities)
    
    # 5. 验证数据集完整性
    verify_dataset_integrity(data_dir, all_entities)
    
    # 6. 创建优化的训练脚本
    create_optimized_training_script()
    
    print("\n=== 修复完成 ===")
    print(f"1. 原有数据集已备份到: {backup_dir}")
    print(f"2. 使用所有 {len(all_entities)} 种实体类型")
    print(f"3. 已更新模型配置文件")
    print(f"4. 已创建优化训练脚本: run_navigation_fixed.sh")
    print(f"5. 现在可以运行: bash run_navigation_fixed.sh")
    
    print(f"\n主要修复:")
    print(f"- 使用完整实体类型映射而不是限制数量")
    print(f"- 确保标签映射与数据集一致")
    print(f"- 验证数据集完整性")

if __name__ == "__main__":
    main()

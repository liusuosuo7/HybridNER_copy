#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试Navigation数据集问题
分析为什么指标都是0
"""

import json
import os
from collections import Counter

def analyze_dataset_structure(data_dir):
    """分析数据集结构"""
    
    print("=== 分析Navigation数据集结构 ===")
    
    for split in ['train', 'dev', 'test']:
        filename = f"navigation_{split}_span.json"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            continue
        
        print(f"\n--- {split} 集分析 ---")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"样本数量: {len(data)}")
        
        if len(data) > 0:
            # 分析第一个样本的结构
            first_sample = data[0]
            print(f"第一个样本的键: {list(first_sample.keys())}")
            
            # 检查文本字段
            text_fields = ['context', 'text', 'sentence']
            for field in text_fields:
                if field in first_sample:
                    text = first_sample[field]
                    print(f"文本字段 '{field}': {text[:100]}...")
            
            # 检查实体字段
            if 'entities' in first_sample:
                entities = first_sample['entities']
                print(f"实体数量: {len(entities)}")
                if entities:
                    print(f"第一个实体: {entities[0]}")
                    
                    # 统计实体标签
                    labels = [ent.get('label', 'unknown') for ent in entities]
                    print(f"实体标签: {labels}")
            
            # 检查span_posLabel字段
            if 'span_posLabel' in first_sample:
                span_posLabel = first_sample['span_posLabel']
                print(f"span_posLabel类型: {type(span_posLabel)}")
                if span_posLabel:
                    print(f"span_posLabel示例: {list(span_posLabel.items())[:3]}")

def analyze_entity_distribution(data_dir):
    """分析实体分布"""
    
    print("\n=== 分析实体分布 ===")
    
    train_file = os.path.join(data_dir, "navigation_train_span.json")
    
    if not os.path.exists(train_file):
        print(f"训练文件不存在: {train_file}")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计实体类型
    entity_types = Counter()
    total_entities = 0
    
    for item in data:
        entities = item.get('entities', [])
        total_entities += len(entities)
        for ent in entities:
            label = ent.get('label', 'unknown')
            entity_types[label] += 1
    
    print(f"总实体数量: {total_entities}")
    print(f"实体类型数量: {len(entity_types)}")
    
    print("\n实体类型分布:")
    for i, (ent_type, count) in enumerate(entity_types.most_common(), 1):
        percentage = count / total_entities * 100
        print(f"  {i}. {ent_type}: {count} ({percentage:.1f}%)")

def check_label_mapping():
    """检查标签映射"""
    
    print("\n=== 检查标签映射 ===")
    
    # 读取当前的标签映射
    spanner_file = "dataloaders/spanner_dataset.py"
    
    if os.path.exists(spanner_file):
        with open(spanner_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找navigation的标签映射
        import re
        pattern = r'elif args\.dataname == \'navigation\':\s*\n\s*#.*?\n\s*label2idx = \{[\s\S]*?\n\s*\}'
        match = re.search(pattern, content)
        
        if match:
            print("找到navigation标签映射:")
            print(match.group())
        else:
            print("未找到navigation标签映射")
    else:
        print(f"文件不存在: {spanner_file}")

def check_model_config():
    """检查模型配置"""
    
    print("\n=== 检查模型配置 ===")
    
    # 检查args_config.py
    args_file = "args_config.py"
    if os.path.exists(args_file):
        with open(args_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'navigation' in content:
            print("✓ navigation在args_config.py中已配置")
        else:
            print("✗ navigation在args_config.py中未配置")

def create_fixed_dataset(data_dir):
    """创建修复后的数据集"""
    
    print("\n=== 创建修复后的数据集 ===")
    
    # 分析训练集获取所有实体类型
    train_file = os.path.join(data_dir, "navigation_train_span.json")
    
    if not os.path.exists(train_file):
        print(f"训练文件不存在: {train_file}")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 统计实体类型
    entity_types = Counter()
    for item in train_data:
        entities = item.get('entities', [])
        for ent in entities:
            label = ent.get('label', 'unknown')
            entity_types[label] += 1
    
    # 获取所有实体类型（不限制数量）
    all_entities = [ent_type for ent_type, count in entity_types.most_common()]
    
    print(f"发现 {len(all_entities)} 种实体类型:")
    for i, (ent_type, count) in enumerate(entity_types.most_common(), 1):
        print(f"  {i}. {ent_type}: {count}")
    
    # 创建新的标签映射
    label2idx = {"O": 0}
    for i, entity_type in enumerate(all_entities, 1):
        label2idx[entity_type] = i
    
    print(f"\n新的标签映射:")
    for label, idx in label2idx.items():
        print(f"  {label}: {idx}")
    
    # 更新模型配置
    update_model_config(label2idx, all_entities)
    
    return all_entities

def update_model_config(label2idx, all_entities):
    """更新模型配置文件"""
    
    print("\n=== 更新模型配置 ===")
    
    # 更新 dataloaders/spanner_dataset.py
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
        import shutil
        backup_file = f"{spanner_file}.backup"
        shutil.copy2(spanner_file, backup_file)
        print(f"备份原文件到: {backup_file}")
        
        # 写入新内容
        with open(spanner_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"已更新 {spanner_file}")

def main():
    """主函数"""
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    
    print("=== Navigation数据集调试 ===")
    
    # 1. 分析数据集结构
    analyze_dataset_structure(data_dir)
    
    # 2. 分析实体分布
    analyze_entity_distribution(data_dir)
    
    # 3. 检查标签映射
    check_label_mapping()
    
    # 4. 检查模型配置
    check_model_config()
    
    # 5. 创建修复后的配置
    all_entities = create_fixed_dataset(data_dir)
    
    print("\n=== 调试完成 ===")
    print("建议:")
    print("1. 使用所有实体类型而不是限制数量")
    print("2. 确保标签映射正确")
    print("3. 重新训练模型")

if __name__ == "__main__":
    main()

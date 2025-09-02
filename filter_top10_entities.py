#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
保留数量排名前十的实体类型
备份原有数据集并创建新的数据集
"""

import json
import os
import shutil
from collections import Counter
from datetime import datetime

def analyze_entity_distribution(data_dir):
    """分析训练集中的实体分布，找出排名前十的实体类型"""
    
    train_file = os.path.join(data_dir, "navigation_train_span.json")
    
    if not os.path.exists(train_file):
        print(f"训练文件不存在: {train_file}")
        return []
    
    print("分析训练集中的实体分布...")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计实体类型
    entity_types = Counter()
    
    for item in data:
        entities = item.get('entities', [])
        for ent in entities:
            label = ent.get('label', 'unknown')
            entity_types[label] += 1
    
    # 获取排名前十的实体类型
    top_10_entities = [ent_type for ent_type, count in entity_types.most_common(10)]
    
    print("排名前十的实体类型:")
    for i, (ent_type, count) in enumerate(entity_types.most_common(10), 1):
        print(f"  {i}. {ent_type}: {count}")
    
    print(f"\n总共发现 {len(entity_types)} 种实体类型，保留前10种")
    
    return top_10_entities

def backup_original_dataset(data_dir):
    """备份原有数据集"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{data_dir}_backup_{timestamp}"
    
    print(f"备份原有数据集到: {backup_dir}")
    
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

def filter_entities_by_type(data, top_10_entities):
    """根据实体类型过滤数据"""
    
    filtered_data = []
    
    for item in data:
        # 获取文本
        text = item.get('context', item.get('text', ''))
        
        # 过滤实体，只保留排名前十的类型
        entities = item.get('entities', [])
        filtered_entities = []
        
        for ent in entities:
            label = ent.get('label', 'unknown')
            if label in top_10_entities:
                filtered_entities.append(ent)
        
        # 如果过滤后还有实体，则保留这个样本
        if filtered_entities:
            new_item = item.copy()
            new_item['entities'] = filtered_entities
            filtered_data.append(new_item)
    
    return filtered_data

def process_dataset(data_dir, top_10_entities):
    """处理数据集，保留排名前十的实体类型"""
    
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        filename = f"navigation_{split}_span.json"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"文件不存在，跳过: {filepath}")
            continue
        
        print(f"\n处理 {split} 集...")
        
        # 读取原始数据
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  原始样本数: {len(data)}")
        
        # 统计原始实体分布
        original_entities = Counter()
        for item in data:
            entities = item.get('entities', [])
            for ent in entities:
                label = ent.get('label', 'unknown')
                original_entities[label] += 1
        
        print(f"  原始实体类型数: {len(original_entities)}")
        
        # 过滤数据
        filtered_data = filter_entities_by_type(data, top_10_entities)
        
        print(f"  过滤后样本数: {len(filtered_data)}")
        
        # 统计过滤后实体分布
        filtered_entities = Counter()
        for item in filtered_data:
            entities = item.get('entities', [])
            for ent in entities:
                label = ent.get('label', 'unknown')
                filtered_entities[label] += 1
        
        print(f"  过滤后实体类型数: {len(filtered_entities)}")
        
        # 保存过滤后的数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"  已保存过滤后的数据到: {filepath}")
        
        # 显示过滤效果
        print(f"  保留率: {len(filtered_data)/len(data)*100:.1f}%")
        print(f"  实体保留率: {sum(filtered_entities.values())/sum(original_entities.values())*100:.1f}%")

def create_entity_mapping(top_10_entities):
    """创建新的实体类型映射"""
    
    # 创建标签映射
    label2idx = {"O": 0}
    for i, entity_type in enumerate(top_10_entities, 1):
        label2idx[entity_type] = i
    
    print("\n新的实体类型映射:")
    for label, idx in label2idx.items():
        print(f"  {label}: {idx}")
    
    return label2idx

def update_model_config(label2idx):
    """更新模型配置文件中的标签映射"""
    
    # 更新 dataloaders/spanner_dataset.py 中的标签映射
    spanner_file = "dataloaders/spanner_dataset.py"
    
    if os.path.exists(spanner_file):
        print(f"\n更新 {spanner_file} 中的标签映射...")
        
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
        pattern = r'elif args\.dataname == \'navigation\':\s*\n\s*# Navigation数据集标签映射（细粒度分类）\s*\n\s*label2idx = \{[\s\S]*?\n\s*\}'
        replacement = f"elif args.dataname == 'navigation':\n        # Navigation数据集标签映射（Top10实体类型）\n{new_mapping}"
        
        new_content = re.sub(pattern, replacement, content)
        
        # 备份原文件
        backup_file = f"{spanner_file}.backup"
        shutil.copy2(spanner_file, backup_file)
        print(f"  备份原文件到: {backup_file}")
        
        # 写入新内容
        with open(spanner_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  已更新 {spanner_file}")

def main():
    """主函数"""
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    
    print("=== 保留数量排名前十的实体类型 ===")
    
    # 1. 分析实体分布
    top_10_entities = analyze_entity_distribution(data_dir)
    
    if not top_10_entities:
        print("无法获取实体分布信息，退出")
        return
    
    # 2. 备份原有数据集
    backup_dir = backup_original_dataset(data_dir)
    print(f"备份完成: {backup_dir}")
    
    # 3. 处理数据集
    process_dataset(data_dir, top_10_entities)
    
    # 4. 创建新的实体映射
    label2idx = create_entity_mapping(top_10_entities)
    
    # 5. 更新模型配置
    update_model_config(label2idx)
    
    print("\n=== 处理完成 ===")
    print(f"1. 原有数据集已备份到: {backup_dir}")
    print(f"2. 已保留排名前十的实体类型: {', '.join(top_10_entities)}")
    print(f"3. 已更新模型配置文件")
    print(f"4. 现在可以使用优化后的数据集进行训练")

if __name__ == "__main__":
    main()

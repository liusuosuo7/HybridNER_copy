#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Navigation数据集分析脚本
分析数据集特征，帮助优化模型性能
"""

import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_dataset():
    """分析数据集特征"""
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    
    print("=== Navigation数据集分析 ===")
    
    for split in ['train', 'dev', 'test']:
        filename = f"navigation_{split}_span.json"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            continue
            
        print(f"\n--- 分析 {split} 集 ---")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 基本统计
        print(f"样本数量: {len(data)}")
        
        # 文本长度统计
        text_lengths = []
        entity_counts = []
        entity_types = Counter()
        entity_lengths = []
        
        for item in data:
            # 获取文本
            text = item.get('context', item.get('text', ''))
            text_lengths.append(len(text))
            
            # 统计实体
            entities = item.get('entities', [])
            entity_counts.append(len(entities))
            
            for ent in entities:
                label = ent.get('label', 'unknown')
                entity_types[label] += 1
                
                # 计算实体长度
                start = ent.get('start_offset', ent.get('start', 0))
                end = ent.get('end_offset', ent.get('end', 0))
                entity_lengths.append(end - start)
        
        print(f"平均文本长度: {sum(text_lengths)/len(text_lengths):.1f}")
        print(f"最大文本长度: {max(text_lengths)}")
        print(f"最小文本长度: {min(text_lengths)}")
        print(f"平均实体数量: {sum(entity_counts)/len(entity_counts):.1f}")
        
        if entity_lengths:
            print(f"平均实体长度: {sum(entity_lengths)/len(entity_lengths):.1f}")
            print(f"最大实体长度: {max(entity_lengths)}")
            print(f"最小实体长度: {min(entity_lengths)}")
        
        print(f"实体类型分布:")
        for ent_type, count in entity_types.most_common():
            print(f"  {ent_type}: {count}")
        
        # 计算实体密度
        total_chars = sum(text_lengths)
        total_entities = sum(entity_counts)
        entity_density = total_entities / total_chars * 100
        print(f"实体密度: {entity_density:.2f}%")
        
        # 分析实体重叠情况
        overlapping_entities = 0
        for item in data:
            entities = item.get('entities', [])
            if len(entities) > 1:
                # 检查是否有重叠的实体
                for i in range(len(entities)):
                    for j in range(i+1, len(entities)):
                        ent1 = entities[i]
                        ent2 = entities[j]
                        start1 = ent1.get('start_offset', ent1.get('start', 0))
                        end1 = ent1.get('end_offset', ent1.get('end', 0))
                        start2 = ent2.get('start_offset', ent2.get('start', 0))
                        end2 = ent2.get('end_offset', ent2.get('end', 0))
                        
                        if not (end1 <= start2 or end2 <= start1):
                            overlapping_entities += 1
        
        print(f"重叠实体对数量: {overlapping_entities}")

def analyze_entity_distribution():
    """分析实体分布情况"""
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    train_file = os.path.join(data_dir, "navigation_train_span.json")
    
    if not os.path.exists(train_file):
        print("训练文件不存在")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计每个实体类型的长度分布
    entity_length_by_type = {}
    
    for item in data:
        entities = item.get('entities', [])
        for ent in entities:
            label = ent.get('label', 'unknown')
            start = ent.get('start_offset', ent.get('start', 0))
            end = ent.get('end_offset', ent.get('end', 0))
            length = end - start
            
            if label not in entity_length_by_type:
                entity_length_by_type[label] = []
            entity_length_by_type[label].append(length)
    
    print("\n=== 各实体类型的长度分布 ===")
    for ent_type, lengths in entity_length_by_type.items():
        avg_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        min_len = min(lengths)
        print(f"{ent_type}: 平均{avg_len:.1f}, 最大{max_len}, 最小{min_len}, 数量{len(lengths)}")

def suggest_optimizations():
    """基于分析结果提出优化建议"""
    
    print("\n=== 优化建议 ===")
    print("1. 标签策略:")
    print("   - 使用细粒度标签分类而不是统一的'Entity'")
    print("   - 根据实体长度分布调整max_spanLen参数")
    
    print("\n2. 模型参数优化:")
    print("   - 降低学习率到2e-5，提高收敛稳定性")
    print("   - 增加max_spanLen到8-12，捕获更长实体")
    print("   - 增加bert_max_length到256-512")
    print("   - 减小batch_size到16，提高训练稳定性")
    
    print("\n3. 训练策略:")
    print("   - 增加训练轮数到50轮")
    print("   - 增加早停耐心到10轮")
    print("   - 使用梯度裁剪防止梯度爆炸")
    print("   - 降低dropout到0.1，减少正则化")
    
    print("\n4. 数据处理:")
    print("   - 检查实体标注的一致性")
    print("   - 处理重叠实体问题")
    print("   - 确保字符级对齐正确")

if __name__ == "__main__":
    analyze_dataset()
    analyze_entity_distribution()
    suggest_optimizations()

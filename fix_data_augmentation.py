#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
内存优化的数据增强脚本
解决内存溢出问题，使用流式处理和分批处理
"""

import json
import random
import os
import gc
from collections import Counter

def synonym_replacement(text, entities, synonym_dict):
    """同义词替换增强 - 内存优化版本"""
    augmented_samples = []
    
    for entity in entities:
        label = entity.get('label', '')
        if label in synonym_dict:
            synonyms = synonym_dict[label]
            for synonym in synonyms[:2]:  # 减少到最多2个同义词
                new_text = text.replace(entity.get('text', ''), synonym)
                new_entities = []
                for ent in entities:
                    if ent != entity:
                        new_entities.append(ent)
                    else:
                        new_ent = ent.copy()
                        new_ent['text'] = synonym
                        new_entities.append(new_ent)
                
                augmented_samples.append({
                    'context': new_text,
                    'entities': new_entities
                })
                
                # 限制增强样本数量，避免内存溢出
                if len(augmented_samples) >= 1:
                    break
    
    return augmented_samples

def entity_replacement(text, entities, entity_templates):
    """实体替换增强 - 内存优化版本"""
    augmented_samples = []
    
    # 只使用第一个模板，减少内存使用
    if entity_templates:
        template = entity_templates[0]
        new_text = text
        new_entities = []
        
        for entity in entities:
            label = entity.get('label', '')
            if label in template:
                replacement = random.choice(template[label])
                new_text = new_text.replace(entity.get('text', ''), replacement)
                new_ent = entity.copy()
                new_ent['text'] = replacement
                new_entities.append(new_ent)
            else:
                new_entities.append(entity)
        
        if new_entities:
            augmented_samples.append({
                'context': new_text,
                'entities': new_entities
            })
    
    return augmented_samples

def process_batch(data_batch, synonym_dict, entity_templates, output_file, mode='a'):
    """处理一批数据并直接写入文件"""
    
    batch_augmented = []
    
    for item in data_batch:
        # 原始数据
        batch_augmented.append(item)
        
        # 同义词替换
        entities = item.get('entities', [])
        if entities:
            synonym_samples = synonym_replacement(item.get('context', ''), entities, synonym_dict)
            batch_augmented.extend(synonym_samples)
            
            # 实体替换
            entity_samples = entity_replacement(item.get('context', ''), entities, entity_templates)
            batch_augmented.extend(entity_samples)
    
    # 直接写入文件，避免在内存中累积
    with open(output_file, mode, encoding='utf-8') as f:
        for i, sample in enumerate(batch_augmented):
            if mode == 'w' and i == 0:
                f.write('[\n')
            elif mode == 'a' and i == 0:
                f.write('[\n')
            else:
                f.write(',\n')
            json.dump(sample, f, ensure_ascii=False, indent=2)
    
    return len(batch_augmented)

def create_augmented_dataset():
    """创建增强数据集 - 内存优化版本"""
    
    # 读取原始数据
    data_file = '/root/autodl-tmp/HybridNER/dataset/navigation/navigation_train_span.json'
    
    if not os.path.exists(data_file):
        print(f"训练文件不存在: {data_file}")
        return None
    
    # 同义词字典
    synonym_dict = {
        '对抗国家': ['敌对国家', '对手国家'],
        '硬摧毁武器': ['硬杀伤武器', '摧毁性武器'],
        '全球定位系统': ['GPS系统', '定位系统'],
        '北斗卫星导航系统': ['北斗系统', '北斗导航'],
        '压制式干扰技术': ['压制干扰', '干扰技术'],
        '对抗地点': ['敌对地点', '对抗区域'],
        '格洛纳斯卫星导航系统': ['格洛纳斯系统', '俄罗斯导航系统'],
        '伽利略卫星导航系统': ['伽利略系统', '欧洲导航系统'],
        '系统端防御技术': ['系统防御', '防御技术'],
        '压制式干扰装备': ['压制装备', '干扰装备'],
        '对抗单位': ['敌对单位', '对手单位']
    }
    
    # 实体模板 - 简化版本
    entity_templates = [
        {
            '对抗国家': ['美国', '日本'],
            '对抗地点': ['南海', '东海'],
            '硬摧毁武器': ['导弹', '鱼雷']
        }
    ]
    
    # 输出文件
    output_file = '/root/autodl-tmp/HybridNER/dataset/navigation/navigation_train_augmented.json'
    
    print("开始内存优化的数据增强...")
    
    # 分批处理数据
    batch_size = 50  # 减小批次大小
    total_processed = 0
    total_augmented = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据样本数: {len(data)}")
    print(f"批次大小: {batch_size}")
    
    # 清空输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('')
    
    # 分批处理
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        print(f"处理批次 {batch_num}/{total_batches} (样本 {i+1}-{min(i+batch_size, len(data))})")
        
        # 处理当前批次
        mode = 'w' if i == 0 else 'a'
        augmented_count = process_batch(batch, synonym_dict, entity_templates, output_file, mode)
        
        total_processed += len(batch)
        total_augmented += augmented_count
        
        # 强制垃圾回收
        gc.collect()
        
        # 显示进度
        progress = (i + batch_size) / len(data) * 100
        print(f"进度: {progress:.1f}% | 已处理: {total_processed}/{len(data)} | 增强样本: {total_augmented}")
    
    # 完成JSON文件
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n]')
    
    print(f"\n数据增强完成！")
    print(f"原始样本: {len(data)}")
    print(f"增强后样本: {total_augmented}")
    print(f"增强比例: {total_augmented / len(data):.2f}x")
    print(f"输出文件: {output_file}")
    
    return output_file

def create_simple_augmentation():
    """创建简单的数据增强版本 - 最小内存使用"""
    
    data_file = '/root/autodl-tmp/HybridNER/dataset/navigation/navigation_train_span.json'
    output_file = '/root/autodl-tmp/HybridNER/dataset/navigation/navigation_train_augmented.json'
    
    if not os.path.exists(data_file):
        print(f"训练文件不存在: {data_file}")
        return None
    
    print("开始简单数据增强（最小内存使用）...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据样本数: {len(data)}")
    
    # 简单的数据增强：只复制原始数据并添加少量变体
    augmented_data = []
    
    for i, item in enumerate(data):
        # 原始数据
        augmented_data.append(item)
        
        # 每10个样本添加一个简单的增强版本
        if i % 10 == 0 and item.get('entities'):
            # 简单的文本替换
            context = item.get('context', '')
            entities = item.get('entities', [])
            
            if entities:
                # 随机选择一个实体进行简单替换
                entity = random.choice(entities)
                if entity.get('text') and len(entity.get('text', '')) > 1:
                    # 简单的字符替换
                    old_text = entity.get('text')
                    new_text = old_text[0] + 'X' + old_text[1:] if len(old_text) > 1 else old_text
                    
                    new_context = context.replace(old_text, new_text, 1)
                    new_entities = []
                    for ent in entities:
                        if ent == entity:
                            new_ent = ent.copy()
                            new_ent['text'] = new_text
                            new_entities.append(new_ent)
                        else:
                            new_entities.append(ent)
                    
                    augmented_data.append({
                        'context': new_context,
                        'entities': new_entities
                    })
        
        # 显示进度
        if (i + 1) % 100 == 0:
            print(f"处理进度: {i + 1}/{len(data)}")
            gc.collect()  # 强制垃圾回收
    
    # 保存增强数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n简单数据增强完成！")
    print(f"原始样本: {len(data)}")
    print(f"增强后样本: {len(augmented_data)}")
    print(f"增强比例: {len(augmented_data) / len(data):.2f}x")
    print(f"输出文件: {output_file}")
    
    return output_file

if __name__ == "__main__":
    print("=== Navigation数据集数据增强 ===")
    print("内存优化版本：解决内存溢出问题")
    
    try:
        # 尝试内存优化的数据增强
        print("尝试内存优化的数据增强...")
        create_augmented_dataset()
    except Exception as e:
        print(f"内存优化版本失败: {e}")
        print("尝试简单数据增强...")
        try:
            create_simple_augmentation()
        except Exception as e2:
            print(f"简单数据增强也失败: {e2}")
            print("建议跳过数据增强，直接进行模型训练")
    
    print("数据增强处理完成！")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全面修复0指标问题
深入诊断和修复所有可能的问题
"""

import json
import os
import re
from collections import Counter

def analyze_dataset_structure():
    """深入分析数据集结构"""
    print("=== 深入分析数据集结构 ===")
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(data_dir, f"navigation_{split}_span.json")
        
        if os.path.exists(file_path):
            print(f"\n分析 {split} 数据集:")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  样本数量: {len(data)}")
            
            # 检查前3个样本的详细结构
            for i, item in enumerate(data[:3]):
                print(f"  样本 {i+1} 结构:")
                print(f"    字段: {list(item.keys())}")
                
                # 检查文本字段
                text_fields = ['context', 'text', 'sentence']
                for field in text_fields:
                    if field in item:
                        text = item[field]
                        print(f"    {field}: {text[:50]}... (长度: {len(text)})")
                
                # 检查实体字段
                if 'entities' in item:
                    entities = item['entities']
                    print(f"    实体数量: {len(entities)}")
                    for j, entity in enumerate(entities[:2]):
                        print(f"      实体{j+1}: {entity}")
                
                # 检查span_posLabel字段
                if 'span_posLabel' in item:
                    span_labels = item['span_posLabel']
                    print(f"    span_posLabel: {span_labels}")
                
                print()

def check_label_consistency():
    """检查标签一致性"""
    print("\n=== 检查标签一致性 ===")
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    all_labels = set()
    label_sources = {}
    
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(data_dir, f"navigation_{split}_span.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                # 从entities中收集标签
                if 'entities' in item:
                    for entity in item['entities']:
                        if 'label' in entity:
                            label = entity['label']
                            all_labels.add(label)
                            if label not in label_sources:
                                label_sources[label] = []
                            label_sources[label].append('entities')
                
                # 从span_posLabel中收集标签
                if 'span_posLabel' in item:
                    for span, label in item['span_posLabel'].items():
                        all_labels.add(label)
                        if label not in label_sources:
                            label_sources[label] = []
                        label_sources[label].append('span_posLabel')
    
    print(f"发现的所有标签: {sorted(all_labels)}")
    print(f"标签来源统计: {label_sources}")

def fix_dataset_format():
    """修复数据集格式"""
    print("\n=== 修复数据集格式 ===")
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(data_dir, f"navigation_{split}_span.json")
        backup_path = file_path + ".backup"
        
        if os.path.exists(file_path):
            print(f"修复 {split} 数据集...")
            
            # 备份原文件
            if not os.path.exists(backup_path):
                os.system(f"cp {file_path} {backup_path}")
                print(f"  已备份: {backup_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            fixed_data = []
            fixed_count = 0
            
            for item in data:
                fixed_item = item.copy()
                
                # 确保有context字段
                if 'context' not in fixed_item:
                    if 'text' in fixed_item:
                        fixed_item['context'] = fixed_item['text']
                    elif 'sentence' in fixed_item:
                        fixed_item['context'] = fixed_item['sentence']
                    else:
                        print(f"  警告: 样本缺少文本字段")
                        continue
                
                # 确保有span_posLabel字段
                if 'span_posLabel' not in fixed_item and 'entities' in fixed_item:
                    span_posLabel = {}
                    entities = fixed_item['entities']
                    
                    for entity in entities:
                        if 'start_offset' in entity and 'end_offset' in entity and 'label' in entity:
                            start = entity['start_offset']
                            end = entity['end_offset']
                            label = entity['label']
                            
                            # 确保是整数
                            try:
                                start = int(start)
                                end = int(end)
                                span_posLabel[f"{start};{end}"] = label
                            except:
                                continue
                    
                    fixed_item['span_posLabel'] = span_posLabel
                    fixed_count += 1
                
                fixed_data.append(fixed_item)
            
            # 保存修复后的数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(fixed_data, f, ensure_ascii=False, indent=2)
            
            print(f"  修复完成: {fixed_count} 个样本被修复")

def update_model_config():
    """更新模型配置"""
    print("\n=== 更新模型配置 ===")
    
    # 分析数据集中的实际标签
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    all_labels = set()
    
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(data_dir, f"navigation_{split}_span.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if 'span_posLabel' in item:
                    for span, label in item['span_posLabel'].items():
                        all_labels.add(label)
    
    if not all_labels:
        print("✗ 未找到任何标签，使用默认标签")
        all_labels = {'O', '对抗单位', '对抗国家', '对抗时间', '对抗地点'}
    
    # 创建标签映射
    label2idx = {'O': 0}
    for i, label in enumerate(sorted(all_labels - {'O'}), 1):
        label2idx[label] = i
    
    print(f"标签映射: {label2idx}")
    
    # 更新配置文件
    config_file = "dataloaders/spanner_dataset.py"
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 备份原文件
        backup_file = config_file + ".backup2"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  已备份: {backup_file}")
        
        # 查找并替换navigation的标签映射
        old_pattern = r'elif args\.dataname == \'navigation\':\s*\n\s*# Navigation数据集标签映射.*?\n\s*label2idx = \{[^}]+\}'
        
        new_mapping = f'''elif args.dataname == 'navigation':
        # Navigation数据集标签映射（自动生成）
        label2idx = {label2idx}'''
        
        if re.search(old_pattern, content, re.DOTALL):
            new_content = re.sub(old_pattern, new_mapping, content, flags=re.DOTALL)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("  已更新配置文件")
        else:
            print("  未找到navigation标签映射，手动添加")
            # 在合适的位置添加
            insert_pos = content.find("elif args.dataname == 'cmeee':")
            if insert_pos != -1:
                before = content[:insert_pos]
                after = content[insert_pos:]
                new_content = before + new_mapping + "\n    " + after
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print("  已手动添加标签映射")

def create_minimal_test():
    """创建最小化测试"""
    print("\n=== 创建最小化测试 ===")
    
    test_script = '''#!/bin/bash
# 最小化测试
echo "=== 最小化测试 ==="

# 创建输出目录
mkdir -p /root/autodl-tmp/HybridNER/output/navigation_minimal_test
mkdir -p ./log/navigation_minimal_test
mkdir -p ./results/navigation_minimal_test

# 清理GPU内存
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# 最小化训练测试
CUDA_VISIBLE_DEVICES=0 python main.py \\
    --dataname navigation \\
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-base-chinese \\
    --state train \\
    --batch_size 4 \\
    --lr 5e-5 \\
    --max_spanLen 3 \\
    --bert_max_length 64 \\
    --iteration 10 \\
    --loss ce \\
    --etrans_func softmax \\
    --model_save_dir /root/autodl-tmp/HybridNER/output/navigation_minimal_test \\
    --logger_dir ./log/navigation_minimal_test \\
    --results_dir ./results/navigation_minimal_test \\
    --warmup_steps 50 \\
    --weight_decay 0.01 \\
    --model_dropout 0.1 \\
    --bert_dropout 0.1 \\
    --early_stop 5 \\
    --clip_grad True \\
    --seed 42

echo "最小化测试完成！"
'''
    
    with open("run_minimal_test.sh", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    os.chmod("run_minimal_test.sh", 0o755)
    print("  已创建最小化测试脚本: run_minimal_test.sh")

def create_debug_evaluation():
    """创建调试评估脚本"""
    print("\n=== 创建调试评估脚本 ===")
    
    debug_script = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试评估过程
"""

import json
import os
import torch
import numpy as np

def debug_evaluation():
    """调试评估过程"""
    print("=== 调试评估过程 ===")
    
    # 检查数据集
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(data_dir, f"navigation_{split}_span.json")
        
        if os.path.exists(file_path):
            print(f"\\n检查 {split} 数据集:")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 统计实体
            entity_count = 0
            label_count = {}
            
            for item in data:
                if 'span_posLabel' in item:
                    for span, label in item['span_posLabel'].items():
                        entity_count += 1
                        label_count[label] = label_count.get(label, 0) + 1
            
            print(f"  实体总数: {entity_count}")
            print(f"  标签分布: {label_count}")
            
            # 检查前几个样本
            for i, item in enumerate(data[:2]):
                print(f"  样本 {i+1}:")
                if 'span_posLabel' in item:
                    print(f"    span_posLabel: {item['span_posLabel']}")
                else:
                    print(f"    无span_posLabel")

def check_model_output():
    """检查模型输出"""
    print("\\n=== 检查模型输出 ===")
    
    # 检查最近的日志
    log_dirs = [
        "./log/navigation_minimal_test",
        "./log/navigation_fixed",
        "./log/navigation_large_model"
    ]
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
            
            for log_file in log_files:
                log_path = os.path.join(log_dir, log_file)
                print(f"\\n检查日志: {log_path}")
                
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # 查找关键信息
                    for line in lines[-20:]:
                        if 'precision: 0.0000' in line or 'F1: 0.0000' in line:
                            print(f"  发现0指标: {line.strip()}")
                        elif 'loss:' in line:
                            print(f"  损失值: {line.strip()}")
                        elif 'step:' in line:
                            print(f"  训练步数: {line.strip()}")
                except Exception as e:
                    print(f"  读取日志失败: {e}")

if __name__ == "__main__":
    debug_evaluation()
    check_model_output()
'''
    
    with open("debug_evaluation.py", "w", encoding="utf-8") as f:
        f.write(debug_script)
    
    print("  已创建调试评估脚本: debug_evaluation.py")

def main():
    """主函数"""
    print("=== 全面修复0指标问题 ===")
    
    # 1. 深入分析数据集结构
    analyze_dataset_structure()
    
    # 2. 检查标签一致性
    check_label_consistency()
    
    # 3. 修复数据集格式
    fix_dataset_format()
    
    # 4. 更新模型配置
    update_model_config()
    
    # 5. 创建最小化测试
    create_minimal_test()
    
    # 6. 创建调试评估脚本
    create_debug_evaluation()
    
    print("\n=== 全面修复完成 ===")
    print("请按以下步骤操作:")
    print("1. 运行调试评估: python debug_evaluation.py")
    print("2. 运行最小化测试: bash run_minimal_test.sh")
    print("3. 如果最小化测试成功，再运行完整训练")
    print("4. 监控训练过程，检查指标是否正常")

if __name__ == "__main__":
    main()


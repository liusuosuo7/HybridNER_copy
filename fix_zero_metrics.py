#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复0指标问题
分析数据集并更新标签映射
"""

import json
import os
from collections import Counter

def analyze_dataset_labels():
    """分析数据集中的实际标签"""
    print("=== 分析数据集标签 ===")
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    all_entities = set()
    
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(data_dir, f"navigation_{split}_span.json")
        
        if os.path.exists(file_path):
            print(f"\n分析 {split} 数据集:")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entity_count = 0
            split_entities = set()
            
            for item in data:
                if 'entities' in item:
                    for entity in item['entities']:
                        if 'label' in entity:
                            all_entities.add(entity['label'])
                            split_entities.add(entity['label'])
                            entity_count += 1
            
            print(f"  实体数量: {entity_count}")
            print(f"  实体类型: {sorted(split_entities)}")
        else:
            print(f"✗ 文件不存在: {file_path}")
    
    print(f"\n所有数据集中的实体类型: {sorted(all_entities)}")
    return sorted(all_entities)

def update_label_mapping(all_entities):
    """更新标签映射"""
    print("\n=== 更新标签映射 ===")
    
    # 创建新的标签映射
    label2idx = {'O': 0}
    for i, entity in enumerate(all_entities, 1):
        label2idx[entity] = i
    
    print(f"新的标签映射: {label2idx}")
    
    # 更新配置文件
    config_file = "dataloaders/spanner_dataset.py"
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 备份原文件
        backup_file = config_file + ".backup"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 已备份原文件: {backup_file}")
        
        # 更新navigation的标签映射
        import re
        
        # 查找navigation的标签映射部分
        pattern = r'(elif args\.dataname == \'navigation\':\s*\n\s*# Navigation数据集标签映射.*?\n\s*label2idx = \{[^}]+\})'
        
        new_mapping = f'''elif args.dataname == 'navigation':
        # Navigation数据集标签映射（自动生成）
        label2idx = {label2idx}'''
        
        if re.search(pattern, content, re.DOTALL):
            new_content = re.sub(pattern, new_mapping, content, flags=re.DOTALL)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("✓ 已更新配置文件")
        else:
            print("✗ 未找到navigation标签映射位置，手动替换")
            # 手动替换
            old_mapping = '''elif args.dataname == 'navigation':
        # Navigation数据集标签映射（细粒度分类）
        label2idx = {
            "O": 0,
            "对抗国家": 1,
            "对抗时间": 2,
            "对抗地点": 3,
            "硬摧毁武器": 4,
            "压制式干扰技术": 5,
            "全球定位系统": 6,
            "北斗卫星导航系统": 7,
            "格洛纳斯卫星导航系统": 8,
            "伽利略卫星导航系统": 9,
            "系统端防御技术": 10,
            "压制式干扰装备": 11,
            "对抗单位": 12
        }'''
            
            if old_mapping in content:
                new_content = content.replace(old_mapping, new_mapping)
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print("✓ 已手动更新配置文件")
            else:
                print("✗ 无法找到要替换的内容")
    else:
        print(f"✗ 配置文件不存在: {config_file}")

def create_debug_script():
    """创建调试脚本"""
    print("\n=== 创建调试脚本 ===")
    
    debug_script = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试脚本：检查数据处理过程
"""

import json
import os

def debug_data_loading():
    """调试数据加载过程"""
    print("=== 调试数据加载 ===")
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    file_path = os.path.join(data_dir, "navigation_train_span.json")
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"数据集大小: {len(data)}")
        
        # 检查前几个样本
        for i, item in enumerate(data[:3]):
            print(f"\\n样本 {i+1}:")
            print(f"  文本: {item.get('context', item.get('text', 'N/A'))[:100]}...")
            
            if 'entities' in item:
                print(f"  实体数量: {len(item['entities'])}")
                for j, entity in enumerate(item['entities'][:3]):
                    print(f"    实体{j+1}: {entity}")
            else:
                print("  无实体信息")
            
            if 'span_posLabel' in item:
                print(f"  span_posLabel: {item['span_posLabel']}")
            else:
                print("  无span_posLabel")

if __name__ == "__main__":
    debug_data_loading()
'''
    
    with open("debug_data.py", "w", encoding="utf-8") as f:
        f.write(debug_script)
    
    print("✓ 已创建调试脚本: debug_data.py")

def create_optimized_training_script():
    """创建优化训练脚本"""
    print("\n=== 创建优化训练脚本 ===")
    
    training_script = '''#!/bin/bash
# 修复后的优化训练
echo "=== 修复后的优化训练 ==="

# 创建输出目录
mkdir -p /root/autodl-tmp/HybridNER/output/navigation_fixed
mkdir -p ./log/navigation_fixed
mkdir -p ./results/navigation_fixed

# 清理GPU内存
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# 运行训练
CUDA_VISIBLE_DEVICES=0 python main.py \\
    --dataname navigation \\
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \\
    --state train \\
    --batch_size 12 \\
    --lr 1e-5 \\
    --max_spanLen 8 \\
    --bert_max_length 256 \\
    --iteration 80 \\
    --loss ce \\
    --etrans_func softmax \\
    --model_save_dir /root/autodl-tmp/HybridNER/output/navigation_fixed \\
    --logger_dir ./log/navigation_fixed \\
    --results_dir ./results/navigation_fixed \\
    --warmup_steps 400 \\
    --weight_decay 0.02 \\
    --model_dropout 0.2 \\
    --bert_dropout 0.2 \\
    --early_stop 20 \\
    --clip_grad True \\
    --seed 42

echo "训练完成！"
'''
    
    with open("run_fixed_training.sh", "w", encoding="utf-8") as f:
        f.write(training_script)
    
    os.chmod("run_fixed_training.sh", 0o755)
    print("✓ 已创建训练脚本: run_fixed_training.sh")

def main():
    """主函数"""
    print("=== 修复0指标问题 ===")
    
    # 分析数据集标签
    all_entities = analyze_dataset_labels()
    
    if not all_entities:
        print("✗ 未找到任何实体标签，请检查数据集")
        return
    
    # 更新标签映射
    update_label_mapping(all_entities)
    
    # 创建调试脚本
    create_debug_script()
    
    # 创建优化训练脚本
    create_optimized_training_script()
    
    print("\n=== 修复完成 ===")
    print("请按以下步骤操作:")
    print("1. 运行调试脚本: python debug_data.py")
    print("2. 运行修复后的训练: bash run_fixed_training.sh")
    print("3. 监控训练过程，检查指标是否正常")

if __name__ == "__main__":
    main()

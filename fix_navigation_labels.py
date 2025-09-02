#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复navigation数据集的标签映射问题
确保所有实体类型都被正确识别和处理
"""

import json
import os
from collections import Counter
import shutil
from datetime import datetime


def analyze_navigation_dataset(data_dir):
    """分析navigation数据集，获取所有实体类型"""
    print("=== 分析navigation数据集 ===")

    # 数据集文件路径
    train_file = os.path.join(data_dir, "navigation_train_span.json")
    dev_file = os.path.join(data_dir, "dev", "navigation_dev_span.json")
    test_file = os.path.join(data_dir, "test", "navigation_test_span.json")

    all_entities = set()
    entity_counts = Counter()

    # 分析训练集
    if os.path.exists(train_file):
        print(f"分析训练集: {train_file}")
        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            if 'entities' in item:
                for entity in item['entities']:
                    if 'type' in entity:
                        entity_type = entity['type']
                        all_entities.add(entity_type)
                        entity_counts[entity_type] += 1

    # 分析验证集
    if os.path.exists(dev_file):
        print(f"分析验证集: {dev_file}")
        with open(dev_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            if 'entities' in item:
                for entity in item['entities']:
                    if 'type' in entity:
                        entity_type = entity['type']
                        all_entities.add(entity_type)
                        entity_counts[entity_type] += 1

    # 分析测试集
    if os.path.exists(test_file):
        print(f"分析测试集: {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            if 'entities' in item:
                for entity in item['entities']:
                    if 'type' in entity:
                        entity_type = entity['type']
                        all_entities.add(entity_type)
                        entity_counts[entity_type] += 1

    print(f"\n发现的实体类型总数: {len(all_entities)}")
    print("\n实体类型统计:")
    for entity_type, count in entity_counts.most_common():
        print(f"  {entity_type}: {count}")

    return all_entities, entity_counts


def update_spanner_dataset_labels(all_entities):
    """更新spanner_dataset.py中的标签映射"""
    print("\n=== 更新标签映射 ===")

    # 备份原文件
    backup_file = f"dataloaders/spanner_dataset.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2("dataloaders/spanner_dataset.py", backup_file)
    print(f"已备份原文件到: {backup_file}")

    # 读取原文件
    with open("dataloaders/spanner_dataset.py", 'r', encoding='utf-8') as f:
        content = f.read()

    # 生成新的标签映射
    label2idx = {"O": 0}
    for i, entity_type in enumerate(sorted(all_entities)):
        label2idx[entity_type] = i + 1

    # 构建新的标签映射代码
    new_labels_code = f"""    elif args.dataname == 'navigation':
        # Navigation数据集标签映射（动态生成，包含所有发现的实体类型）
        label2idx = {{
            "O": 0"""

    for entity_type in sorted(all_entities):
        new_labels_code += f',\n            "{entity_type}": {label2idx[entity_type]}'

    new_labels_code += "\n        }"

    # 替换原标签映射
    import re
    pattern = r'elif args\.dataname == \'navigation\':\s*# Navigation数据集标签映射.*?}'
    replacement = new_labels_code

    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        print("已更新标签映射")
    else:
        print("警告：未找到原标签映射，请手动更新")
        return False

    # 写入新文件
    with open("dataloaders/spanner_dataset.py", 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("标签映射更新完成！")
    return True


def create_debug_script():
    """创建调试脚本"""
    debug_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
调试navigation数据集加载
\"\"\"

import sys
import os
sys.path.append('.')

from dataloaders.spanner_dataset import get_span_labels, BERTNERDataset
from transformers import AutoTokenizer
import argparse

def test_dataset_loading():
    # 模拟参数
    class Args:
        def __init__(self):
            self.dataname = 'navigation'
            self.data_dir = '/root/autodl-tmp/HybridNER/dataset/navigation'
            self.bert_config_dir = '/root/autodl-tmp/HybridNER/models/bert-large-cased'
            self.bert_max_length = 256
            self.max_spanLen = 6

    args = Args()

    # 获取标签映射
    label2idx_list, morph2idx_list = get_span_labels(args)
    print("=== 标签映射 ===")
    for label, idx in label2idx_list:
        print(f"  {label}: {idx}")

    print(f"\\n总标签数: {len(label2idx_list)}")

    # 测试数据集加载
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_config_dir)
        dataset = BERTNERDataset(
            json_path=os.path.join(args.data_dir, "navigation_train_span.json"),
            tokenizer=tokenizer,
            max_length=args.bert_max_length,
            max_spanLen=args.max_spanLen,
            label2idx_list=label2idx_list,
            morph2idx_list=morph2idx_list
        )

        print(f"\\n数据集加载成功！")
        print(f"数据集大小: {len(dataset)}")

        # 测试第一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\\n第一个样本:")
            print(f"  input_ids shape: {sample['input_ids'].shape}")
            print(f"  attention_mask shape: {sample['attention_mask'].shape}")
            print(f"  span_labels shape: {sample['span_labels'].shape}")
            print(f"  span_labels sum: {sample['span_labels'].sum()}")

    except Exception as e:
        print(f"数据集加载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_loading()
"""

    with open("debug_navigation_labels.py", 'w', encoding='utf-8') as f:
        f.write(debug_script)

    print("已创建调试脚本: debug_navigation_labels.py")


def main():
    """主函数"""
    print("=== Navigation数据集标签修复工具 ===")

    # 数据集路径
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"

    if not os.path.exists(data_dir):
        print(f"错误：数据集目录不存在: {data_dir}")
        return

    # 分析数据集
    all_entities, entity_counts = analyze_navigation_dataset(data_dir)

    if not all_entities:
        print("错误：未发现任何实体类型")
        return

    # 更新标签映射
    if update_spanner_dataset_labels(all_entities):
        print("\n=== 修复完成 ===")
        print("现在可以运行训练脚本了")

        # 创建调试脚本
        create_debug_script()

        print("\n建议执行以下步骤:")
        print("1. 运行调试脚本: python debug_navigation_labels.py")
        print("2. 如果调试成功，运行训练: bash run_4090_optimized_training.sh")
    else:
        print("修复失败，请手动更新标签映射")


if __name__ == "__main__":
    main()

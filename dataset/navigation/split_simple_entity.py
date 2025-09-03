#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 entity.jsonl 数据集分割脚本
快速将 entity.jsonl 分割为训练集、验证集和测试集
"""

import json
import random

# 配置参数
INPUT_FILE = "entity.jsonl"
TRAIN_RATIO = 0.8  # 训练集比例
VAL_RATIO = 0.1  # 验证集比例
TEST_RATIO = 0.1  # 测试集比例
RANDOM_SEED = 42  # 随机种子


def main():
    print("开始分割 entity.jsonl 数据集...")

    # 读取数据
    print("正在读取 entity.jsonl...")
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))

    print(f"总共读取 {len(data)} 条数据")

    # 统计基本信息
    total_entities = 0
    for item in data:
        if 'entities' in item:
            total_entities += len(item['entities'])

    print(f"总实体数量: {total_entities}")
    print(f"平均每个样本实体数: {total_entities / len(data):.2f}")

    # 随机打乱
    random.seed(RANDOM_SEED)
    random.shuffle(data)

    # 计算分割点
    total = len(data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    # 分割数据
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print("正在保存分割后的文件...")

    # 保存训练集
    with open("entity_train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 保存验证集
    with open("entity_val.jsonl", 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 保存测试集
    with open("entity_test.jsonl", 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ 分割完成！")
    print(f"  训练集: {len(train_data)} 条 -> entity_train.jsonl")
    print(f"  验证集: {len(val_data)} 条 -> entity_val.jsonl")
    print(f"  测试集: {len(test_data)} 条 -> entity_test.jsonl")

    # 显示数据集统计信息
    print(f"\n数据集统计:")
    print(f"  训练集比例: {len(train_data) / total:.1%}")
    print(f"  验证集比例: {len(val_data) / total:.1%}")
    print(f"  测试集比例: {len(test_data) / total:.1%}")

    # 验证实体分布
    for name, dataset in [("训练集", train_data), ("验证集", val_data), ("测试集", test_data)]:
        entities_count = sum(len(item.get('entities', [])) for item in dataset)
        print(f"  {name}实体数: {entities_count}")


if __name__ == "__main__":
    main()

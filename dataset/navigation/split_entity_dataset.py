#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entity.jsonl 数据集分割脚本
将 entity.jsonl 文件分割为训练集、验证集和测试集
专门针对 NER 数据集的 JSONL 格式
"""

import json
import random
import os
from typing import List, Dict, Any
from collections import Counter


def load_jsonl(file_path: str) -> List[Dict[Any, Any]]:
    """加载 JSONL 文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"警告: 第 {line_num} 行 JSON 解析失败 - {e}")
                        continue
        print(f"成功加载 {len(data)} 条数据从 {file_path}")
        return data
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return []
    except Exception as e:
        print(f"错误: 加载文件时出现问题 - {e}")
        return []


def save_jsonl(data: List[Dict[Any, Any]], file_path: str) -> None:
    """保存数据到 JSONL 文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"成功保存 {len(data)} 条数据到 {file_path}")
    except Exception as e:
        print(f"错误: 保存文件时出现问题 - {e}")


def analyze_dataset(data: List[Dict[Any, Any]]) -> None:
    """分析数据集的基本信息"""
    print("\n=== 数据集分析 ===")
    print(f"总样本数量: {len(data)}")

    # 统计实体数量
    total_entities = 0
    entity_labels = []
    text_lengths = []
    entity_counts_per_sample = []

    for item in data:
        if 'entities' in item:
            entities = item['entities']
            total_entities += len(entities)
            entity_counts_per_sample.append(len(entities))
            for entity in entities:
                if 'label' in entity:
                    entity_labels.append(entity['label'])
        else:
            entity_counts_per_sample.append(0)

        if 'text' in item:
            text_lengths.append(len(item['text']))

    print(f"总实体数量: {total_entities}")
    print(f"平均每个样本实体数: {total_entities / len(data):.2f}")
    print(f"最多实体数的样本: {max(entity_counts_per_sample)} 个实体")
    print(f"最少实体数的样本: {min(entity_counts_per_sample)} 个实体")
    print(f"平均文本长度: {sum(text_lengths) / len(text_lengths):.1f} 字符")
    print(f"最长文本: {max(text_lengths)} 字符")
    print(f"最短文本: {min(text_lengths)} 字符")

    # 统计实体标签分布
    label_counts = Counter(entity_labels)
    print(f"\n实体标签分布 (Top 15):")
    for label, count in label_counts.most_common(15):
        percentage = (count / total_entities) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    print("=" * 50)


def split_dataset(data: List[Dict[Any, Any]],
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  random_seed: int = 42,
                  stratify_by_entity_count: bool = True) -> tuple:
    """
    分割数据集，支持按实体数量分层

    Args:
        data: 原始数据
        train_ratio: 训练集比例 (默认 0.8)
        val_ratio: 验证集比例 (默认 0.1)
        test_ratio: 测试集比例 (默认 0.1)
        random_seed: 随机种子 (默认 42)
        stratify_by_entity_count: 是否按实体数量分层 (默认 True)

    Returns:
        (train_data, val_data, test_data)
    """
    # 检查比例是否合法
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        print("警告: 分割比例之和不等于1，将自动归一化")
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

    # 设置随机种子确保可重现性
    random.seed(random_seed)

    if stratify_by_entity_count:
        # 按实体数量分层
        print("使用实体数量分层策略...")

        # 计算每个样本的实体数量并分组
        entity_groups = {}
        for item in data:
            entity_count = len(item.get('entities', []))
            # 将实体数量分组（0，1-5，6-10，11-20，21+）
            if entity_count == 0:
                group = "0"
            elif entity_count <= 5:
                group = "1-5"
            elif entity_count <= 10:
                group = "6-10"
            elif entity_count <= 20:
                group = "11-20"
            else:
                group = "21+"

            if group not in entity_groups:
                entity_groups[group] = []
            entity_groups[group].append(item)

        print(f"实体数量分组情况:")
        for group, items in entity_groups.items():
            print(f"  {group} 个实体: {len(items)} 样本")

        # 对每组进行分层分割
        train_data, val_data, test_data = [], [], []

        for group, items in entity_groups.items():
            random.shuffle(items)

            group_size = len(items)
            train_size = int(group_size * train_ratio)
            val_size = int(group_size * val_ratio)

            train_data.extend(items[:train_size])
            val_data.extend(items[train_size:train_size + val_size])
            test_data.extend(items[train_size + val_size:])

        # 再次随机打乱
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

    else:
        # 完全随机打乱
        print("使用完全随机分割策略...")
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)

        total_size = len(shuffled_data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        # 分割数据
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]

    print(f"\n数据集分割完成:")
    print(f"原始数据: {len(data)} 条")
    print(f"训练集: {len(train_data)} 条 ({len(train_data) / len(data):.1%})")
    print(f"验证集: {len(val_data)} 条 ({len(val_data) / len(data):.1%})")
    print(f"测试集: {len(test_data)} 条 ({len(test_data) / len(data):.1%})")

    return train_data, val_data, test_data


def verify_split(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> None:
    """验证分割结果"""
    print(f"\n=== 分割结果验证 ===")

    # 检查ID重复
    all_ids = set()
    for dataset_name, dataset in [("训练集", train_data), ("验证集", val_data), ("测试集", test_data)]:
        ids = [item.get('id') for item in dataset if 'id' in item]
        duplicate_ids = [id for id in ids if id in all_ids]
        if duplicate_ids:
            print(f"警告: {dataset_name} 中发现重复ID: {duplicate_ids}")
        all_ids.update(ids)
        print(f"{dataset_name} ID 数量: {len(ids)}")

    # 统计每个数据集的实体分布
    for dataset_name, dataset in [("训练集", train_data), ("验证集", val_data), ("测试集", test_data)]:
        entity_labels = []
        entity_counts = []

        for item in dataset:
            if 'entities' in item:
                entities = item['entities']
                entity_counts.append(len(entities))
                for entity in entities:
                    if 'label' in entity:
                        entity_labels.append(entity['label'])
            else:
                entity_counts.append(0)

        label_counts = Counter(entity_labels)
        avg_entities = sum(entity_counts) / len(entity_counts) if entity_counts else 0

        print(f"\n{dataset_name} 统计:")
        print(f"  总实体数: {sum(entity_counts)}")
        print(f"  平均每样本实体数: {avg_entities:.2f}")
        print(f"  实体标签分布 (Top 5):")
        for label, count in label_counts.most_common(5):
            print(f"    {label}: {count}")


def main():
    """主函数"""
    # 设置文件路径
    input_file = "entity.jsonl"
    output_dir = "."  # 当前目录

    # 输出文件名
    train_file = os.path.join(output_dir, "entity_train.jsonl")
    val_file = os.path.join(output_dir, "entity_val.jsonl")
    test_file = os.path.join(output_dir, "entity_test.jsonl")

    print("=" * 60)
    print("Entity.jsonl NER 数据集分割工具")
    print("=" * 60)

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        print("请确保 entity.jsonl 文件在当前目录下")
        return

    # 加载数据
    print(f"\n正在加载数据文件: {input_file}")
    data = load_jsonl(input_file)

    if not data:
        print("没有数据可以分割，程序退出")
        return

    # 分析数据集
    analyze_dataset(data)

    # 分割数据集
    print(f"\n正在分割数据集...")
    train_data, val_data, test_data = split_dataset(
        data,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        stratify_by_entity_count=True  # 对于NER任务，按实体数量分层有助于平衡
    )

    # 验证分割结果
    verify_split(train_data, val_data, test_data)

    # 保存分割后的数据
    print(f"\n正在保存分割后的数据...")
    save_jsonl(train_data, train_file)
    save_jsonl(val_data, val_file)
    save_jsonl(test_data, test_file)

    print(f"\n✓ 数据集分割完成！")
    print(f"  训练集: {train_file}")
    print(f"  验证集: {val_file}")
    print(f"  测试集: {test_file}")
    print("\n分割后的文件已保存在当前目录下")
    print("=" * 60)


if __name__ == "__main__":
    main()

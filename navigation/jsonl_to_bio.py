#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL to BIO 格式转换脚本
将命名实体识别的JSONL格式数据转换为标准的BIO格式
支持中文字符级标注
"""

import json
import os
from typing import List, Dict, Tuple
import re


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"警告: 第 {line_num} 行JSON解析失败 - {e}")
                        continue
        print(f"成功加载 {len(data)} 条数据从 {file_path}")
        return data
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return []
    except Exception as e:
        print(f"错误: 加载文件时出现问题 - {e}")
        return []


def clean_text(text: str) -> str:
    """清理文本，移除特殊字符但保留空格"""
    # 移除多余的空白字符，但保留单个空格
    text = re.sub(r'\s+', ' ', text)
    # 移除首尾空格
    text = text.strip()
    return text


def tokenize_chinese(text: str) -> List[str]:
    """
    中文字符级分词
    将文本分割为字符列表，保留空格和标点符号
    """
    tokens = []
    for char in text:
        if char.strip():  # 非空白字符
            tokens.append(char)
        elif char == ' ':  # 保留空格
            tokens.append(char)
    return tokens


def create_bio_tags(text: str, entities: List[Dict]) -> List[str]:
    """
    根据实体标注创建BIO标签序列

    Args:
        text: 原始文本
        entities: 实体列表，每个实体包含 start_offset, end_offset, label

    Returns:
        BIO标签列表
    """
    # 清理文本
    clean_text_str = clean_text(text)
    tokens = tokenize_chinese(clean_text_str)

    # 初始化所有标签为O
    bio_tags = ['O'] * len(tokens)

    # 对实体按起始位置排序
    entities_sorted = sorted(entities, key=lambda x: x.get('start_offset', 0))

    # 创建字符位置到token位置的映射
    char_to_token = {}
    token_idx = 0
    char_idx = 0

    for token in tokens:
        if token == ' ':
            char_to_token[char_idx] = token_idx
            char_idx += 1
            token_idx += 1
        else:
            char_to_token[char_idx] = token_idx
            char_idx += 1
            token_idx += 1

    # 处理每个实体
    for entity in entities_sorted:
        start_offset = entity.get('start_offset', 0)
        end_offset = entity.get('end_offset', 0)
        label = entity.get('label', 'MISC')

        # 调整偏移量以适应清理后的文本
        # 这里需要考虑原始文本和清理后文本的差异
        adjusted_start = min(start_offset, len(clean_text_str))
        adjusted_end = min(end_offset, len(clean_text_str))

        # 确保实体范围有效
        if adjusted_start >= adjusted_end or adjusted_start >= len(tokens):
            continue

        # 找到对应的token位置
        start_token_idx = None
        end_token_idx = None

        # 寻找起始token位置
        for char_pos in range(adjusted_start, len(clean_text_str)):
            if char_pos in char_to_token:
                start_token_idx = char_to_token[char_pos]
                break

        # 寻找结束token位置
        for char_pos in range(adjusted_end - 1, -1, -1):
            if char_pos in char_to_token:
                end_token_idx = char_to_token[char_pos] + 1
                break

        # 如果找不到有效位置，跳过
        if start_token_idx is None or end_token_idx is None:
            continue

        # 确保索引范围有效
        start_token_idx = max(0, min(start_token_idx, len(tokens)))
        end_token_idx = max(0, min(end_token_idx, len(tokens)))

        if start_token_idx >= end_token_idx:
            continue

        # 设置BIO标签
        for i in range(start_token_idx, end_token_idx):
            if i < len(bio_tags):
                if i == start_token_idx:
                    bio_tags[i] = f'B-{label}'
                else:
                    bio_tags[i] = f'I-{label}'

    return bio_tags


def convert_jsonl_to_bio(jsonl_data: List[Dict]) -> List[Tuple[List[str], List[str]]]:
    """
    将JSONL数据转换为BIO格式

    Returns:
        List of (tokens, bio_tags) tuples
    """
    bio_data = []

    for item in jsonl_data:
        text = item.get('text', '')
        entities = item.get('entities', [])

        if not text.strip():
            continue

        # 清理文本
        clean_text_str = clean_text(text)
        tokens = tokenize_chinese(clean_text_str)

        # 创建BIO标签
        bio_tags = create_bio_tags(text, entities)

        # 确保tokens和bio_tags长度一致
        if len(tokens) != len(bio_tags):
            print(f"警告: tokens长度({len(tokens)})与bio_tags长度({len(bio_tags)})不一致")
            # 调整到相同长度
            min_len = min(len(tokens), len(bio_tags))
            tokens = tokens[:min_len]
            bio_tags = bio_tags[:min_len]

        if tokens and bio_tags:
            bio_data.append((tokens, bio_tags))

    return bio_data


def save_bio_format(bio_data: List[Tuple[List[str], List[str]]], output_file: str):
    """
    保存BIO格式数据到文件
    格式: token tag
    句子之间用空行分隔
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for tokens, bio_tags in bio_data:
                for token, tag in zip(tokens, bio_tags):
                    f.write(f'{token}\t{tag}\n')
                f.write('\n')  # 句子间空行分隔
        print(f"成功保存BIO格式数据到 {output_file}")
    except Exception as e:
        print(f"错误: 保存文件时出现问题 - {e}")


def analyze_conversion(bio_data: List[Tuple[List[str], List[str]]]):
    """分析转换结果"""
    print("\n=== 转换结果分析 ===")

    total_tokens = 0
    total_entities = 0
    label_counts = {}

    for tokens, bio_tags in bio_data:
        total_tokens += len(tokens)

        # 统计实体
        current_entity = None
        for tag in bio_tags:
            if tag.startswith('B-'):
                total_entities += 1
                label = tag[2:]
                current_entity = label
                label_counts[label] = label_counts.get(label, 0) + 1
            elif tag.startswith('I-'):
                if current_entity is None:
                    # I标签前面没有B标签，这是错误的
                    print(f"警告: 发现无效的I标签: {tag}")

    print(f"总句子数: {len(bio_data)}")
    print(f"总token数: {total_tokens}")
    print(f"总实体数: {total_entities}")

    if label_counts:
        print(f"\n实体标签分布:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}")

    print("=" * 50)


def main():
    """主函数"""
    input_files = [
        "entity_train.jsonl",
        "entity_val.jsonl",
        "entity_test.jsonl"
    ]

    output_files = [
        "entity_train.bio",
        "entity_val.bio",
        "entity_test.bio"
    ]

    print("=" * 60)
    print("JSONL to BIO 格式转换工具")
    print("=" * 60)

    for input_file, output_file in zip(input_files, output_files):
        if not os.path.exists(input_file):
            print(f"跳过: 文件 {input_file} 不存在")
            continue

        print(f"\n正在处理: {input_file} -> {output_file}")

        # 加载JSONL数据
        jsonl_data = load_jsonl(input_file)
        if not jsonl_data:
            print(f"跳过: {input_file} 无有效数据")
            continue

        # 转换为BIO格式
        bio_data = convert_jsonl_to_bio(jsonl_data)

        # 分析转换结果
        analyze_conversion(bio_data)

        # 保存BIO格式文件
        save_bio_format(bio_data, output_file)

    print(f"\n✓ 所有文件转换完成！")
    print("生成的BIO格式文件:")
    for output_file in output_files:
        if os.path.exists(output_file):
            print(f"  ✓ {output_file}")
        else:
            print(f"  ✗ {output_file} (未生成)")

    print("\nBIO格式说明:")
    print("  - B-LABEL: 实体开始")
    print("  - I-LABEL: 实体内部")
    print("  - O: 非实体")
    print("  - 每行格式: token\\ttag")
    print("  - 句子间用空行分隔")
    print("=" * 60)


if __name__ == "__main__":
    main()

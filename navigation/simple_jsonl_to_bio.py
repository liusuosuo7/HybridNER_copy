#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 JSONL to BIO 转换脚本
更直接的字符级转换，适合中文文本
"""

import json
import os
from typing import List, Dict


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def convert_to_bio_simple(text: str, entities: List[Dict]) -> tuple:
    """
    简化版BIO转换
    直接按字符处理，避免复杂的偏移计算
    """
    # 将文本转为字符列表
    chars = list(text)
    tags = ['O'] * len(chars)

    # 处理每个实体
    for entity in entities:
        start = entity.get('start_offset', 0)
        end = entity.get('end_offset', 0)
        label = entity.get('label', 'MISC')

        # 确保偏移量在有效范围内
        start = max(0, min(start, len(chars)))
        end = max(0, min(end, len(chars)))

        if start < end and start < len(chars):
            # 设置B-标签
            tags[start] = f'B-{label}'
            # 设置I-标签
            for i in range(start + 1, end):
                if i < len(chars):
                    tags[i] = f'I-{label}'

    # 过滤掉空格等无意义字符，但保留有意义的标点
    filtered_chars = []
    filtered_tags = []

    for char, tag in zip(chars, tags):
        # 保留中文字符、英文字母、数字、常见标点符号
        if char.strip() and (
                '\u4e00' <= char <= '\u9fff' or  # 中文字符
                char.isalnum() or  # 字母数字
                char in '.,!?;:()[]{}"\'-、。，！？；：（）【】《》""''…'
        ):
            filtered_chars.append(char)
            filtered_tags.append(tag)

    return filtered_chars, filtered_tags


def save_bio_file(bio_data: List[tuple], output_file: str):
    """保存BIO格式文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for chars, tags in bio_data:
            for char, tag in zip(chars, tags):
                f.write(f'{char}\t{tag}\n')
            f.write('\n')  # 句子分隔符


def main():
    """主函数"""
    files_to_convert = [
        ("entity_train.jsonl", "entity_train.bio"),
        ("entity_val.jsonl", "entity_val.bio"),
        ("entity_test.jsonl", "entity_test.bio")
    ]

    print("开始转换JSONL到BIO格式...")

    for input_file, output_file in files_to_convert:
        if not os.path.exists(input_file):
            print(f"跳过: {input_file} 不存在")
            continue

        print(f"转换: {input_file} -> {output_file}")

        # 加载数据
        data = load_jsonl(input_file)

        # 转换为BIO格式
        bio_data = []
        for item in data:
            text = item.get('text', '')
            entities = item.get('entities', [])

            if text.strip():
                chars, tags = convert_to_bio_simple(text, entities)
                if chars and tags:
                    bio_data.append((chars, tags))

        # 保存文件
        save_bio_file(bio_data, output_file)

        # 统计信息
        total_chars = sum(len(chars) for chars, _ in bio_data)
        total_entities = sum(
            1 for _, tags in bio_data
            for tag in tags if tag.startswith('B-')
        )

        print(f"  句子数: {len(bio_data)}")
        print(f"  字符数: {total_chars}")
        print(f"  实体数: {total_entities}")

    print("\n✓ 转换完成!")
    print("生成的文件:")
    for _, output_file in files_to_convert:
        if os.path.exists(output_file):
            print(f"  ✓ {output_file}")


if __name__ == "__main__":
    main()

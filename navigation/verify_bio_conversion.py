#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证BIO格式转换结果
检查BIO标注的正确性和一致性
"""

import os
from collections import Counter


def load_bio_file(file_path: str):
    """加载BIO格式文件"""
    sentences = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, tag = parts
                    current_sentence.append((token, tag))
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []

    # 添加最后一个句子
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def validate_bio_tags(sentences):
    """验证BIO标签的正确性"""
    errors = []
    tag_stats = Counter()
    entity_stats = Counter()

    for sent_idx, sentence in enumerate(sentences):
        prev_tag = 'O'
        current_entity = None

        for token_idx, (token, tag) in enumerate(sentence):
            tag_stats[tag] += 1

            # 检查标签格式
            if tag not in ['O'] and not (tag.startswith('B-') or tag.startswith('I-')):
                errors.append(f"句子{sent_idx}, token{token_idx}: 无效标签格式 '{tag}'")
                continue

            # 检查I标签是否跟在相应的B或I标签后面
            if tag.startswith('I-'):
                entity_type = tag[2:]
                if prev_tag == 'O':
                    errors.append(f"句子{sent_idx}, token{token_idx}: I-{entity_type} 前面没有对应的B标签")
                elif prev_tag.startswith('B-') or prev_tag.startswith('I-'):
                    prev_entity_type = prev_tag[2:]
                    if entity_type != prev_entity_type:
                        errors.append(f"句子{sent_idx}, token{token_idx}: I-{entity_type} 与前面的{prev_tag}不匹配")

            # 统计实体
            if tag.startswith('B-'):
                entity_type = tag[2:]
                entity_stats[entity_type] += 1
                current_entity = entity_type
            elif tag == 'O':
                current_entity = None

            prev_tag = tag

    return errors, tag_stats, entity_stats


def analyze_bio_file(file_path: str):
    """分析BIO文件"""
    print(f"\n=== 分析 {file_path} ===")

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    # 加载数据
    sentences = load_bio_file(file_path)

    # 基本统计
    total_sentences = len(sentences)
    total_tokens = sum(len(sent) for sent in sentences)

    print(f"句子数量: {total_sentences}")
    print(f"token数量: {total_tokens}")

    if total_sentences == 0:
        print("文件为空或格式错误")
        return

    # 验证标签
    errors, tag_stats, entity_stats = validate_bio_tags(sentences)

    # 显示错误
    if errors:
        print(f"\n发现 {len(errors)} 个错误:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")
    else:
        print("✓ 未发现BIO标签错误")

    # 标签统计
    print(f"\n标签分布:")
    for tag, count in tag_stats.most_common():
        percentage = (count / total_tokens) * 100
        print(f"  {tag}: {count} ({percentage:.1f}%)")

    # 实体统计
    if entity_stats:
        print(f"\n实体类型分布:")
        total_entities = sum(entity_stats.values())
        for entity_type, count in entity_stats.most_common():
            percentage = (count / total_entities) * 100
            print(f"  {entity_type}: {count} ({percentage:.1f}%)")

    # 平均句长
    avg_length = total_tokens / total_sentences
    print(f"\n平均句长: {avg_length:.1f} tokens")


def compare_datasets():
    """比较训练集、验证集、测试集的分布"""
    files = ["entity_train.bio", "entity_val.bio", "entity_test.bio"]
    all_stats = {}

    print("\n=== 数据集对比 ===")

    for file_path in files:
        if os.path.exists(file_path):
            sentences = load_bio_file(file_path)
            _, tag_stats, entity_stats = validate_bio_tags(sentences)

            total_tokens = sum(tag_stats.values())
            total_entities = sum(entity_stats.values())

            all_stats[file_path] = {
                'sentences': len(sentences),
                'tokens': total_tokens,
                'entities': total_entities,
                'entity_stats': entity_stats
            }

    # 显示对比
    print(f"{'数据集':<15} {'句子数':<8} {'Token数':<8} {'实体数':<8} {'实体密度':<10}")
    print("-" * 55)

    for file_path, stats in all_stats.items():
        name = file_path.replace('entity_', '').replace('.bio', '')
        density = stats['entities'] / stats['tokens'] * 100 if stats['tokens'] > 0 else 0
        print(f"{name:<15} {stats['sentences']:<8} {stats['tokens']:<8} {stats['entities']:<8} {density:<10.2f}%")


def show_sample_data(file_path: str, num_samples: int = 3):
    """显示样本数据"""
    print(f"\n=== {file_path} 样本数据 ===")

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    sentences = load_bio_file(file_path)

    for i, sentence in enumerate(sentences[:num_samples]):
        print(f"\n样本 {i + 1}:")
        for token, tag in sentence[:20]:  # 只显示前20个token
            print(f"  {token}\t{tag}")
        if len(sentence) > 20:
            print(f"  ... (还有 {len(sentence) - 20} 个token)")


def main():
    """主函数"""
    print("=" * 60)
    print("BIO格式验证工具")
    print("=" * 60)

    bio_files = ["entity_train.bio", "entity_val.bio", "entity_test.bio"]

    # 分析每个文件
    for file_path in bio_files:
        analyze_bio_file(file_path)

    # 数据集对比
    compare_datasets()

    # 显示样本数据
    if os.path.exists("entity_train.bio"):
        show_sample_data("entity_train.bio", 2)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
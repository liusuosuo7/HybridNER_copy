#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试中文NER配置
"""

import os
import sys
from transformers import BertTokenizer, BertConfig

def test_bert_models():
    """测试BERT模型是否可以正常加载"""
    
    models = [
        "/root/autodl-tmp/HybridNER/models/bert-base-cased",
        "/root/autodl-tmp/HybridNER/models/bert-large-cased"
    ]
    
    for model_path in models:
        print(f"\n测试模型: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"错误: 模型路径不存在 {model_path}")
            continue
            
        try:
            # 测试分词器
            tokenizer = BertTokenizer.from_pretrained(model_path)
            print("✓ 分词器加载成功")
            
            # 测试配置
            config = BertConfig.from_pretrained(model_path)
            print(f"✓ 配置加载成功 (hidden_size: {config.hidden_size})")
            
            # 测试中文分词
            text = "张三在北京大学读书"
            tokens = tokenizer.tokenize(text)
            print(f"✓ 中文分词测试: {text} -> {tokens}")
            
        except Exception as e:
            print(f"✗ 加载失败: {e}")

def test_chinese_processing():
    """测试中文处理功能"""
    
    print("\n测试中文处理功能:")
    
    # 测试字符特征提取
    text = "张三在北京大学读书123！"
    
    features = []
    for char in text:
        if char.isdigit():
            features.append("isdigit")
        elif not char.isalnum():
            features.append("ispunct")
        else:
            features.append("other")
    
    print(f"中文特征提取: {text}")
    print(f"特征: {features}")
    
    # 测试标签映射
    label_mapping = {"O": 0, "PER": 1, "LOC": 2, "ORG": 3, "MISC": 4}
    print(f"标签映射: {label_mapping}")

def test_dataset_structure():
    """测试数据集结构"""
    
    print("\n测试数据集结构:")
    
    # 检查多个数据集路径
    data_dirs = [
        "/root/autodl-tmp/HybridNER/dataset/msra",
        "/root/autodl-tmp/HybridNER/dataset/cluener",
        "/root/autodl-tmp/HybridNER/dataset/cmeee"
    ]
    
    for data_dir in data_dirs:
        print(f"\n检查数据集: {data_dir}")
        if os.path.exists(data_dir):
            print(f"✓ 数据目录存在: {data_dir}")
            
            # 检查常见的数据文件格式
            possible_files = ["spanner.train", "spanner.dev", "spanner.test", 
                            "train.json", "dev.json", "test.json",
                            "train.txt", "dev.txt", "test.txt"]
            
            found_files = []
            for file in possible_files:
                file_path = os.path.join(data_dir, file)
                if os.path.exists(file_path):
                    found_files.append(file)
                    print(f"✓ 数据文件存在: {file}")
            
            if not found_files:
                print("⚠ 未找到标准格式的数据文件")
                # 列出目录中的所有文件
                try:
                    all_files = os.listdir(data_dir)
                    print(f"目录中的文件: {all_files[:10]}...")  # 只显示前10个文件
                except:
                    print("无法列出目录内容")
        else:
            print(f"✗ 数据目录不存在: {data_dir}")

def main():
    """主函数"""
    
    print("=== 中文NER配置测试 ===")
    
    # 测试BERT模型
    test_bert_models()
    
    # 测试中文处理
    test_chinese_processing()
    
    # 测试数据集结构
    test_dataset_structure()
    
    print("\n=== 测试完成 ===")
    
    print("\n下一步:")
    print("1. 准备MSRA数据集在 data/msra/ 目录")
    print("2. 运行: python run_chinese_ner.py train")
    print("3. 或运行: python main.py --dataname msra --data_dir data/msra")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建示例的spanner格式数据文件
用于MSRA、CLUENER、CMEEE数据集
"""

import json
import os

def create_msra_sample():
    """创建MSRA数据集的示例数据"""
    
    # 示例中文文本和实体标注
    sample_data = [
        {
            "context": "张三在北京大学读书。",
            "span_posLabel": {
                "0;1": "PER",  # 张三
                "3;5": "LOC",  # 北京
                "6;9": "ORG"   # 北京大学
            }
        },
        {
            "context": "李四在上海交通大学工作。",
            "span_posLabel": {
                "0;1": "PER",   # 李四
                "3;5": "LOC",   # 上海
                "6;12": "ORG"   # 上海交通大学
            }
        },
        {
            "context": "王五在清华大学获得了硕士学位。",
            "span_posLabel": {
                "0;1": "PER",     # 王五
                "3;6": "ORG",     # 清华大学
                "9;12": "MISC"    # 硕士学位
            }
        }
    ]
    
    # 创建目录
    os.makedirs("/root/autodl-tmp/HybridNER/dataset/msra", exist_ok=True)
    
    # 保存训练数据
    train_path = "/root/autodl-tmp/HybridNER/dataset/msra/msra_train_span.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # 保存验证数据
    dev_path = "/root/autodl-tmp/HybridNER/dataset/msra/msra_dev_span.json"
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(sample_data[:2], f, ensure_ascii=False, indent=2)
    
    # 保存测试数据
    test_path = "/root/autodl-tmp/HybridNER/dataset/msra/msra_test_span.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(sample_data[1:], f, ensure_ascii=False, indent=2)
    
    print(f"✓ MSRA示例数据已创建:")
    print(f"  - 训练集: {train_path}")
    print(f"  - 验证集: {dev_path}")
    print(f"  - 测试集: {test_path}")

def create_cluener_sample():
    """创建CLUENER数据集的示例数据"""
    
    sample_data = [
        {
            "context": "《红楼梦》是中国古典文学名著。",
            "span_posLabel": {
                "0;4": "book",      # 《红楼梦》
                "6;8": "LOC",       # 中国
                "9;13": "MISC"      # 古典文学
            }
        },
        {
            "context": "腾讯公司位于深圳。",
            "span_posLabel": {
                "0;3": "company",   # 腾讯公司
                "6;8": "LOC"        # 深圳
            }
        },
        {
            "context": "张三担任总经理职位。",
            "span_posLabel": {
                "0;1": "name",      # 张三
                "4;7": "position"   # 总经理
            }
        }
    ]
    
    # 创建目录
    os.makedirs("/root/autodl-tmp/HybridNER/dataset/cluener", exist_ok=True)
    
    # 保存训练数据
    train_path = "/root/autodl-tmp/HybridNER/dataset/cluener/spanner.train"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # 保存验证数据
    dev_path = "/root/autodl-tmp/HybridNER/dataset/cluener/spanner.dev"
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(sample_data[:2], f, ensure_ascii=False, indent=2)
    
    # 保存测试数据
    test_path = "/root/autodl-tmp/HybridNER/dataset/cluener/spanner.test"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(sample_data[1:], f, ensure_ascii=False, indent=2)
    
    print(f"✓ CLUENER示例数据已创建:")
    print(f"  - 训练集: {train_path}")
    print(f"  - 验证集: {dev_path}")
    print(f"  - 测试集: {test_path}")

def create_cmeee_sample():
    """创建CMEEE数据集的示例数据"""
    
    sample_data = [
        {
            "context": "患者出现头痛和发热症状。",
            "span_posLabel": {
                "3;5": "sym",      # 头痛
                "6;8": "sym"       # 发热
            }
        },
        {
            "context": "医生建议使用阿司匹林治疗。",
            "span_posLabel": {
                "0;1": "name",     # 医生
                "7;11": "dru",     # 阿司匹林
                "12;14": "pro"     # 治疗
            }
        },
        {
            "context": "使用CT扫描检查肺部。",
            "span_posLabel": {
                "3;6": "equ",      # CT扫描
                "9;11": "bod"      # 肺部
            }
        }
    ]
    
    # 创建目录
    os.makedirs("/root/autodl-tmp/HybridNER/dataset/cmeee", exist_ok=True)
    
    # 保存训练数据
    train_path = "/root/autodl-tmp/HybridNER/dataset/cmeee/spanner.train"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # 保存验证数据
    dev_path = "/root/autodl-tmp/HybridNER/dataset/cmeee/spanner.dev"
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(sample_data[:2], f, ensure_ascii=False, indent=2)
    
    # 保存测试数据
    test_path = "/root/autodl-tmp/HybridNER/dataset/cmeee/spanner.test"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(sample_data[1:], f, ensure_ascii=False, indent=2)
    
    print(f"✓ CMEEE示例数据已创建:")
    print(f"  - 训练集: {train_path}")
    print(f"  - 验证集: {dev_path}")
    print(f"  - 测试集: {test_path}")

def main():
    """主函数"""
    print("=== 创建示例数据文件 ===")
    
    # 创建所有数据集的示例数据
    create_msra_sample()
    create_cluener_sample()
    create_cmeee_sample()
    
    print("\n=== 示例数据创建完成 ===")
    print("现在您可以尝试运行训练了:")
    print("python run_simple.py msra")
    print("python run_simple.py cluener")
    print("python run_simple.py cmeee")

if __name__ == "__main__":
    main()

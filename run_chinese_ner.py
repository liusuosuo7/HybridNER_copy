#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中文NER运行脚本
使用现有的bert-base-cased和bert-large-cased模型
"""

import os
import sys

def run_chinese_ner():
    """运行中文NER训练"""
    
    # 基础配置
    base_cmd = [
        "python", "main.py",
        "--dataname", "msra",  # 使用MSRA数据集
        "--data_dir", "/root/autodl-tmp/HybridNER/dataset/msra",  # 数据目录
        "--bert_config_dir", "/root/autodl-tmp/HybridNER/models/bert-base-cased",  # 使用现有的BERT模型
        "--batch_size", "32",  # 批次大小，中文数据可能需要更小的批次
        "--lr", "3e-5",  # 学习率
        "--iteration", "30",  # 训练轮数
        "--bert_max_length", "128",  # 最大序列长度
        "--max_spanLen", "5",  # 最大span长度
        "--model_save_dir", "results/chinese_ner/",  # 模型保存目录
        "--logger_dir", "log/chinese_ner/",  # 日志目录
        "--results_dir", "test_res/chinese_ner/",  # 结果目录
    ]
    
    print("开始运行中文NER训练...")
    print("命令:", " ".join(base_cmd))
    
    # 创建必要的目录
    os.makedirs("results/chinese_ner/", exist_ok=True)
    os.makedirs("log/chinese_ner/", exist_ok=True)
    os.makedirs("test_res/chinese_ner/", exist_ok=True)
    
    # 运行命令
    os.system(" ".join(base_cmd))

def run_chinese_ner_large():
    """使用bert-large-cased模型运行中文NER"""
    
    base_cmd = [
        "python", "main.py",
        "--dataname", "msra",
        "--data_dir", "/root/autodl-tmp/HybridNER/dataset/msra",
        "--bert_config_dir", "/root/autodl-tmp/HybridNER/models/bert-large-cased",  # 使用large模型
        "--batch_size", "16",  # large模型需要更小的批次
        "--lr", "2e-5",  # large模型可能需要更小的学习率
        "--iteration", "30",
        "--bert_max_length", "128",
        "--max_spanLen", "5",
        "--model_save_dir", "results/chinese_ner_large/",
        "--logger_dir", "log/chinese_ner_large/",
        "--results_dir", "test_res/chinese_ner_large/",
    ]
    
    print("开始运行中文NER训练（使用BERT-Large）...")
    print("命令:", " ".join(base_cmd))
    
    # 创建必要的目录
    os.makedirs("results/chinese_ner_large/", exist_ok=True)
    os.makedirs("log/chinese_ner_large/", exist_ok=True)
    os.makedirs("test_res/chinese_ner_large/", exist_ok=True)
    
    # 运行命令
    os.system(" ".join(base_cmd))

def run_inference():
    """运行推理"""
    
    base_cmd = [
        "python", "main.py",
        "--state", "inference",
        "--dataname", "msra",
        "--data_dir", "/root/autodl-tmp/HybridNER/dataset/msra",
        "--bert_config_dir", "/root/autodl-tmp/HybridNER/models/bert-base-cased",
        "--inference_model", "results/chinese_ner/best_model.pkl",  # 使用训练好的模型
        "--batch_size", "32",
        "--bert_max_length", "128",
        "--max_spanLen", "5",
        "--results_dir", "test_res/chinese_ner_inference/",
    ]
    
    print("开始运行中文NER推理...")
    print("命令:", " ".join(base_cmd))
    
    # 创建必要的目录
    os.makedirs("test_res/chinese_ner_inference/", exist_ok=True)
    
    # 运行命令
    os.system(" ".join(base_cmd))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            run_chinese_ner()
        elif sys.argv[1] == "train_large":
            run_chinese_ner_large()
        elif sys.argv[1] == "inference":
            run_inference()
        else:
            print("用法: python run_chinese_ner.py [train|train_large|inference]")
    else:
        # 默认运行训练
        run_chinese_ner()

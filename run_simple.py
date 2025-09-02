#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的中文NER运行脚本
"""

import os
import sys

def run_msra_simple():
    """运行MSRA数据集的简化版本"""
    
    # 基础配置
    cmd = [
        "python", "main.py",
        "--dataname", "msra",
        "--data_dir", "/root/autodl-tmp/HybridNER/dataset/msra",
        "--bert_config_dir", "/root/autodl-tmp/HybridNER/models/bert-base-cased",
        "--batch_size", "32",
        "--lr", "3e-5",
        "--iteration", "30",
        "--bert_max_length", "128",
        "--max_spanLen", "5",
        "--model_save_dir", "results/msra/",
        "--logger_dir", "log/msra/",
        "--results_dir", "test_res/msra/",
    ]
    
    print("开始运行MSRA数据集训练...")
    print("命令:", " ".join(cmd))
    
    # 创建必要的目录
    os.makedirs("results/msra/", exist_ok=True)
    os.makedirs("log/msra/", exist_ok=True)
    os.makedirs("test_res/msra/", exist_ok=True)
    
    # 运行命令
    result = os.system(" ".join(cmd))
    
    if result == 0:
        print("训练完成！")
    else:
        print(f"训练失败，错误代码: {result}")

def run_cluener_simple():
    """运行CLUENER数据集的简化版本"""
    
    cmd = [
        "python", "main.py",
        "--dataname", "cluener",
        "--data_dir", "/root/autodl-tmp/HybridNER/dataset/cluener",
        "--bert_config_dir", "/root/autodl-tmp/HybridNER/models/bert-base-cased",
        "--batch_size", "32",
        "--lr", "3e-5",
        "--iteration", "30",
        "--bert_max_length", "128",
        "--max_spanLen", "5",
        "--model_save_dir", "results/cluener/",
        "--logger_dir", "log/cluener/",
        "--results_dir", "test_res/cluener/",
    ]
    
    print("开始运行CLUENER数据集训练...")
    print("命令:", " ".join(cmd))
    
    # 创建必要的目录
    os.makedirs("results/cluener/", exist_ok=True)
    os.makedirs("log/cluener/", exist_ok=True)
    os.makedirs("test_res/cluener/", exist_ok=True)
    
    # 运行命令
    result = os.system(" ".join(cmd))
    
    if result == 0:
        print("训练完成！")
    else:
        print(f"训练失败，错误代码: {result}")

def run_cmeee_simple():
    """运行CMEEE数据集的简化版本"""
    
    cmd = [
        "python", "main.py",
        "--dataname", "cmeee",
        "--data_dir", "/root/autodl-tmp/HybridNER/dataset/cmeee",
        "--bert_config_dir", "/root/autodl-tmp/HybridNER/models/bert-base-cased",
        "--batch_size", "32",
        "--lr", "3e-5",
        "--iteration", "30",
        "--bert_max_length", "128",
        "--max_spanLen", "5",
        "--model_save_dir", "results/cmeee/",
        "--logger_dir", "log/cmeee/",
        "--results_dir", "test_res/cmeee/",
    ]
    
    print("开始运行CMEEE数据集训练...")
    print("命令:", " ".join(cmd))
    
    # 创建必要的目录
    os.makedirs("results/cmeee/", exist_ok=True)
    os.makedirs("log/cmeee/", exist_ok=True)
    os.makedirs("test_res/cmeee/", exist_ok=True)
    
    # 运行命令
    result = os.system(" ".join(cmd))
    
    if result == 0:
        print("训练完成！")
    else:
        print(f"训练失败，错误代码: {result}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        
        if dataset == "msra":
            run_msra_simple()
        elif dataset == "cluener":
            run_cluener_simple()
        elif dataset == "cmeee":
            run_cmeee_simple()
        else:
            print("用法: python run_simple.py [msra|cluener|cmeee]")
    else:
        print("用法: python run_simple.py [msra|cluener|cmeee]")
        print("例如: python run_simple.py msra")

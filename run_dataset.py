#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行不同中文数据集的脚本
支持MSRA、CLUENER、CMEEE等数据集
"""

import os
import sys

def run_msra():
    """运行MSRA数据集"""
    base_cmd = [
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
    print("命令:", " ".join(base_cmd))
    
    os.makedirs("results/msra/", exist_ok=True)
    os.makedirs("log/msra/", exist_ok=True)
    os.makedirs("test_res/msra/", exist_ok=True)
    
    os.system(" ".join(base_cmd))

def run_cluener():
    """运行CLUENER数据集"""
    base_cmd = [
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
    print("命令:", " ".join(base_cmd))
    
    os.makedirs("results/cluener/", exist_ok=True)
    os.makedirs("log/cluener/", exist_ok=True)
    os.makedirs("test_res/cluener/", exist_ok=True)
    
    os.system(" ".join(base_cmd))

def run_cmeee():
    """运行CMEEE数据集"""
    base_cmd = [
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
    print("命令:", " ".join(base_cmd))
    
    os.makedirs("results/cmeee/", exist_ok=True)
    os.makedirs("log/cmeee/", exist_ok=True)
    os.makedirs("test_res/cmeee/", exist_ok=True)
    
    os.system(" ".join(base_cmd))

def run_with_large_model(dataset_name):
    """使用BERT-Large模型运行指定数据集"""
    base_cmd = [
        "python", "main.py",
        "--dataname", dataset_name,
        "--data_dir", f"/root/autodl-tmp/HybridNER/dataset/{dataset_name}",
        "--bert_config_dir", "/root/autodl-tmp/HybridNER/models/bert-large-cased",
        "--batch_size", "16",
        "--lr", "2e-5",
        "--iteration", "30",
        "--bert_max_length", "128",
        "--max_spanLen", "5",
        "--model_save_dir", f"results/{dataset_name}_large/",
        "--logger_dir", f"log/{dataset_name}_large/",
        "--results_dir", f"test_res/{dataset_name}_large/",
    ]
    
    print(f"开始运行{dataset_name}数据集训练（使用BERT-Large）...")
    print("命令:", " ".join(base_cmd))
    
    os.makedirs(f"results/{dataset_name}_large/", exist_ok=True)
    os.makedirs(f"log/{dataset_name}_large/", exist_ok=True)
    os.makedirs(f"test_res/{dataset_name}_large/", exist_ok=True)
    
    os.system(" ".join(base_cmd))

def run_inference(dataset_name):
    """运行推理"""
    base_cmd = [
        "python", "main.py",
        "--state", "inference",
        "--dataname", dataset_name,
        "--data_dir", f"/root/autodl-tmp/HybridNER/dataset/{dataset_name}",
        "--bert_config_dir", "/root/autodl-tmp/HybridNER/models/bert-base-cased",
        "--inference_model", f"results/{dataset_name}/best_model.pkl",
        "--batch_size", "32",
        "--bert_max_length", "128",
        "--max_spanLen", "5",
        "--results_dir", f"test_res/{dataset_name}_inference/",
    ]
    
    print(f"开始运行{dataset_name}数据集推理...")
    print("命令:", " ".join(base_cmd))
    
    os.makedirs(f"test_res/{dataset_name}_inference/", exist_ok=True)
    
    os.system(" ".join(base_cmd))

def list_datasets():
    """列出可用的数据集"""
    dataset_dir = "/root/autodl-tmp/HybridNER/dataset"
    if os.path.exists(dataset_dir):
        print("可用的数据集:")
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                print(f"  - {item}")
    else:
        print(f"数据集目录不存在: {dataset_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "msra":
            run_msra()
        elif command == "cluener":
            run_cluener()
        elif command == "cmeee":
            run_cmeee()
        elif command == "list":
            list_datasets()
        elif command.startswith("large_"):
            dataset_name = command[6:]  # 移除"large_"前缀
            run_with_large_model(dataset_name)
        elif command.startswith("inference_"):
            dataset_name = command[10:]  # 移除"inference_"前缀
            run_inference(dataset_name)
        else:
            print("用法:")
            print("  python run_dataset.py msra          # 运行MSRA数据集")
            print("  python run_dataset.py cluener       # 运行CLUENER数据集")
            print("  python run_dataset.py cmeee         # 运行CMEEE数据集")
            print("  python run_dataset.py large_msra    # 使用BERT-Large运行MSRA")
            print("  python run_dataset.py large_cluener # 使用BERT-Large运行CLUENER")
            print("  python run_dataset.py large_cmeee   # 使用BERT-Large运行CMEEE")
            print("  python run_dataset.py inference_msra    # 运行MSRA推理")
            print("  python run_dataset.py inference_cluener # 运行CLUENER推理")
            print("  python run_dataset.py inference_cmeee   # 运行CMEEE推理")
            print("  python run_dataset.py list          # 列出可用数据集")
    else:
        print("用法: python run_dataset.py [dataset_name|list]")
        print("可用的数据集: msra, cluener, cmeee")
        print("使用 'list' 查看所有可用数据集")

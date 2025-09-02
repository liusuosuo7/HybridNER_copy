#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Navigation数据集优化训练脚本
针对F1值提升到70%的优化策略
"""

import os
import sys
import subprocess
import json

def create_optimized_config():
    """创建优化的配置文件"""
    
    # 优化策略1: 使用更细粒度的标签映射
    navigation_labels = {
        "O": 0,
        "对抗国家": 1,
        "对抗时间": 2, 
        "对抗地点": 3,
        "硬摧毁武器": 4,
        "压制式干扰技术": 5,
        "全球定位系统": 6,
        "北斗卫星导航系统": 7,
        "格洛纳斯卫星导航系统": 8,
        "伽利略卫星导航系统": 9,
        "系统端防御技术": 10,
        "压制式干扰装备": 11,
        "对抗单位": 12
    }
    
    # 优化策略2: 调整超参数
    optimized_params = {
        "--dataname": "navigation",
        "--data_dir": "/root/autodl-tmp/HybridNER/dataset/navigation",
        "--bert_config_dir": "/root/autodl-tmp/HybridNER/models/bert-base-chinese",
        "--batch_size": "16",  # 减小批次大小，提高稳定性
        "--lr": "2e-5",  # 降低学习率，提高收敛稳定性
        "--max_spanLen": "8",  # 增加最大span长度，捕获更长实体
        "--bert_max_length": "256",  # 增加序列长度
        "--iteration": "50",  # 增加训练轮数
        "--loss": "ce",  # 使用交叉熵损失
        "--etrans_func": "softmax",  # 使用softmax激活
        "--model_save_dir": "/root/autodl-tmp/HybridNER/output/navigation_optimized",
        "--logger_dir": "./log/navigation_optimized",
        "--results_dir": "./results/navigation_optimized",
        "--warmup_steps": "200",  # 增加warmup步数
        "--weight_decay": "0.01",  # 添加权重衰减
        "--model_dropout": "0.1",  # 降低dropout
        "--bert_dropout": "0.1",  # 降低BERT dropout
        "--early_stop": "10",  # 增加早停耐心
        "--clip_grad": "True",  # 启用梯度裁剪
    }
    
    return optimized_params

def run_optimized_training():
    """运行优化训练"""
    
    print("=== Navigation数据集优化训练 ===")
    
    # 创建必要的目录
    os.makedirs("/root/autodl-tmp/HybridNER/output/navigation_optimized", exist_ok=True)
    os.makedirs("./log/navigation_optimized", exist_ok=True)
    os.makedirs("./results/navigation_optimized", exist_ok=True)
    
    # 获取优化参数
    params = create_optimized_config()
    
    # 构建命令
    cmd = ["CUDA_VISIBLE_DEVICES=0", "python", "main.py"]
    for key, value in params.items():
        cmd.extend([key, value])
    
    print("执行命令:", " ".join(cmd))
    
    # 运行训练
    try:
        result = subprocess.run(" ".join(cmd), shell=True, check=True)
        print("训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return False

def run_ablation_study():
    """运行消融实验，测试不同配置的效果"""
    
    print("\n=== 消融实验 ===")
    
    # 实验配置列表
    experiments = [
        {
            "name": "baseline",
            "params": {
                "--batch_size": "16",
                "--lr": "2e-5",
                "--max_spanLen": "8",
                "--bert_max_length": "256",
            }
        },
        {
            "name": "high_lr",
            "params": {
                "--batch_size": "16",
                "--lr": "5e-5",
                "--max_spanLen": "8",
                "--bert_max_length": "256",
            }
        },
        {
            "name": "longer_span",
            "params": {
                "--batch_size": "16",
                "--lr": "2e-5",
                "--max_spanLen": "12",
                "--bert_max_length": "256",
            }
        },
        {
            "name": "longer_seq",
            "params": {
                "--batch_size": "8",
                "--lr": "2e-5",
                "--max_spanLen": "8",
                "--bert_max_length": "512",
            }
        }
    ]
    
    for exp in experiments:
        print(f"\n--- 实验: {exp['name']} ---")
        
        # 创建实验目录
        exp_dir = f"/root/autodl-tmp/HybridNER/output/navigation_{exp['name']}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # 构建命令
        cmd = [
            "CUDA_VISIBLE_DEVICES=0", "python", "main.py",
            "--dataname", "navigation",
            "--data_dir", "/root/autodl-tmp/HybridNER/dataset/navigation",
            "--bert_config_dir", "/root/autodl-tmp/HybridNER/models/bert-base-chinese",
            "--model_save_dir", exp_dir,
            "--logger_dir", f"./log/navigation_{exp['name']}",
            "--results_dir", f"./results/navigation_{exp['name']}",
            "--iteration", "30",
            "--loss", "ce",
            "--etrans_func", "softmax",
        ]
        
        # 添加实验特定参数
        for key, value in exp['params'].items():
            cmd.extend([key, value])
        
        print("命令:", " ".join(cmd))
        
        # 运行实验
        try:
            result = subprocess.run(" ".join(cmd), shell=True, check=True)
            print(f"实验 {exp['name']} 完成!")
        except subprocess.CalledProcessError as e:
            print(f"实验 {exp['name']} 失败: {e}")

def create_data_analysis_script():
    """创建数据分析脚本"""
    
    script_content = '''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Navigation数据集分析脚本
"""

import json
import os
from collections import Counter

def analyze_dataset():
    """分析数据集特征"""
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    
    for split in ['train', 'dev', 'test']:
        filename = f"navigation_{split}_span.json"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            continue
            
        print(f"\\n=== 分析 {split} 集 ===")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 基本统计
        print(f"样本数量: {len(data)}")
        
        # 文本长度统计
        text_lengths = []
        entity_counts = []
        entity_types = Counter()
        
        for item in data:
            # 获取文本
            text = item.get('context', item.get('text', ''))
            text_lengths.append(len(text))
            
            # 统计实体
            entities = item.get('entities', [])
            entity_counts.append(len(entities))
            
            for ent in entities:
                entity_types[ent.get('label', 'unknown')] += 1
        
        print(f"平均文本长度: {sum(text_lengths)/len(text_lengths):.1f}")
        print(f"平均实体数量: {sum(entity_counts)/len(entity_counts):.1f}")
        print(f"实体类型分布:")
        for ent_type, count in entity_types.most_common():
            print(f"  {ent_type}: {count}")
        
        # 实体长度统计
        entity_lengths = []
        for item in data:
            entities = item.get('entities', [])
            for ent in entities:
                start = ent.get('start_offset', ent.get('start', 0))
                end = ent.get('end_offset', ent.get('end', 0))
                entity_lengths.append(end - start)
        
        if entity_lengths:
            print(f"平均实体长度: {sum(entity_lengths)/len(entity_lengths):.1f}")
            print(f"最大实体长度: {max(entity_lengths)}")
            print(f"最小实体长度: {min(entity_lengths)}")

if __name__ == "__main__":
    analyze_dataset()
'''
    
    with open("analyze_navigation_data.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("数据分析脚本已创建: analyze_navigation_data.py")

if __name__ == "__main__":
    print("Navigation数据集优化训练工具")
    print("1. 运行优化训练")
    print("2. 运行消融实验")
    print("3. 创建数据分析脚本")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        run_optimized_training()
    elif choice == "2":
        run_ablation_study()
    elif choice == "3":
        create_data_analysis_script()
    else:
        print("无效选择")

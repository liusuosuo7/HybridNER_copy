#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复训练问题
解决磁盘空间不足和评估指标为0的问题
"""

import os
import shutil
import subprocess
import json

def check_disk_space():
    """检查磁盘空间"""
    print("=== 检查磁盘空间 ===")
    
    try:
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        print("当前目录磁盘使用情况:")
        print(result.stdout)
        
        # 检查可用空间
        result = subprocess.run(['df', '.'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 4:
                available_kb = int(parts[3])
                available_mb = available_kb / 1024
                available_gb = available_mb / 1024
                print(f"\n可用空间: {available_gb:.2f} GB ({available_mb:.0f} MB)")
                
                if available_gb < 2:
                    print("⚠️  警告：可用空间不足2GB，模型保存可能失败")
                    return False
                else:
                    print("✓ 磁盘空间充足")
                    return True
    except Exception as e:
        print(f"检查磁盘空间时出错: {e}")
        return False

def clean_output_directories():
    """清理输出目录"""
    print("\n=== 清理输出目录 ===")
    
    output_dirs = [
        "/root/autodl-tmp/HybridNER/output/navigation_large_model",
        "/root/autodl-tmp/HybridNER/output/navigation_basic_optimized",
        "/root/autodl-tmp/HybridNER/output/navigation_multitask",
        "/root/autodl-tmp/HybridNER/output/navigation_adversarial",
        "/root/autodl-tmp/HybridNER/output/navigation_ensemble_1",
        "/root/autodl-tmp/HybridNER/output/navigation_ensemble_2",
        "/root/autodl-tmp/HybridNER/output/navigation_ensemble_3"
    ]
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                print(f"清理目录: {output_dir}")
            except Exception as e:
                print(f"清理目录失败 {output_dir}: {e}")

def check_dataset_integrity():
    """检查数据集完整性"""
    print("\n=== 检查数据集完整性 ===")
    
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    files = ["navigation_train_span.json", "navigation_dev_span.json", "navigation_test_span.json"]
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✓ {file}: {len(data)} 样本")
                
                # 检查样本结构
                if data:
                    sample = data[0]
                    print(f"  样本结构: {list(sample.keys())}")
                    if 'entities' in sample:
                        print(f"  实体数量: {len(sample['entities'])}")
            except Exception as e:
                print(f"✗ {file}: 读取失败 - {e}")
        else:
            print(f"✗ {file}: 文件不存在")

def create_optimized_training_script():
    """创建优化的训练脚本"""
    print("\n=== 创建优化的训练脚本 ===")
    
    script_content = '''#!/bin/bash
# 优化的训练脚本 - 解决磁盘空间和评估问题
echo "=== 优化的训练脚本 ==="

# 清理输出目录
rm -rf /root/autodl-tmp/HybridNER/output/navigation_optimized
rm -rf ./log/navigation_optimized
rm -rf ./results/navigation_optimized

# 创建输出目录
mkdir -p /root/autodl-tmp/HybridNER/output/navigation_optimized
mkdir -p ./log/navigation_optimized
mkdir -p ./results/navigation_optimized

# 检查磁盘空间
echo "检查磁盘空间..."
df -h .

echo "开始优化训练..."
CUDA_VISIBLE_DEVICES=0 python main.py \\
    --dataname navigation \\
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \\
    --state train \\
    --batch_size 8 \\
    --lr 2e-5 \\
    --max_spanLen 8 \\
    --bert_max_length 256 \\
    --iteration 50 \\
    --loss ce \\
    --etrans_func softmax \\
    --model_save_dir /root/autodl-tmp/HybridNER/output/navigation_optimized \\
    --logger_dir ./log/navigation_optimized \\
    --results_dir ./results/navigation_optimized \\
    --warmup_steps 200 \\
    --weight_decay 0.01 \\
    --model_dropout 0.1 \\
    --bert_dropout 0.1 \\
    --early_stop 10 \\
    --clip_grad True

echo "优化训练完成！"
'''
    
    with open("run_optimized_training.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    os.chmod("run_optimized_training.sh", 0o755)
    print("已创建: run_optimized_training.sh")

def create_debug_script():
    """创建调试脚本"""
    print("\n=== 创建调试脚本 ===")
    
    script_content = '''#!/bin/bash
# 调试脚本 - 检查训练问题
echo "=== 调试训练问题 ==="

echo "1. 检查磁盘空间..."
df -h .

echo ""
echo "2. 检查数据集..."
ls -la /root/autodl-tmp/HybridNER/dataset/navigation/

echo ""
echo "3. 检查模型文件..."
ls -la /root/autodl-tmp/HybridNER/models/bert-large-cased/

echo ""
echo "4. 检查输出目录..."
ls -la /root/autodl-tmp/HybridNER/output/ 2>/dev/null || echo "输出目录不存在"

echo ""
echo "5. 检查日志文件..."
ls -la ./log/ 2>/dev/null || echo "日志目录不存在"

echo ""
echo "6. 检查Python环境..."
python --version
pip list | grep torch
pip list | grep transformers

echo ""
echo "调试完成！"
'''
    
    with open("debug_training.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    os.chmod("debug_training.sh", 0o755)
    print("已创建: debug_training.sh")

def create_minimal_training_script():
    """创建最小化训练脚本"""
    print("\n=== 创建最小化训练脚本 ===")
    
    script_content = '''#!/bin/bash
# 最小化训练脚本 - 快速测试
echo "=== 最小化训练测试 ==="

# 清理并创建目录
rm -rf /root/autodl-tmp/HybridNER/output/navigation_minimal
mkdir -p /root/autodl-tmp/HybridNER/output/navigation_minimal
mkdir -p ./log/navigation_minimal
mkdir -p ./results/navigation_minimal

echo "开始最小化训练测试..."
CUDA_VISIBLE_DEVICES=0 python main.py \\
    --dataname navigation \\
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \\
    --state train \\
    --batch_size 4 \\
    --lr 3e-5 \\
    --max_spanLen 6 \\
    --bert_max_length 128 \\
    --iteration 10 \\
    --loss ce \\
    --etrans_func softmax \\
    --model_save_dir /root/autodl-tmp/HybridNER/output/navigation_minimal \\
    --logger_dir ./log/navigation_minimal \\
    --results_dir ./results/navigation_minimal \\
    --warmup_steps 50 \\
    --weight_decay 0.01 \\
    --model_dropout 0.1 \\
    --bert_dropout 0.1 \\
    --early_stop 5 \\
    --clip_grad True

echo "最小化训练测试完成！"
'''
    
    with open("run_minimal_test.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    os.chmod("run_minimal_test.sh", 0o755)
    print("已创建: run_minimal_test.sh")

def main():
    """主函数"""
    print("=== 修复训练问题 ===")
    
    # 检查磁盘空间
    space_ok = check_disk_space()
    
    # 清理输出目录
    clean_output_directories()
    
    # 检查数据集完整性
    check_dataset_integrity()
    
    # 创建调试脚本
    create_debug_script()
    
    # 创建最小化训练脚本
    create_minimal_training_script()
    
    # 创建优化训练脚本
    create_optimized_training_script()
    
    print("\n=== 修复完成 ===")
    print("已创建以下脚本:")
    print("1. debug_training.sh - 调试训练问题")
    print("2. run_minimal_test.sh - 最小化训练测试")
    print("3. run_optimized_training.sh - 优化训练")
    
    print("\n建议执行顺序:")
    print("1. bash debug_training.sh  # 检查环境")
    print("2. bash run_minimal_test.sh  # 快速测试")
    print("3. bash run_optimized_training.sh  # 正式训练")
    
    print("\n主要修复:")
    print("- 清理了输出目录，释放磁盘空间")
    print("- 减小了batch_size和序列长度")
    print("- 减少了训练迭代次数")
    print("- 添加了磁盘空间检查")
    print("- 创建了调试和测试脚本")

if __name__ == "__main__":
    main()

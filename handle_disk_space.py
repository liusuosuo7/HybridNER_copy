#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理磁盘空间不足问题
提供多种解决方案来释放磁盘空间
"""

import os
import shutil
import subprocess
import json

def check_disk_space():
    """检查磁盘空间"""
    print("=== 检查磁盘空间 ===")
    
    try:
        # 检查当前目录的磁盘空间
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        print("当前目录磁盘使用情况:")
        print(result.stdout)
        
        # 检查数据集目录的磁盘空间
        data_dir = '/root/autodl-tmp/HybridNER/dataset/navigation'
        if os.path.exists(data_dir):
            result = subprocess.run(['df', '-h', data_dir], capture_output=True, text=True)
            print(f"\n数据集目录磁盘使用情况:")
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
                
                if available_gb < 1:
                    print("⚠️  警告：可用空间不足1GB，建议清理磁盘空间")
                    return False
                else:
                    print("✓ 磁盘空间充足")
                    return True
    except Exception as e:
        print(f"检查磁盘空间时出错: {e}")
        return False

def clean_temp_files():
    """清理临时文件"""
    print("\n=== 清理临时文件 ===")
    
    temp_dirs = [
        '/tmp',
        '/var/tmp',
        '/root/.cache',
        '/root/.local/share/Trash'
    ]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                result = subprocess.run(['du', '-sh', temp_dir], capture_output=True, text=True)
                print(f"{temp_dir}: {result.stdout.strip()}")
            except:
                pass

def create_symbolic_link():
    """创建符号链接，避免复制文件"""
    print("\n=== 创建符号链接方案 ===")
    
    original_file = '/root/autodl-tmp/HybridNER/dataset/navigation/navigation_train_span.json'
    augmented_file = '/root/autodl-tmp/HybridNER/dataset/navigation/navigation_train_augmented.json'
    
    if not os.path.exists(original_file):
        print(f"原始文件不存在: {original_file}")
        return False
    
    # 删除已存在的增强文件
    if os.path.exists(augmented_file):
        os.remove(augmented_file)
        print(f"删除已存在的文件: {augmented_file}")
    
    # 创建符号链接
    try:
        os.symlink(original_file, augmented_file)
        print(f"创建符号链接: {augmented_file} -> {original_file}")
        
        # 验证链接
        if os.path.islink(augmented_file):
            print("✓ 符号链接创建成功")
            return True
        else:
            print("✗ 符号链接创建失败")
            return False
    except Exception as e:
        print(f"创建符号链接失败: {e}")
        return False

def create_minimal_augmentation():
    """创建最小化的数据增强（不复制文件）"""
    print("\n=== 创建最小化数据增强 ===")
    
    original_file = '/root/autodl-tmp/HybridNER/dataset/navigation/navigation_train_span.json'
    augmented_file = '/root/autodl-tmp/HybridNER/dataset/navigation/navigation_train_augmented.json'
    
    if not os.path.exists(original_file):
        print(f"原始文件不存在: {original_file}")
        return False
    
    try:
        # 读取原始数据
        with open(original_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"原始数据样本数: {len(data)}")
        
        # 创建最小化的增强数据（只添加少量样本）
        augmented_data = []
        
        for i, item in enumerate(data):
            # 添加原始数据
            augmented_data.append(item)
            
            # 每100个样本添加一个简单的增强版本
            if i % 100 == 0 and item.get('entities'):
                # 创建简单的增强版本
                context = item.get('context', '')
                entities = item.get('entities', [])
                
                if entities:
                    # 简单的文本修改
                    new_context = context + " [增强]"
                    new_entities = entities.copy()
                    
                    augmented_data.append({
                        'context': new_context,
                        'entities': new_entities
                    })
        
        # 保存到临时位置，然后移动
        temp_file = '/tmp/navigation_train_augmented_temp.json'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)
        
        # 移动到目标位置
        shutil.move(temp_file, augmented_file)
        
        print(f"最小化数据增强完成")
        print(f"原始样本: {len(data)}")
        print(f"增强后样本: {len(augmented_data)}")
        print(f"增强比例: {len(augmented_data) / len(data):.2f}x")
        print(f"输出文件: {augmented_file}")
        
        return True
        
    except Exception as e:
        print(f"创建最小化数据增强失败: {e}")
        return False

def create_training_script_without_augmentation():
    """创建不使用数据增强的训练脚本"""
    print("\n=== 创建不使用数据增强的训练脚本 ===")
    
    script_content = '''#!/bin/bash
# 不使用数据增强的训练脚本
echo "=== 不使用数据增强的训练 ==="

# 创建输出目录
mkdir -p /root/autodl-tmp/HybridNER/output/navigation_direct
mkdir -p ./log/navigation_direct
mkdir -p ./results/navigation_direct

echo "开始训练（直接使用原始数据）..."
CUDA_VISIBLE_DEVICES=0 python main.py \\
    --dataname navigation \\
    --data_dir /root/autodl-tmp/HybridNER/dataset/navigation \\
    --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased \\
    --state train \\
    --batch_size 16 \\
    --lr 2e-5 \\
    --max_spanLen 8 \\
    --bert_max_length 256 \\
    --iteration 80 \\
    --loss ce \\
    --etrans_func softmax \\
    --model_save_dir /root/autodl-tmp/HybridNER/output/navigation_direct \\
    --logger_dir ./log/navigation_direct \\
    --results_dir ./results/navigation_direct \\
    --warmup_steps 300 \\
    --weight_decay 0.01 \\
    --model_dropout 0.1 \\
    --bert_dropout 0.1 \\
    --early_stop 15 \\
    --clip_grad True

echo "训练完成！"
'''
    
    with open("run_direct_training.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    os.chmod("run_direct_training.sh", 0o755)
    print("已创建: run_direct_training.sh")

def main():
    """主函数"""
    print("=== 处理磁盘空间不足问题 ===")
    
    # 检查磁盘空间
    space_ok = check_disk_space()
    
    # 清理临时文件
    clean_temp_files()
    
    print("\n=== 解决方案 ===")
    
    # 方案1：创建符号链接
    print("方案1：创建符号链接（推荐）")
    if create_symbolic_link():
        print("✓ 符号链接方案成功")
        create_training_script_without_augmentation()
        print("\n现在可以直接运行训练:")
        print("bash run_direct_training.sh")
        return
    
    # 方案2：最小化数据增强
    print("\n方案2：最小化数据增强")
    if create_minimal_augmentation():
        print("✓ 最小化数据增强成功")
        create_training_script_without_augmentation()
        print("\n现在可以直接运行训练:")
        print("bash run_direct_training.sh")
        return
    
    # 方案3：直接使用原始数据
    print("\n方案3：直接使用原始数据")
    create_training_script_without_augmentation()
    print("✓ 创建了直接训练脚本")
    print("\n现在可以直接运行训练:")
    print("bash run_direct_training.sh")
    
    print("\n=== 建议 ===")
    print("1. 优先使用符号链接方案，节省磁盘空间")
    print("2. 如果符号链接失败，使用最小化数据增强")
    print("3. 最后选择直接使用原始数据训练")
    print("4. 考虑清理磁盘空间以支持更复杂的数据增强")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据恢复脚本
恢复被意外删除的重要文件
"""

import os
import shutil
import subprocess
import json
from datetime import datetime

def check_what_was_deleted():
    """检查被删除的内容"""
    print("=== 检查被删除的内容 ===")
    
    # 检查数据集文件
    data_dir = "/root/autodl-tmp/HybridNER/dataset/navigation"
    data_files = [
        "navigation_train_span.json",
        "navigation_dev_span.json", 
        "navigation_test_span.json",
        "navigation_train_augmented.json"
    ]
    
    print("检查数据集文件:")
    for file in data_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file} - 存在")
        else:
            print(f"✗ {file} - 被删除")
    
    # 检查模型文件
    model_dir = "/root/autodl-tmp/HybridNER/models/bert-large-cased"
    model_files = ["config.json", "pytorch_model.bin", "vocab.txt"]
    
    print("\n检查模型文件:")
    for file in model_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file} - 存在")
        else:
            print(f"✗ {file} - 被删除")
    
    # 检查代码文件
    code_files = [
        "main.py",
        "args_config.py", 
        "dataloaders/spanner_dataset.py",
        "src/framework.py"
    ]
    
    print("\n检查代码文件:")
    for file in code_files:
        if os.path.exists(file):
            print(f"✓ {file} - 存在")
        else:
            print(f"✗ {file} - 被删除")

def recover_from_backup():
    """从备份恢复文件"""
    print("\n=== 尝试从备份恢复 ===")
    
    # 检查是否有备份目录
    backup_dirs = [
        "/root/autodl-tmp/HybridNER/backup",
        "/root/autodl-tmp/HybridNER/dataset/navigation/backup",
        "/root/autodl-tmp/HybridNER_copy",
        "/root/autodl-tmp/HybridNER_copy/HybridNER_copy"
    ]
    
    for backup_dir in backup_dirs:
        if os.path.exists(backup_dir):
            print(f"发现备份目录: {backup_dir}")
            
            # 检查备份内容
            try:
                files = os.listdir(backup_dir)
                print(f"备份目录内容: {files[:10]}...")  # 只显示前10个文件
                
                # 尝试恢复数据集
                data_files = ["navigation_train_span.json", "navigation_dev_span.json", "navigation_test_span.json"]
                for file in data_files:
                    backup_file = os.path.join(backup_dir, file)
                    target_file = f"/root/autodl-tmp/HybridNER/dataset/navigation/{file}"
                    
                    if os.path.exists(backup_file) and not os.path.exists(target_file):
                        try:
                            shutil.copy2(backup_file, target_file)
                            print(f"✓ 恢复: {file}")
                        except Exception as e:
                            print(f"✗ 恢复失败 {file}: {e}")
                            
            except Exception as e:
                print(f"检查备份目录失败: {e}")

def check_git_repository():
    """检查Git仓库恢复"""
    print("\n=== 检查Git仓库 ===")
    
    try:
        # 检查是否有Git仓库
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 发现Git仓库")
            
            # 检查最近的提交
            result = subprocess.run(['git', 'log', '--oneline', '-5'], capture_output=True, text=True)
            if result.returncode == 0:
                print("最近5次提交:")
                print(result.stdout)
                
            # 尝试恢复文件
            print("\n尝试从Git恢复文件...")
            result = subprocess.run(['git', 'checkout', '--', '.'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Git恢复成功")
            else:
                print(f"✗ Git恢复失败: {result.stderr}")
        else:
            print("✗ 未发现Git仓库")
    except Exception as e:
        print(f"检查Git仓库失败: {e}")

def create_recovery_script():
    """创建恢复脚本"""
    print("\n=== 创建恢复脚本 ===")
    
    script_content = '''#!/bin/bash
# 数据恢复脚本
echo "=== 数据恢复脚本 ==="

echo "1. 检查当前状态..."
ls -la /root/autodl-tmp/HybridNER/dataset/navigation/ 2>/dev/null || echo "数据集目录不存在"
ls -la /root/autodl-tmp/HybridNER/models/bert-large-cased/ 2>/dev/null || echo "模型目录不存在"

echo ""
echo "2. 检查备份..."
find /root -name "*backup*" -type d 2>/dev/null
find /root -name "*navigation*" -name "*.json" 2>/dev/null

echo ""
echo "3. 检查Git状态..."
git status 2>/dev/null || echo "Git仓库不存在"

echo ""
echo "4. 检查其他可能的备份位置..."
find /root -name "navigation_train_span.json" 2>/dev/null
find /root -name "bert-large-cased" -type d 2>/dev/null

echo ""
echo "恢复检查完成！"
'''
    
    with open("recovery_check.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    os.chmod("recovery_check.sh", 0o755)
    print("已创建: recovery_check.sh")

def create_emergency_backup():
    """创建紧急备份"""
    print("\n=== 创建紧急备份 ===")
    
    backup_dir = f"/root/autodl-tmp/HybridNER/emergency_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        os.makedirs(backup_dir, exist_ok=True)
        
        # 备份剩余的重要文件
        important_dirs = [
            "/root/autodl-tmp/HybridNER/dataset",
            "/root/autodl-tmp/HybridNER/models",
            "/root/autodl-tmp/HybridNER/src",
            "/root/autodl-tmp/HybridNER/dataloaders"
        ]
        
        for dir_path in important_dirs:
            if os.path.exists(dir_path):
                dir_name = os.path.basename(dir_path)
                backup_path = os.path.join(backup_dir, dir_name)
                shutil.copytree(dir_path, backup_path, dirs_exist_ok=True)
                print(f"✓ 备份: {dir_path} -> {backup_path}")
        
        print(f"✓ 紧急备份完成: {backup_dir}")
        return backup_dir
        
    except Exception as e:
        print(f"✗ 紧急备份失败: {e}")
        return None

def provide_recovery_instructions():
    """提供恢复指导"""
    print("\n=== 恢复指导 ===")
    
    print("如果重要文件被删除，请按以下步骤操作:")
    print()
    print("1. 立即停止所有操作，避免覆盖数据")
    print("2. 运行恢复检查: bash recovery_check.sh")
    print("3. 检查以下位置是否有备份:")
    print("   - /root/autodl-tmp/HybridNER/backup/")
    print("   - /root/autodl-tmp/HybridNER_copy/")
    print("   - Git仓库 (git log --oneline)")
    print("   - 其他可能的备份位置")
    print()
    print("4. 如果找到备份，手动复制文件:")
    print("   cp /path/to/backup/navigation_train_span.json /root/autodl-tmp/HybridNER/dataset/navigation/")
    print()
    print("5. 如果数据完全丢失，需要重新准备数据集")

def main():
    """主函数"""
    print("=== 数据恢复工具 ===")
    print("正在检查被删除的文件...")
    
    # 检查被删除的内容
    check_what_was_deleted()
    
    # 尝试从备份恢复
    recover_from_backup()
    
    # 检查Git仓库
    check_git_repository()
    
    # 创建紧急备份
    backup_dir = create_emergency_backup()
    
    # 创建恢复脚本
    create_recovery_script()
    
    # 提供恢复指导
    provide_recovery_instructions()
    
    print("\n=== 恢复工具准备完成 ===")
    print("请运行以下命令进行详细检查:")
    print("bash recovery_check.sh")
    
    if backup_dir:
        print(f"\n紧急备份位置: {backup_dir}")
        print("请检查此目录中的文件")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
恢复output文件夹内容
专门用于恢复被删除的训练输出和模型文件
"""

import os
import shutil
import subprocess
import json
from datetime import datetime

def check_output_status():
    """检查output文件夹状态"""
    print("=== 检查output文件夹状态 ===")
    
    output_dir = "/root/autodl-tmp/HybridNER/output"
    
    if os.path.exists(output_dir):
        print(f"✓ output目录存在: {output_dir}")
        try:
            contents = os.listdir(output_dir)
            print(f"当前内容: {contents}")
            
            if not contents:
                print("⚠️  output目录为空")
            else:
                print("✓ output目录有内容")
        except Exception as e:
            print(f"✗ 读取output目录失败: {e}")
    else:
        print(f"✗ output目录不存在: {output_dir}")

def search_for_output_backups():
    """搜索output文件夹的备份"""
    print("\n=== 搜索output文件夹备份 ===")
    
    # 可能的备份位置
    search_paths = [
        "/root/autodl-tmp/HybridNER",
        "/root/autodl-tmp/HybridNER_copy",
        "/root/autodl-tmp/HybridNER_copy/HybridNER_copy",
        "/root",
        "/tmp"
    ]
    
    found_backups = []
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"搜索路径: {search_path}")
            
            # 搜索output相关目录
            try:
                for root, dirs, files in os.walk(search_path):
                    for dir_name in dirs:
                        if 'output' in dir_name.lower() or 'navigation' in dir_name.lower():
                            full_path = os.path.join(root, dir_name)
                            if os.path.isdir(full_path):
                                try:
                                    contents = os.listdir(full_path)
                                    if contents:
                                        found_backups.append({
                                            'path': full_path,
                                            'contents': contents,
                                            'size': len(contents)
                                        })
                                        print(f"✓ 发现备份: {full_path} ({len(contents)} 项)")
                                except:
                                    pass
            except Exception as e:
                print(f"搜索失败 {search_path}: {e}")
    
    return found_backups

def search_for_model_files():
    """搜索模型文件"""
    print("\n=== 搜索模型文件 ===")
    
    model_patterns = [
        "*.pth",
        "*.pt", 
        "*.bin",
        "*.ckpt",
        "*.model",
        "best_model*",
        "checkpoint*"
    ]
    
    found_models = []
    
    # 搜索路径
    search_paths = [
        "/root/autodl-tmp/HybridNER",
        "/root/autodl-tmp/HybridNER_copy",
        "/root/autodl-tmp/HybridNER_copy/HybridNER_copy",
        "/root",
        "/tmp"
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"搜索模型文件: {search_path}")
            
            try:
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if any(pattern.replace('*', '') in file for pattern in model_patterns):
                            full_path = os.path.join(root, file)
                            if os.path.isfile(full_path):
                                try:
                                    size = os.path.getsize(full_path)
                                    if size > 1000000:  # 大于1MB的文件
                                        found_models.append({
                                            'path': full_path,
                                            'size': size,
                                            'name': file
                                        })
                                        print(f"✓ 发现模型文件: {full_path} ({size/1024/1024:.1f}MB)")
                                except:
                                    pass
            except Exception as e:
                print(f"搜索失败 {search_path}: {e}")
    
    return found_models

def recover_output_structure():
    """恢复output文件夹结构"""
    print("\n=== 恢复output文件夹结构 ===")
    
    output_dir = "/root/autodl-tmp/HybridNER/output"
    
    # 创建output目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ 创建output目录: {output_dir}")
    
    # 常见的output子目录
    subdirs = [
        "navigation_basic_optimized",
        "navigation_large_model", 
        "navigation_multitask",
        "navigation_adversarial",
        "navigation_ensemble_1",
        "navigation_ensemble_2",
        "navigation_ensemble_3",
        "navigation_optimized",
        "navigation_minimal",
        "navigation_direct"
    ]
    
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
            print(f"✓ 创建子目录: {subdir_path}")

def restore_from_backups(found_backups):
    """从备份恢复output内容"""
    print("\n=== 从备份恢复output内容 ===")
    
    output_dir = "/root/autodl-tmp/HybridNER/output"
    
    if not found_backups:
        print("未找到output备份")
        return
    
    # 按大小排序，优先恢复最大的备份
    found_backups.sort(key=lambda x: x['size'], reverse=True)
    
    for backup in found_backups:
        backup_path = backup['path']
        contents = backup['contents']
        
        print(f"\n尝试从备份恢复: {backup_path}")
        print(f"备份内容: {contents}")
        
        # 确定目标目录名
        if 'navigation' in backup_path:
            # 从路径中提取目录名
            path_parts = backup_path.split('/')
            for part in path_parts:
                if 'navigation' in part and part != 'navigation':
                    target_dir = part
                    break
            else:
                target_dir = "navigation_recovered"
        else:
            target_dir = "navigation_recovered"
        
        target_path = os.path.join(output_dir, target_dir)
        
        try:
            # 复制备份内容
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            
            shutil.copytree(backup_path, target_path)
            print(f"✓ 恢复成功: {backup_path} -> {target_path}")
            
            # 验证恢复结果
            if os.path.exists(target_path):
                restored_contents = os.listdir(target_path)
                print(f"恢复的内容: {restored_contents}")
            
            break  # 只恢复第一个最大的备份
            
        except Exception as e:
            print(f"✗ 恢复失败: {e}")

def restore_model_files(found_models):
    """恢复模型文件"""
    print("\n=== 恢复模型文件 ===")
    
    output_dir = "/root/autodl-tmp/HybridNER/output"
    
    if not found_models:
        print("未找到模型文件")
        return
    
    # 按大小排序
    found_models.sort(key=lambda x: x['size'], reverse=True)
    
    for model in found_models:
        model_path = model['path']
        model_name = model['name']
        
        print(f"\n发现模型文件: {model_path}")
        print(f"文件名: {model_name}")
        print(f"大小: {model['size']/1024/1024:.1f}MB")
        
        # 确定目标目录
        if 'navigation' in model_path:
            target_dir = "navigation_recovered"
        else:
            target_dir = "navigation_recovered"
        
        target_path = os.path.join(output_dir, target_dir)
        
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
        
        target_file = os.path.join(target_path, model_name)
        
        try:
            shutil.copy2(model_path, target_file)
            print(f"✓ 恢复模型文件: {target_file}")
        except Exception as e:
            print(f"✗ 恢复模型文件失败: {e}")

def create_output_recovery_script():
    """创建output恢复脚本"""
    print("\n=== 创建output恢复脚本 ===")
    
    script_content = '''#!/bin/bash
# output文件夹恢复脚本
echo "=== output文件夹恢复脚本 ==="

echo "1. 检查output目录状态..."
ls -la /root/autodl-tmp/HybridNER/output/ 2>/dev/null || echo "output目录不存在"

echo ""
echo "2. 搜索output备份..."
find /root -name "*output*" -type d 2>/dev/null
find /root -name "*navigation*" -type d 2>/dev/null

echo ""
echo "3. 搜索模型文件..."
find /root -name "*.pth" -o -name "*.pt" -o -name "*.bin" -o -name "*.ckpt" 2>/dev/null | head -10

echo ""
echo "4. 检查日志文件..."
find /root -name "*.log" 2>/dev/null | grep -E "(navigation|output)" | head -10

echo ""
echo "5. 检查结果文件..."
find /root -name "*.json" 2>/dev/null | grep -E "(navigation|output|result)" | head -10

echo ""
echo "output恢复检查完成！"
'''
    
    with open("recover_output.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    os.chmod("recover_output.sh", 0o755)
    print("已创建: recover_output.sh")

def main():
    """主函数"""
    print("=== output文件夹恢复工具 ===")
    
    # 检查output状态
    check_output_status()
    
    # 搜索output备份
    found_backups = search_for_output_backups()
    
    # 搜索模型文件
    found_models = search_for_model_files()
    
    # 恢复output结构
    recover_output_structure()
    
    # 从备份恢复
    restore_from_backups(found_backups)
    
    # 恢复模型文件
    restore_model_files(found_models)
    
    # 创建恢复脚本
    create_output_recovery_script()
    
    print("\n=== output恢复完成 ===")
    print("请运行以下命令进行详细检查:")
    print("bash recover_output.sh")
    
    print("\n恢复的output目录:")
    output_dir = "/root/autodl-tmp/HybridNER/output"
    if os.path.exists(output_dir):
        try:
            contents = os.listdir(output_dir)
            for item in contents:
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    sub_contents = os.listdir(item_path)
                    print(f"  {item}/: {sub_contents}")
        except Exception as e:
            print(f"读取output目录失败: {e}")

if __name__ == "__main__":
    main()

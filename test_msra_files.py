#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试MSRA数据集文件是否存在
"""

import os

def test_msra_files():
    """测试MSRA数据集文件"""
    
    msra_dir = "/root/autodl-tmp/HybridNER/dataset/msra"
    
    print("=== 检查MSRA数据集文件 ===")
    
    if not os.path.exists(msra_dir):
        print(f"✗ 数据集目录不存在: {msra_dir}")
        return False
    
    print(f"✓ 数据集目录存在: {msra_dir}")
    
    # 检查文件
    files_to_check = [
        'msra_train_span.json',
        'msra_dev_span.json', 
        'msra_test_span.json'
    ]
    
    all_exist = True
    for filename in files_to_check:
        file_path = os.path.join(msra_dir, filename)
        if os.path.exists(file_path):
            print(f"✓ 文件存在: {filename}")
        else:
            print(f"✗ 文件不存在: {filename}")
            all_exist = False
    
    if all_exist:
        print("\n=== 所有MSRA文件都存在 ===")
        return True
    else:
        print("\n=== 部分MSRA文件缺失 ===")
        return False

if __name__ == "__main__":
    test_msra_files()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动下载中文BERT模型
"""

import os
from transformers import BertTokenizer, BertConfig, BertModel

def download_chinese_bert():
    """下载中文BERT模型"""
    
    model_dir = "/root/autodl-tmp/HybridNER/models/bert-base-chinese"
    
    print("=== 开始下载中文BERT模型 ===")
    
    # 创建目录
    os.makedirs(model_dir, exist_ok=True)
    print(f"✓ 创建目录: {model_dir}")
    
    try:
        print("正在下载tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=model_dir)
        tokenizer.save_pretrained(model_dir)
        print("✓ tokenizer下载完成")
        
        print("正在下载配置...")
        config = BertConfig.from_pretrained("bert-base-chinese", cache_dir=model_dir)
        config.save_pretrained(model_dir)
        print("✓ 配置下载完成")
        
        print("正在下载模型...")
        model = BertModel.from_pretrained("bert-base-chinese", cache_dir=model_dir)
        model.save_pretrained(model_dir)
        print("✓ 模型下载完成")
        
        # 验证文件
        files = os.listdir(model_dir)
        print(f"\n✓ 下载完成！文件列表:")
        for file in files:
            print(f"  - {file}")
            
        return True
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False

if __name__ == "__main__":
    download_chinese_bert()

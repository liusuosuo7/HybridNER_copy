#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复tokenizer问题 - 使用FastTokenizer
"""

from tokenizers import BertWordPieceTokenizer
import os

def create_fixed_loader():
    """创建修复后的数据加载器"""
    
    # 使用原始的BertWordPieceTokenizer，它支持offset mapping
    vocab_path = "/root/autodl-tmp/HybridNER/models/bert-base-cased/vocab.txt"
    
    if os.path.exists(vocab_path):
        print(f"✓ 找到vocab文件: {vocab_path}")
        tokenizer = BertWordPieceTokenizer(vocab_path)
        
        # 测试中文文本
        text = "张三在北京大学读书。"
        print(f"测试文本: {text}")
        
        try:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            print(f"✓ tokenization成功")
            print(f"input_ids: {tokens.ids}")
            print(f"offsets: {tokens.offsets}")
            print(f"长度匹配: {len(tokens.ids)} == {len(tokens.offsets)}")
            return True
        except Exception as e:
            print(f"✗ tokenization失败: {e}")
            return False
    else:
        print(f"✗ 找不到vocab文件: {vocab_path}")
        return False

if __name__ == "__main__":
    create_fixed_loader()

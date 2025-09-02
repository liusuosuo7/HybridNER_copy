#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试分词器工作
"""

from transformers import BertTokenizer

def test_tokenizer():
    """测试BERT分词器"""
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/HybridNER/models/bert-base-cased')
    
    # 测试中文文本
    text = "张三在北京大学读书。"
    
    print(f"原始文本: {text}")
    
    # 测试不同的编码方式
    try:
        # 方式1：使用 encode_plus with return_offsets_mapping
        print("\n方式1: encode_plus with return_offsets_mapping")
        encoded = tokenizer.encode_plus(text, add_special_tokens=True, return_offsets_mapping=True)
        print(f"input_ids length: {len(encoded['input_ids'])}")
        print(f"offset_mapping length: {len(encoded['offset_mapping'])}")
        print(f"input_ids: {encoded['input_ids']}")
        print(f"offset_mapping: {encoded['offset_mapping']}")
        print("✓ 成功")
    except Exception as e:
        print(f"✗ 失败: {e}")
    
    try:
        # 方式2：使用 encode_plus without return_offsets_mapping
        print("\n方式2: encode_plus without return_offsets_mapping")
        encoded = tokenizer.encode_plus(text, add_special_tokens=True)
        print(f"input_ids length: {len(encoded['input_ids'])}")
        print(f"input_ids: {encoded['input_ids']}")
        
        # 手动计算offset mapping
        tokens_text = tokenizer.tokenize(text)
        print(f"tokens: {tokens_text}")
        
        offsets = []
        current_pos = 0
        
        # 为 [CLS] token 添加 offset
        offsets.append((0, 0))
        
        for token in tokens_text:
            if token == '[CLS]' or token == '[SEP]':
                offsets.append((0, 0))
            else:
                start = text.find(token, current_pos)
                if start == -1:
                    start = current_pos
                end = start + len(token)
                offsets.append((start, end))
                current_pos = end
        
        # 为 [SEP] token 添加 offset
        offsets.append((0, 0))
        
        print(f"手动计算的offset_mapping: {offsets}")
        print(f"手动计算的offset_mapping length: {len(offsets)}")
        print(f"input_ids length: {len(encoded['input_ids'])}")
        
        if len(offsets) == len(encoded['input_ids']):
            print("✓ 长度匹配成功")
        else:
            print(f"✗ 长度不匹配: offsets={len(offsets)}, input_ids={len(encoded['input_ids'])}")
            
    except Exception as e:
        print(f"✗ 失败: {e}")

if __name__ == "__main__":
    test_tokenizer()

# -*- coding: utf-8 -*-
"""
中文数据集配置文件
"""

# 中文NER标签映射
CHINESE_LABEL_MAPPING = {
    "msra": {
        "PER": 1,  # 人名
        "LOC": 2,  # 地名
        "ORG": 3,  # 机构名
        "MISC": 4,  # 其他
        "O": 0      # 非实体
    },
    "weibo": {
        "PER": 1,  # 人名
        "LOC": 2,  # 地名
        "ORG": 3,  # 机构名
        "MISC": 4,  # 其他
        "O": 0      # 非实体
    },
    "ontonotes4": {
        "PER": 1,  # 人名
        "LOC": 2,  # 地名
        "ORG": 3,  # 机构名
        "MISC": 4,  # 其他
        "O": 0      # 非实体
    }
}

# 中文特征映射（基于字符特征）
CHINESE_MORPH_MAPPING = {
    "isdigit": 0,   # 数字
    "ispunct": 1,   # 标点符号
    "other": 2      # 其他字符
}

# 英文特征映射（基于大小写）
ENGLISH_MORPH_MAPPING = {
    "isupper": 0,   # 全大写
    "islower": 1,   # 全小写
    "istitle": 2,   # 首字母大写
    "isdigit": 3,   # 数字
    "other": 4      # 其他
}

def get_label_mapping(dataset_name):
    """获取指定数据集的标签映射"""
    if dataset_name in CHINESE_LABEL_MAPPING:
        return CHINESE_LABEL_MAPPING[dataset_name]
    else:
        # 默认使用英文标签映射
        return {
            "PER": 1,
            "LOC": 2,
            "ORG": 3,
            "MISC": 4,
            "O": 0
        }

def get_morph_mapping(dataset_name):
    """获取指定数据集的特征映射"""
    if dataset_name in ['msra', 'weibo', 'ontonotes4']:
        return CHINESE_MORPH_MAPPING
    else:
        return ENGLISH_MORPH_MAPPING

def get_label2idx_list(dataset_name):
    """获取标签到索引的列表"""
    label_mapping = get_label_mapping(dataset_name)
    return [(label, idx) for label, idx in label_mapping.items()]

def get_morph2idx_list(dataset_name):
    """获取特征到索引的列表"""
    morph_mapping = get_morph_mapping(dataset_name)
    return [(morph, idx) for morph, idx in morph_mapping.items()]

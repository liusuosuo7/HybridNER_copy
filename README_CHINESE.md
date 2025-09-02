# HybridNER 中文数据集适配说明

## 概述

本项目已经适配了中文NER数据集，支持MSRA、Weibo、OntoNotes4等中文数据集。

## 主要修改

### 1. BERT模型配置
- 默认使用中文BERT模型：`bert-base-chinese`
- 支持中文分词和字符级特征提取

### 2. 数据集配置
新增支持的中文数据集：
- `msra`: MSRA新闻数据集
- `weibo`: 微博数据集  
- `ontonotes4`: OntoNotes4数据集
- `cluener`: CLUENER细粒度中文NER数据集
- `cmeee`: CMEEE医学实体识别数据集

### 3. 特征提取
- 中文数据集使用字符级特征：数字、标点符号、其他
- 英文数据集使用词级特征：大小写、数字、其他

### 4. 标签映射
不同数据集的标签映射：

**MSRA数据集:**
```python
{
    "O": 0,      # 非实体
    "PER": 1,    # 人名
    "LOC": 2,    # 地名
    "ORG": 3,    # 机构名
    "MISC": 4    # 其他
}
```

**CLUENER数据集:**
```python
{
    "O": 0, "address": 1, "book": 2, "company": 3, "game": 4, 
    "government": 5, "movie": 6, "name": 7, "organization": 8, 
    "position": 9, "scene": 10
}
```

**CMEEE数据集（医学实体）:**
```python
{
    "O": 0, "dis": 1, "sym": 2, "pro": 3, "equ": 4, 
    "dru": 5, "ite": 6, "bod": 7
}
```

## 使用方法

### 1. 准备BERT模型
确保在 `/root/autodl-tmp/HybridNER/models/` 目录下有BERT模型：
```
/root/autodl-tmp/HybridNER/models/
├── bert-base-cased/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── bert-large-cased/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
```

### 2. 准备数据集
数据集应使用spanner格式的JSON文件，例如：
```json
[
  {
    "context": "张三在北京大学读书。",
    "span_posLabel": {
      "0;1": "PER",
      "3;5": "LOC",
      "6;9": "ORG"
    }
  }
]
```

### 3. 运行训练
```bash
# 使用MSRA数据集训练（BERT-Base）
python main.py --dataname msra --data_dir /root/autodl-tmp/HybridNER/dataset/msra --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-base-cased

# 使用CLUENER数据集训练
python main.py --dataname cluener --data_dir /root/autodl-tmp/HybridNER/dataset/cluener --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-base-cased

# 使用CMEEE数据集训练
python main.py --dataname cmeee --data_dir /root/autodl-tmp/HybridNER/dataset/cmeee --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-base-cased

# 或者使用便捷脚本
python run_dataset.py msra          # 运行MSRA数据集
python run_dataset.py cluener       # 运行CLUENER数据集
python run_dataset.py cmeee         # 运行CMEEE数据集
python run_dataset.py large_msra    # 使用BERT-Large运行MSRA
python run_dataset.py list          # 列出所有可用数据集
```

### 4. 数据目录结构
```
/root/autodl-tmp/HybridNER/dataset/
├── msra/
│   ├── spanner.train
│   ├── spanner.dev
│   └── spanner.test
├── cluener/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── cmeee/
│   ├── train.json
│   ├── dev.json
│   └── test.json
└── 其他数据集...
```

## 配置参数

### 中文数据集特有参数
- `--dataname`: 数据集名称，可选 `msra`, `cluener`, `cmeee`, `weibo`, `ontonotes4`
- `--bert_config_dir`: BERT模型目录，建议使用 `/root/autodl-tmp/HybridNER/models/bert-base-cased` 或 `bert-large-cased`
- `--data_sign`: 数据签名，中文数据集使用 `zh_msra`

### 通用参数
- `--batch_size`: 批次大小，建议32或64
- `--lr`: 学习率，建议3e-5
- `--max_spanLen`: 最大span长度，建议5
- `--bert_max_length`: 最大序列长度，建议128

## 注意事项

1. **中文分词**: 使用字符级分词，每个汉字作为一个token
2. **特征提取**: 中文使用字符特征（数字、标点、其他），英文使用词特征（大小写等）
3. **标签格式**: 使用BIO格式，但按字符标注
4. **模型选择**: 使用英文BERT模型（bert-base-cased/bert-large-cased）处理中文数据，支持中文字符处理

## 性能优化建议

1. **GPU内存**: 中文序列通常较长，建议调整batch_size
2. **学习率**: 中文BERT可能需要较小的学习率
3. **数据增强**: 可以考虑使用中文数据增强技术
4. **预训练**: 建议在中文NER数据上继续预训练

## 故障排除

### 常见问题
1. **内存不足**: 减小batch_size或max_length
2. **分词错误**: 确保使用中文BERT模型
3. **标签不匹配**: 检查数据集格式和标签映射

### 调试模式
```bash
# 使用BERT-Base调试
python main.py --dataname msra --batch_size 16 --bert_max_length 64 --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-base-cased

# 使用BERT-Large调试
python main.py --dataname msra --batch_size 8 --bert_max_length 64 --bert_config_dir /root/autodl-tmp/HybridNER/models/bert-large-cased
```

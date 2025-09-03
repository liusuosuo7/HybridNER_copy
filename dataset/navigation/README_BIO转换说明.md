# JSONL to BIO 格式转换说明

## 功能概述
将NER数据集从JSONL格式转换为标准的BIO格式，适用于训练传统的序列标注模型。

## 转换脚本介绍

### 1. jsonl_to_bio.py (完整版)
**特点**:
- ✅ 完整的字符偏移处理
- ✅ 文本清理和标准化
- ✅ 详细的转换分析
- ✅ 错误检测和报告
- ✅ 支持复杂文本处理

**适用场景**: 需要精确处理复杂文本的场景

### 2. simple_jsonl_to_bio.py (简化版)
**特点**:
- ⚡ 执行速度快
- 🎯 直接字符级处理
- 📝 避免复杂偏移计算
- 🔧 更稳定可靠

**适用场景**: 大多数标准NER任务，推荐使用

### 3. verify_bio_conversion.py (验证工具)
**特点**:
- 🔍 验证BIO标签正确性
- 📊 统计分析转换结果
- 📈 对比数据集分布
- 🎯 显示样本数据

## 使用方法

### 第一步：准备数据
确保您已经有分割好的JSONL文件：
- `entity_train.jsonl`
- `entity_val.jsonl`
- `entity_test.jsonl`

### 第二步：运行转换脚本

**推荐使用简化版（更稳定）**:
```bash
cd /path/to/navigation/
python simple_jsonl_to_bio.py
```

**或使用完整版（更详细）**:
```bash
python jsonl_to_bio.py
```

### 第三步：验证转换结果
```bash
python verify_bio_conversion.py
```

## 输出文件

转换后将生成以下BIO格式文件：
- **entity_train.bio** - 训练集BIO格式
- **entity_val.bio** - 验证集BIO格式
- **entity_test.bio** - 测试集BIO格式

## BIO格式说明

### 标注方案
- **B-LABEL**: 实体的开始 (Begin)
- **I-LABEL**: 实体的内部 (Inside)  
- **O**: 非实体 (Outside)

### 文件格式
```
中	B-对抗国家
国	I-对抗国家
在	O
南	B-对抗地点
海	I-对抗地点
部	O
署	O
北	B-北斗卫星导航系统
斗	I-北斗卫星导航系统
系	I-北斗卫星导航系统
统	I-北斗卫星导航系统

美	B-对抗国家
军	I-对抗国家
使	O
用	O
G	B-全球定位系统
P	I-全球定位系统
S	I-全球定位系统
```

**格式要点**:
- 每行格式: `字符\t标签`
- 句子间用空行分隔
- UTF-8编码
- 制表符(\t)分隔字符和标签

## 转换特点

### 中文字符级处理
- 按字符进行标注，适合中文NER任务
- 保留重要标点符号
- 过滤无意义的空白字符

### 实体类型保持
转换后保持原有的所有实体类型：
- 对抗国家、对抗时间、对抗地点
- 硬摧毁武器、压制式干扰技术
- 全球定位系统、北斗卫星导航系统
- 等等...

### 数据完整性
- 保持训练集、验证集、测试集的比例
- 维持实体分布的一致性
- 确保标注的正确性

## 验证功能

### BIO标签验证
- 检查标签格式正确性
- 验证B-I标签的连续性
- 发现标注错误并报告

### 统计分析
```
=== 分析 entity_train.bio ===
句子数量: 1137
token数量: 156789
✓ 未发现BIO标签错误

标签分布:
  O: 145234 (92.6%)
  B-对抗国家: 1876 (1.2%)
  I-对抗国家: 892 (0.6%)
  B-对抗时间: 1245 (0.8%)
  ...

实体类型分布:
  对抗国家: 1876 (23.4%)
  对抗时间: 1245 (15.5%)
  硬摧毁武器: 987 (12.3%)
  ...
```

### 数据集对比
```
数据集          句子数   Token数  实体数   实体密度
train           1137     156789   8012     5.11%
val             142      19634    1001     5.10%
test            142      19578    998      5.09%
```

## 使用场景

### 传统序列标注模型
- BiLSTM-CRF
- CRF
- Hidden Markov Model
- Conditional Random Field

### 深度学习模型
- BERT + CRF
- RoBERTa + CRF  
- LSTM + CRF
- Transformer + CRF

### 工具集成
- **spaCy**: 可直接使用BIO格式训练
- **Flair**: 支持BIO格式数据
- **AllenNLP**: 原生支持BIO格式
- **Transformers**: 可配合使用

## 常见问题

### Q: 转换后实体数量不对？
A: 检查原始JSONL文件的实体标注是否正确，使用验证脚本检查转换结果。

### Q: 出现标签错误？
A: 运行 `verify_bio_conversion.py` 查看具体错误，通常是实体偏移量问题。

### Q: 中文字符处理问题？
A: 使用简化版脚本 `simple_jsonl_to_bio.py`，更适合中文处理。

### Q: 如何自定义标签格式？
A: 修改脚本中的标签生成部分，支持自定义BIO变体（如BIOES）。

## 注意事项

1. **文本编码**: 确保所有文件使用UTF-8编码
2. **内存使用**: 大文件会全部加载到内存
3. **标点处理**: 保留有意义的标点，过滤多余空格
4. **实体边界**: 确保实体边界准确，避免字符偏移错误
5. **数据备份**: 转换前备份原始JSONL文件

## 文件结构

转换完成后的目录结构：
```
navigation/
├── entity.jsonl                    # 原始数据
├── entity_train.jsonl             # 训练集JSONL
├── entity_val.jsonl               # 验证集JSONL  
├── entity_test.jsonl              # 测试集JSONL
├── entity_train.bio               # 训练集BIO
├── entity_val.bio                 # 验证集BIO
├── entity_test.bio                # 测试集BIO
├── jsonl_to_bio.py                # 完整版转换脚本
├── simple_jsonl_to_bio.py         # 简化版转换脚本
├── verify_bio_conversion.py       # 验证脚本
└── README_BIO转换说明.md          # 本说明文档
```

## 后续处理

转换为BIO格式后，您可以：
1. **模型训练**: 使用各种NER框架训练模型
2. **格式转换**: 转换为其他格式（BIOES、IOBES等）
3. **数据增强**: 基于BIO格式进行数据增强
4. **模型评估**: 使用标准NER评估指标

---

**提示**: 建议先使用简化版脚本进行转换，如遇问题再尝试完整版。转换后务必运行验证脚本检查结果。

# HybridNER 组合器功能集成总结

## 概述

本次修改成功将SpanNER的组合器功能集成到HybridNER中，使HybridNER不仅具有基本的命名实体识别功能，还具备了组合多个NER模型结果的能力。

## 主要修改内容

### 1. 核心文件创建

#### `models/comb_voting.py`
- 完整实现了组合器核心功能
- 支持4种组合策略：
  - 多数投票 (Majority Voting)
  - 基于整体F1加权投票 (Weighted by Overall F1)
  - 基于类别F1加权投票 (Weighted by Category F1)
  - 基于Span预测分数投票 (Span Prediction Score)

#### `models/dataread.py`
- 实现了数据读取功能
- 支持序列标注格式和Span格式的模型结果文件
- 自动提取模型性能指标

#### `models/evaluate_metric.py`
- 实现了评估指标计算功能
- 支持chunk级别的评估
- 支持按类别评估

### 2. 框架集成

#### `src/framework.py`
- 添加了 `run_combiner()` 方法
- 支持在训练或推理后自动运行组合器
- 支持多种组合策略选择
- 集成了日志记录功能

#### `main.py`
- 增强了组合器功能的调用
- 支持独立运行组合器
- 支持训练/推理后自动运行组合器
- 改进了错误处理和用户提示

#### `args_config.py`
- 添加了组合器相关的命令行参数：
  - `--use_combiner`: 启用组合器功能
  - `--comb_model_results`: 模型结果文件列表
  - `--comb_strategy`: 组合策略选择
  - `--comb_result_dir`: 结果保存目录
  - `--auto_combine_after_train`: 训练后自动运行
  - `--auto_combine_after_inference`: 推理后自动运行

### 3. 示例和文档

#### `scripts/run_hybrid_combiner.sh`
- Linux/Mac环境下的示例脚本
- 演示了各种使用场景

#### `scripts/run_hybrid_combiner.py`
- 跨平台的Python示例脚本
- 提供了完整的测试用例

#### `README_Combiner.md`
- 详细的功能说明文档
- 包含使用方法、参数说明、故障排除等

#### `test_combiner.py`
- 功能测试脚本
- 验证组合器核心功能和框架集成

## 功能特点

### 1. 多种组合策略
- **多数投票**: 适用于模型性能相近的情况
- **整体F1加权**: 适用于模型性能差异较大的情况
- **类别F1加权**: 适用于不同模型在不同实体类型上表现差异较大的情况
- **Span预测分数**: 适用于需要平衡预测置信度和模型性能的情况

### 2. 灵活的使用方式
- 可以独立运行组合器功能
- 可以在训练模型后自动运行
- 可以在推理后自动运行
- 支持自定义组合策略

### 3. 自动化功能
- 自动从文件名中提取F1分数
- 自动创建结果保存目录
- 自动处理不同格式的模型结果文件

### 4. 用户友好
- 详细的日志输出
- 清晰的错误提示
- 完整的参数说明
- 丰富的示例脚本

## 使用方法

### 基本用法
```bash
python main.py \
    --dataname conll03 \
    --data_dir data/conll03 \
    --use_combiner True \
    --comb_model_results "model1_9201.txt" "model2_9246.txt" \
    --comb_strategy all
```

### 训练后自动运行
```bash
python main.py \
    --dataname conll03 \
    --data_dir data/conll03 \
    --state train \
    --use_combiner True \
    --comb_model_results "model1_9201.txt" "model2_9246.txt" \
    --auto_combine_after_train True
```

### 推理后自动运行
```bash
python main.py \
    --dataname conll03 \
    --data_dir data/conll03 \
    --state inference \
    --inference_model results/model.pkl \
    --use_combiner True \
    --comb_model_results "model1_9201.txt" "model2_9246.txt" \
    --auto_combine_after_inference True
```

## 文件格式支持

### 序列标注格式
```
word1    O    O
word2    B-PER    B-PER
word3    I-PER    I-PER
```

### Span格式
```
sentence_content    span1:: 0,1:: B-PER:: B-PER    span2:: 3,4:: B-ORG:: B-ORG
```

## 输出结果

组合器会在指定目录中生成以下文件：
- `VM_combine_XXXX.pkl`: 多数投票结果
- `VOF1_combine_XXXX.pkl`: 整体F1加权结果
- `VCF1_combine_XXXX.pkl`: 类别F1加权结果
- `SpanNER_combine_XXXX.pkl`: Span预测分数结果

其中XXXX表示F1分数（乘以10000后的整数）。

## 测试验证

运行测试脚本验证功能：
```bash
python test_combiner.py
```

## 兼容性

- 完全兼容原有的HybridNER功能
- 不影响现有的训练和推理流程
- 组合器功能为可选功能，默认关闭
- 支持多种操作系统（Windows/Linux/Mac）

## 总结

通过本次集成，HybridNER现在具备了：

1. **完整的组合器功能**: 可以组合多个NER模型的结果
2. **灵活的集成方式**: 支持独立运行和自动运行
3. **多种组合策略**: 适应不同的应用场景
4. **用户友好的接口**: 简单易用的命令行参数
5. **完善的文档**: 详细的使用说明和示例

这使得HybridNER不仅是一个强大的命名实体识别框架，还是一个具有组合器功能的综合NER解决方案。

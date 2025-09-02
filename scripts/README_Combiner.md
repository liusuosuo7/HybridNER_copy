# HybridNER 组合器功能说明

## 概述

HybridNER现在支持组合器功能，可以将多个命名实体识别模型的结果进行组合，以提高整体性能。这个功能基于SpanNER的组合器实现，并集成到了HybridNER框架中。

## 功能特点

1. **多种组合策略**：支持4种不同的组合策略
2. **灵活的使用方式**：可以独立运行，也可以在训练/推理后自动运行
3. **自动F1分数提取**：从文件名中自动提取模型性能分数
4. **结果保存**：组合结果自动保存到指定目录

## 组合策略

### 1. 多数投票 (Majority Voting)
- 对每个实体span，统计所有模型的预测结果
- 选择被最多模型预测的标签作为最终结果
- 适用于模型性能相近的情况

### 2. 基于整体F1加权投票 (Weighted by Overall F1)
- 根据每个模型的整体F1分数进行加权
- 性能更好的模型具有更大的投票权重
- 适用于模型性能差异较大的情况

### 3. 基于类别F1加权投票 (Weighted by Category F1)
- 根据每个模型在不同实体类别上的F1分数进行加权
- 针对不同实体类型选择最擅长的模型
- 适用于不同模型在不同实体类型上表现差异较大的情况

### 4. 基于Span预测分数投票 (Span Prediction Score)
- 结合模型的预测置信度和F1分数
- 同时考虑预测的确定性和模型的历史性能
- 适用于需要平衡预测置信度和模型性能的情况

## 使用方法

### 1. 基本用法

```bash
python main.py \
    --dataname conll03 \
    --data_dir data/conll03 \
    --use_combiner True \
    --comb_model_results \
        "model1_9201.txt" \
        "model2_9246.txt" \
        "model3_9302.txt" \
    --comb_strategy all
```

### 2. 训练后自动运行组合器

```bash
python main.py \
    --dataname conll03 \
    --data_dir data/conll03 \
    --state train \
    --use_combiner True \
    --comb_model_results "model1_9201.txt" "model2_9246.txt" \
    --comb_strategy majority \
    --auto_combine_after_train True
```

### 3. 推理后自动运行组合器

```bash
python main.py \
    --dataname conll03 \
    --data_dir data/conll03 \
    --state inference \
    --inference_model results/model.pkl \
    --use_combiner True \
    --comb_model_results "model1_9201.txt" "model2_9246.txt" \
    --comb_strategy weighted_f1 \
    --auto_combine_after_inference True
```

## 参数说明

### 组合器相关参数

- `--use_combiner`: 是否启用组合器功能 (True/False)
- `--comb_model_results`: 待组合的模型结果文件路径列表
- `--comb_strategy`: 组合策略选择
  - `majority`: 多数投票
  - `weighted_f1`: 整体F1加权
  - `weighted_cat`: 类别F1加权
  - `span_score`: Span预测分数
  - `all`: 全部策略
- `--comb_result_dir`: 组合结果保存目录 (默认: comb_result)
- `--auto_combine_after_train`: 训练后是否自动运行组合器 (默认: True)
- `--auto_combine_after_inference`: 推理后是否自动运行组合器 (默认: True)

## 文件格式要求

### 模型结果文件格式

组合器支持两种格式的模型结果文件：

1. **序列标注格式** (适用于BiLSTM-CRF等模型)
```
word1    O    O
word2    B-PER    B-PER
word3    I-PER    I-PER
word4    O    O

word5    B-ORG    B-ORG
word6    I-ORG    I-ORG
```

2. **Span格式** (适用于SpanNER等模型)
```
sentence_content    span1:: 0,1:: B-PER:: B-PER    span2:: 3,4:: B-ORG:: B-ORG
```

### 文件名格式建议

建议使用以下格式命名模型结果文件：
```
model_name_f1score.txt
```

例如：
- `bert_lstm_crf_9201.txt` (F1=0.9201)
- `flair_glove_9246.txt` (F1=0.9246)
- `span_ner_9302.txt` (F1=0.9302)

系统会自动从文件名中提取F1分数用于加权计算。

## 输出结果

组合器会在指定的结果目录中生成以下文件：

- `VM_combine_XXXX.pkl`: 多数投票结果
- `VOF1_combine_XXXX.pkl`: 整体F1加权结果
- `VCF1_combine_XXXX.pkl`: 类别F1加权结果
- `SpanNER_combine_XXXX.pkl`: Span预测分数结果

其中XXXX表示F1分数（乘以10000后的整数）。

## 示例脚本

项目提供了两个示例脚本：

1. `scripts/run_hybrid_combiner.sh` (Linux/Mac)
2. `scripts/run_hybrid_combiner.py` (Windows/Linux/Mac)

运行示例：
```bash
# Linux/Mac
bash scripts/run_hybrid_combiner.sh

# Windows/Linux/Mac
python scripts/run_hybrid_combiner.py
```

## 注意事项

1. 确保所有模型结果文件都存在于指定的数据目录中
2. 所有模型结果文件应该基于相同的测试集
3. 文件名中的F1分数应该是0-1之间的小数（如0.9201）
4. 组合器会自动创建结果保存目录
5. 如果文件名中没有F1分数，系统会使用默认值0.8

## 性能优化建议

1. **模型选择**：选择性能相近但结构不同的模型进行组合
2. **策略选择**：
   - 模型性能相近：使用多数投票
   - 模型性能差异大：使用加权策略
   - 有预测置信度：使用Span预测分数策略
3. **文件管理**：使用有意义的文件名，包含模型信息和性能指标

## 故障排除

### 常见问题

1. **文件不存在错误**
   - 检查模型结果文件路径是否正确
   - 确保文件存在于指定的数据目录中

2. **F1分数提取失败**
   - 检查文件名格式是否符合要求
   - 系统会使用默认值0.8

3. **组合结果为空**
   - 检查模型结果文件格式是否正确
   - 确保文件包含有效的预测结果

### 调试模式

可以通过查看日志输出来调试组合器功能：
```bash
python main.py --use_combiner True --comb_model_results file1.txt file2.txt 2>&1 | tee combiner.log
```

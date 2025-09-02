# 4090显卡优化训练解决方案

## 问题分析

根据你的训练日志，主要问题是：
1. **严重过拟合**：训练时F1值46%，验证时F1值仅2-3%
2. **标签映射问题**：可能存在实体类型不匹配
3. **训练参数不当**：学习率、正则化等参数需要优化

## 解决方案概述

### 方案1：快速修复 + 优化训练（推荐）
- 修复标签映射问题
- 使用4090优化的训练参数
- 解决过拟合问题

### 方案2：数据增强 + 反过拟合训练
- 平衡实体分布
- 应用数据增强
- 使用强正则化训练

## 执行步骤

### 步骤1：修复标签映射问题
```bash
# 运行标签修复脚本
python fix_navigation_labels.py
```

这个脚本会：
- 分析你的navigation数据集
- 自动发现所有实体类型
- 更新`dataloaders/spanner_dataset.py`中的标签映射
- 创建调试脚本

### 步骤2：快速验证修复效果
```bash
# 运行快速验证脚本
bash quick_validation.sh
```

这个脚本会：
- 修复标签映射
- 测试数据集加载
- 运行5轮快速训练测试
- 验证F1值是否>0

### 步骤3：运行4090优化训练
```bash
# 运行4090优化训练
bash run_4090_optimized_training.sh
```

### 步骤4：如果仍有问题，使用反过拟合训练
```bash
# 准备增强数据
python anti_overfitting_training.py

# 运行反过拟合训练
bash run_anti_overfitting_training.sh
```

## 核心优化参数

### 4090优化参数
- **batch_size**: 8 (避免内存溢出)
- **lr**: 5e-6 (降低学习率，减少过拟合)
- **max_spanLen**: 6 (适合中文实体)
- **bert_max_length**: 256 (平衡性能和内存)
- **iteration**: 150 (增加训练轮数)
- **weight_decay**: 0.1 (增强正则化)
- **dropout**: 0.3/0.2 (增强正则化)
- **early_stop**: 10 (防止过拟合)

### 反过拟合参数
- **batch_size**: 6 (更小的批次)
- **lr**: 3e-6 (更低的学习率)
- **weight_decay**: 0.15 (更强的正则化)
- **dropout**: 0.4/0.3 (更强的dropout)
- **early_stop**: 15 (更严格的早停)

## 预期效果

### 修复后
- F1值从2-3%提升到30-50%
- 解决标签映射问题
- 模型能正确识别实体

### 优化后
- F1值从30-50%提升到60%+
- 解决过拟合问题
- 模型泛化能力增强

## 故障排除

### 如果F1值仍为0
1. 检查标签映射是否正确更新
2. 运行调试脚本：`python debug_navigation_labels.py`
3. 检查数据集格式是否正确

### 如果内存不足
1. 减少`batch_size`到4
2. 减少`bert_max_length`到128
3. 使用`nvidia-smi --gpu-reset`清理GPU

### 如果训练速度慢
1. 检查GPU使用率：`nvidia-smi`
2. 确保使用4090显卡：`export CUDA_VISIBLE_DEVICES=0`
3. 优化系统设置：`nvidia-smi -pm 1`

## 文件说明

- `fix_navigation_labels.py`: 修复标签映射问题
- `quick_validation.sh`: 快速验证修复效果
- `run_4090_optimized_training.sh`: 4090优化训练脚本
- `anti_overfitting_training.py`: 反过拟合数据准备
- `run_anti_overfitting_training.sh`: 反过拟合训练脚本

## 建议执行顺序

1. **立即执行**：`python fix_navigation_labels.py`
2. **验证修复**：`bash quick_validation.sh`
3. **如果成功**：`bash run_4090_optimized_training.sh`
4. **如果仍有问题**：`python anti_overfitting_training.py` + `bash run_anti_overfitting_training.sh`

## 注意事项

1. **备份重要文件**：脚本会自动备份原文件
2. **监控训练过程**：关注验证集F1值变化
3. **及时停止**：如果验证集F1值开始下降，及时停止训练
4. **保存最佳模型**：训练完成后保存验证集F1值最高的模型

## 预期时间

- 标签修复：1-2分钟
- 快速验证：5-10分钟
- 完整训练：2-4小时（取决于数据量）
- 反过拟合训练：3-6小时

按照这个方案，你的F1值应该能从2-3%提升到60%以上！


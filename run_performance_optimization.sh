#!/bin/bash

# Navigation数据集性能优化
echo "=== Navigation数据集性能优化 ==="
echo "目标：将F1值从30%提升到70%以上"
echo "适配环境：使用本地bert-large-cased模型"

# 检查数据集目录是否存在
DATA_DIR="/root/autodl-tmp/HybridNER/dataset/navigation"
if [ ! -d "$DATA_DIR" ]; then
    echo "错误：数据集目录不存在: $DATA_DIR"
    exit 1
fi

# 检查训练文件是否存在
TRAIN_FILE="$DATA_DIR/navigation_train_span.json"
if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误：训练文件不存在: $TRAIN_FILE"
    exit 1
fi

# 检查模型目录是否存在
MODEL_DIR="/root/autodl-tmp/HybridNER/models/bert-large-cased"
if [ ! -d "$MODEL_DIR" ]; then
    echo "错误：bert-large-cased模型目录不存在: $MODEL_DIR"
    echo "请确保模型已正确下载到指定路径"
    echo "检查本地模型目录: /root/autodl-tmp/HybridNER/models/"
    ls -la /root/autodl-tmp/HybridNER/models/ 2>/dev/null || echo "模型目录不存在"
    exit 1
fi

# 检查模型文件
MODEL_FILES=("config.json" "pytorch_model.bin" "vocab.txt")
for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "$MODEL_DIR/$file" ]; then
        echo "警告：模型文件 $file 不存在: $MODEL_DIR/$file"
    fi
done

echo "数据集目录: $DATA_DIR"
echo "训练文件: $TRAIN_FILE"
echo "模型目录: $MODEL_DIR"

# 创建优化脚本
echo "开始创建性能优化脚本..."
python optimize_navigation_performance.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=== 性能优化脚本创建完成 ==="
    echo ""
    echo "已创建以下优化脚本："
    echo "1. data_augmentation.py - 数据增强"
    echo "2. run_basic_optimized.sh - 基础优化训练"
    echo "3. run_large_model.sh - 大模型训练"
    echo "4. run_multitask.sh - 多任务学习"
    echo "5. run_adversarial.sh - 对抗训练"
    echo "6. run_ensemble.sh - 集成学习"
    echo "7. hyperparameter_tuning.py - 超参数调优"
    echo "8. evaluate_models.py - 模型评估"
    echo "9. optimization_plan.md - 优化计划"
    echo ""
    echo "建议执行顺序："
    echo ""
    echo "第一阶段：基础优化 (目标: 40-50%)"
    echo "1. python data_augmentation.py"
    echo "2. bash run_basic_optimized.sh"
    echo "3. bash run_large_model.sh"
    echo ""
    echo "第二阶段：高级优化 (目标: 50-60%)"
    echo "4. bash run_multitask.sh"
    echo "5. bash run_adversarial.sh"
    echo "6. python hyperparameter_tuning.py"
    echo ""
    echo "第三阶段：精细优化 (目标: 60-70%)"
    echo "7. bash run_ensemble.sh"
    echo "8. python evaluate_models.py"
    echo ""
    echo "注意事项："
    echo "- 所有模型都使用本地路径，避免网络下载"
    echo "- 根据GPU内存调整batch_size参数"
    echo "- 监控训练过程中的内存使用情况"
    echo "- 定期保存检查点以防训练中断"
    echo ""
    echo "现在可以开始执行优化策略！"
else
    echo "创建优化脚本时出现错误，请检查日志"
    exit 1
fi

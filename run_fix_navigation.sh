#!/bin/bash

# 修复Navigation数据集问题
echo "=== 修复Navigation数据集问题 ==="
echo "目标：解决指标为0的问题，使用完整实体类型映射"

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

echo "数据集目录: $DATA_DIR"
echo "训练文件: $TRAIN_FILE"

# 运行调试脚本
echo "开始调试数据集..."
python debug_navigation_dataset.py

echo ""

# 运行修复脚本
echo "开始修复数据集..."
python fix_navigation_dataset.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=== 修复完成 ==="
    echo "请检查以下内容："
    echo "1. 备份目录是否创建成功"
    echo "2. 模型配置文件是否已更新"
    echo "3. 新的训练脚本是否已创建"
    echo ""
    echo "现在可以使用修复后的配置进行训练："
    echo "bash run_navigation_fixed.sh"
else
    echo "修复过程中出现错误，请检查日志"
    exit 1
fi

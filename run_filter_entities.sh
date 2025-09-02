#!/bin/bash

# 保留数量排名前十的实体类型
echo "=== 开始处理Navigation数据集 ==="
echo "目标：保留数量排名前十的实体类型"

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

# 运行Python脚本
echo "开始执行过滤脚本..."
python filter_top10_entities.py

if [ $? -eq 0 ]; then
    echo "=== 处理完成 ==="
    echo "请检查以下内容："
    echo "1. 备份目录是否创建成功"
    echo "2. 数据集文件是否已更新"
    echo "3. 模型配置文件是否已更新"
    echo ""
    echo "现在可以使用优化后的数据集进行训练："
    echo "bash run_navigation_optimized.sh"
else
    echo "处理过程中出现错误，请检查日志"
    exit 1
fi

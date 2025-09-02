#!/bin/bash

# 快速清理磁盘空间
echo "=== 快速清理磁盘空间 ==="

echo "当前磁盘使用情况:"
df -h .

echo ""
echo "开始快速清理..."

# 清理输出目录
echo "1. 清理输出目录..."
rm -rf /root/autodl-tmp/HybridNER/output/* 2>/dev/null || true
rm -rf ./log/* 2>/dev/null || true
rm -rf ./results/* 2>/dev/null || true

# 清理临时文件
echo "2. 清理临时文件..."
rm -rf /tmp/* 2>/dev/null || true
rm -rf /var/tmp/* 2>/dev/null || true

# 清理缓存
echo "3. 清理缓存..."
rm -rf ~/.cache/* 2>/dev/null || true
rm -rf /root/.cache/* 2>/dev/null || true

# 清理Python缓存
echo "4. 清理Python缓存..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 清理模型缓存
echo "5. 清理模型缓存..."
rm -rf ~/.cache/huggingface 2>/dev/null || true
rm -rf ~/.cache/torch 2>/dev/null || true

# 清理垃圾箱
echo "6. 清理垃圾箱..."
rm -rf ~/.local/share/Trash/* 2>/dev/null || true

echo ""
echo "清理后的磁盘使用情况:"
df -h .

echo ""
echo "快速清理完成！"

# 检查可用空间
AVAILABLE=$(df . | awk 'NR==2 {print $4}')
AVAILABLE_MB=$((AVAILABLE / 1024))
AVAILABLE_GB=$((AVAILABLE_MB / 1024))

echo ""
echo "可用空间: ${AVAILABLE_GB} GB (${AVAILABLE_MB} MB)"

if [ $AVAILABLE_MB -gt 2048 ]; then
    echo "✓ 磁盘空间充足，可以继续训练"
else
    echo "⚠️  磁盘空间仍然不足，建议进一步清理"
fi

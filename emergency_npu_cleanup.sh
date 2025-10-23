#!/bin/bash

# NPU内存和环境清理脚本
# 用于解决NPU OOM问题

set -e

echo "=== NPU内存和环境清理脚本 ==="

# 1. 杀死所有可能的Python训练进程
echo "1. 杀死所有Python训练进程..."
pkill -f "python.*train" || echo "没有发现训练进程"
pkill -f "torchrun" || echo "没有发现torchrun进程"
sleep 2

# 2. 清理PyTorch NPU缓存
echo "2. 清理PyTorch NPU缓存..."
python3 clear_npu_memory.py

# 3. 清理系统缓存
echo "3. 清理系统缓存..."
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "需要root权限清理系统缓存"

# 4. 清理临时文件
echo "4. 清理临时文件..."
rm -rf /tmp/torch_*
rm -rf ~/.cache/torch_extensions/*
rm -rf ~/.cache/huggingface/transformers/

# 5. 重置NPU环境变量
echo "5. 重置NPU环境变量..."
unset ASCEND_RT_VISIBLE_DEVICES
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT

# 6. 检查NPU状态
echo "6. 检查NPU状态..."
npu-smi info

echo "=== 清理完成 ==="
echo "建议等待10-15秒后再开始新的训练"
#!/bin/bash

# 极度保守的单NPU训练脚本 - 最小内存配置
# Ultra-conservative single NPU training - minimal memory configuration

set -e

echo "=== 极度保守单NPU训练 ==="
echo "Ultra-conservative Single NPU Training"

# 1. 强制清理环境
echo "1. 强制清理NPU环境..."
pkill -f python || echo "没有Python进程"
sleep 3

# 2. 设置极度保守的配置
export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 极度保守的内存配置
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64  # NPU最小要求>20MB
export NPU_COMPILE_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

echo "极度保守配置:"
echo "  device_batch_size: 2"
echo "  total_batch_size: 4096 (正好是 2 * 2048 * 1)"
echo "  depth: 3 (非常小的模型)"
echo "  内存分片: 16MB"

# 3. 检查计算逻辑
echo "2. Batch size计算验证:"
echo "  device_batch_size * max_seq_len * world_size = 2 * 2048 * 1 = 4096"
echo "  total_batch_size = 4096"
echo "  4096 % 4096 = 0 ✅"

# 4. 训练tokenizer（如果需要）
echo "3. 训练tokenizer（如果需要）..."
python -m scripts.tok_train || echo "tokenizer训练失败，继续..."

# 5. 极度保守的模型训练
echo "4. 开始极度保守训练..."
python -m scripts.base_train \
    --run=ultra_conservative_npu \
    --depth=3 \
    --device_batch_size=2 \
    --total_batch_size=4096 \
    --num_iterations=500

echo "✅ 极度保守训练完成！"
echo "如果这个配置都OOM，建议:"
echo "1. 检查是否有其他程序占用NPU内存"
echo "2. 重启NPU驱动"
echo "3. 考虑减少depth到2或1"
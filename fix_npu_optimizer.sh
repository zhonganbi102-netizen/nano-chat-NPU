#!/bin/bash

# NPU优化器修复脚本
# 解决Muon优化器在NPU上卡住的问题

set -e

echo "=== NPU优化器兼容性修复 ==="

# 1. 清理环境
echo "1. 清理环境..."
./emergency_npu_cleanup.sh || echo "清理脚本执行失败，继续..."
sleep 5

# 2. 设置环境变量
echo "2. 设置环境变量..."
export ASCEND_RT_VISIBLE_DEVICES=0  # 只使用一个NPU
export WORLD_SIZE=1
export TORCH_COMPILE_DISABLE=1  # 禁用编译

# 3. 应用优化器兼容性设置
echo "3. 设置NPU优化器兼容性..."
export NPU_OPTIMIZER_COMPATIBLE=1
export NPU_USE_ADAM_ONLY=1  # 强制使用Adam替代Muon
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32  # 限制内存分片

# 4. 开始优化器兼容性训练
echo "4. 开始兼容性训练..."
python -m scripts.base_train \
    --run=npu_optimizer_fixed \
    --depth=6 \
    --device_batch_size=4 \
    --total_batch_size=16384 \
    --num_iterations=500 \
    --embedding_lr=0.01 \
    --unembedding_lr=0.001 \
    --matrix_lr=0.005 \
    --grad_clip=0.5

echo "训练完成！"

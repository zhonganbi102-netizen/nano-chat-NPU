#!/bin/bash

echo "🚀 快速启动4NPU训练..."

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 检查NPU状态
echo "检查NPU状态..."
python3 -c "
import torch_npu
print(f'可用NPU数量: {torch_npu.npu.device_count()}')
for i in range(min(4, torch_npu.npu.device_count())):
    print(f'NPU {i}: {torch_npu.npu.get_device_name(i)}')
"

# 清理之前的进程
echo "清理之前的训练进程..."
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 2

# 启动4NPU训练
echo "启动4NPU分布式训练..."
echo "配置: 4个NPU, depth=12, device_batch_size=8, total_batch_size=262144"
echo ""

# 生成运行名称
RUN_NAME="4npu_quick_$(date +%Y%m%d_%H%M%S)"
echo "运行名称: $RUN_NAME"

torchrun --standalone --nproc_per_node=4 scripts/base_train.py \
    --depth 12 \
    --device_batch_size 8 \
    --total_batch_size 262144 \
    --run "$RUN_NAME"

echo ""
echo "训练完成: $(date)"
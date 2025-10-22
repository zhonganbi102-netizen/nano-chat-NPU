#!/bin/bash

echo "=== 🚀 4NPU分布式训练启动脚本 ==="

# 清理之前的训练进程
echo "1. 清理环境..."
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true

# 等待进程清理完成
sleep 2

# 设置NPU环境变量
echo "2. 配置NPU环境..."
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3  # 使用前4个NPU
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 设置HCCL环境
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

echo "NPU设备: $ASCEND_RT_VISIBLE_DEVICES"
echo "世界大小: $WORLD_SIZE"
echo "主节点: $MASTER_ADDR:$MASTER_PORT"

# 检查NPU状态
echo "3. 检查NPU设备..."
python3 -c "
import torch_npu
device_count = torch_npu.npu.device_count()
print(f'可用NPU设备数: {device_count}')
for i in range(min(4, device_count)):
    print(f'NPU {i}: {torch_npu.npu.get_device_name(i)}')
"

# 验证数据路径
echo "4. 验证数据路径..."
python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')
from nanochat.dataset import list_parquet_files
files = list_parquet_files()
print(f'数据文件数: {len(files)}')
if len(files) == 0:
    print('❌ 数据文件未找到!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ 数据验证失败，请先修复数据路径"
    exit 1
fi

echo "5. 启动4NPU分布式训练..."

# 训练参数
DEPTH=12
DEVICE_BATCH_SIZE=8  # 每个NPU的batch size
TOTAL_BATCH_SIZE=262144  # 总batch size

echo "训练配置:"
echo "  模型深度: $DEPTH"
echo "  单设备batch size: $DEVICE_BATCH_SIZE"
echo "  总batch size: $TOTAL_BATCH_SIZE"
echo "  梯度累积步数: $((TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * 2048 * 4)))"

# 启动分布式训练
echo ""
echo "🚀 启动分布式训练..."

torchrun \
    --standalone \
    --nproc_per_node=4 \
    -- \
    scripts/base_train.py \
    --depth $DEPTH \
    --device_batch_size $DEVICE_BATCH_SIZE \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --num_iterations 2000 \
    --run "4npu_training_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "训练结束时间: $(date)"
#!/bin/bash

# 超轻量级单NPU训练脚本 - 专门解决OOM问题
# Ultra-lightweight single NPU training script - specifically for OOM issues

set -e

echo "=== 超轻量级单NPU训练 ==="
echo "Ultra-lightweight Single NPU Training"

# 1. 强制清理环境
echo "1. 强制清理NPU环境..."
pkill -f python || echo "没有Python进程"
sleep 3

# 清理系统缓存
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "无法清理系统缓存（需要root权限）"

# 2. 设置最小内存配置
export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 关键内存优化设置
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32
export NPU_COMPILE_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

echo "超轻量级配置:"
echo "  使用NPU: 0"
echo "  内存分片: 32MB"
echo "  编译优化: 禁用"

# 3. 检查NPU状态
echo "2. 检查NPU状态..."
npu-smi info || echo "npu-smi命令不可用"

python3 -c "
import torch
import torch_npu
import gc
if torch_npu.npu.is_available():
    # 强制清理
    torch_npu.npu.empty_cache()
    gc.collect()
    
    # 检查内存
    device = torch_npu.npu.current_device()
    allocated = torch_npu.npu.memory_allocated(device) / 1024**3
    reserved = torch_npu.npu.memory_reserved(device) / 1024**3
    print(f'NPU {device}: 已分配 {allocated:.2f} GiB, 保留 {reserved:.2f} GiB')
"

# 4. 训练tokenizer（最小配置）
echo "3. 训练tokenizer（如果需要）..."
python -m scripts.tok_train || echo "tokenizer训练失败，继续..."

# 5. 超小模型训练
echo "4. 开始超轻量级训练..."
python -m scripts.base_train \
    --run=ultra_light_npu \
    --depth=4 \
    --device_batch_size=4 \
    --total_batch_size=4096 \
    --num_iterations=1000

echo "✅ 超轻量级训练完成！"
echo "如果还是OOM，请检查:"
echo "1. 是否有其他程序占用NPU内存"
echo "2. 考虑进一步减少depth到2或3"
echo "3. 减少device_batch_size到2或1"
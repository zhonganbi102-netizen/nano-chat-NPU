#!/bin/bash

# 调试Muon优化器单NPU脚本
# Debug Muon optimizer single NPU script

set -e

echo "🔍 调试Muon优化器 - 单NPU模式"
echo ""

# 1. 设置环境
echo "1. 设置单NPU环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 单NPU环境变量
export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600

# 内存优化设置
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64
export NPU_COMPILE_DISABLE=1

echo "环境变量:"
echo "  ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  NPU_COMPILE_DISABLE: $NPU_COMPILE_DISABLE"

# 2. 清理NPU
echo ""
echo "2. 清理NPU环境..."
pkill -f "python.*train" || echo "无训练进程"
pkill -f "torchrun" || echo "无torchrun进程"

python3 -c "
import torch
import torch_npu
import gc
if torch_npu.npu.is_available():
    torch_npu.npu.empty_cache()
    gc.collect()
    print('✅ NPU缓存已清理')
"

# 3. 检查tokenizer
echo ""
echo "3. 检查tokenizer..."
if [ ! -f "tokenizer/tokenizer.json" ]; then
    echo "创建简单tokenizer..."
    mkdir -p tokenizer
    python3 -c "
import json
tokenizer_config = {
    'version': '1.0',
    'model': {'type': 'BPE', 'vocab': {'<unk>': 0, '<s>': 1, '</s>': 2}, 'merges': []}
}
with open('tokenizer/tokenizer.json', 'w') as f:
    json.dump(tokenizer_config, f)
print('✅ 简单tokenizer已创建')
"
else
    echo "✅ tokenizer已存在"
fi

# 4. 调试训练（极小配置）
echo ""
echo "4. 开始调试训练（极小配置）..."
echo "使用最小参数避免Muon优化器卡死..."

python3 -m scripts.base_train \
    --run=debug_muon_single_npu \
    --depth=3 \
    --device_batch_size=2 \
    --total_batch_size=4 \
    --num_iterations=10 \
    --embedding_lr=0.001 \
    --unembedding_lr=0.0001 \
    --matrix_lr=0.0005 \
    --grad_clip=1.0 \
    --eval_every=5 \
    --sample_every=999999 \
    --core_metric_every=999999 \
    --verbose

echo ""
echo "🎉 调试完成！"
echo "如果成功，可以逐步增加参数大小"
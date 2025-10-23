#!/bin/bash

# 单NPU训练脚本 - 专门针对内存限制优化
# Single NPU training script - optimized for memory constraints

set -e

echo "=== 单NPU训练脚本 ==="
echo "Single NPU Training Script"

# 1. 环境检查
echo "1. 检查NPU环境..."
python3 -c "
import torch
import torch_npu
assert torch_npu.npu.is_available(), 'NPU不可用'
print(f'NPU设备数量: {torch_npu.npu.device_count()}')
print(f'当前设备: {torch_npu.npu.current_device()}')
print('✅ NPU环境正常')
"

# 2. 清理环境
echo "2. 清理NPU环境..."
# 杀死可能的残留进程
pkill -f "python.*train" || echo "没有发现训练进程"
pkill -f "torchrun" || echo "没有发现torchrun进程"

# 清理NPU缓存
python3 -c "
import torch
import torch_npu
import gc
if torch_npu.npu.is_available():
    torch_npu.npu.empty_cache()
    gc.collect()
    print('✅ NPU缓存已清理')
"

# 3. 设置单NPU环境变量
export ASCEND_RT_VISIBLE_DEVICES=0  # 只使用第一个NPU
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 设置NPU内存优化
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export NPU_COMPILE_DISABLE=1  # 禁用编译优化以减少内存

echo "单NPU配置:"
echo "  ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  PYTORCH_NPU_ALLOC_CONF: $PYTORCH_NPU_ALLOC_CONF"

# 4. 训练tokenizer（如果需要）
if [ ! -f "~/.cache/nanochat/tok.model" ]; then
    echo "3. 训练tokenizer..."
    python -m scripts.tok_train
else
    echo "3. tokenizer已存在，跳过训练"
fi

# 5. 单NPU base model训练
echo "4. 开始单NPU base model训练..."
python -m scripts.base_train \
    --run=single_npu_base \
    --depth=8 \
    --device_batch_size=16 \
    --total_batch_size=32768 \
    --num_iterations=3000

echo "5. 单NPU base训练完成！"

# 6. 可选的SFT训练
read -p "是否继续SFT训练? (y/N): " continue_sft
if [[ $continue_sft =~ ^[Yy]$ ]]; then
    echo "6. 开始单NPU SFT训练..."
    python -m scripts.chat_sft \
        --run=single_npu_sft \
        --device_batch_size=4 \
        --target_examples_per_step=16 \
        --num_iterations=2000
    
    echo "✅ SFT训练完成！"
else
    echo "跳过SFT训练"
fi

echo "=== 单NPU训练完成! ==="
echo "模型保存在: ~/.cache/nanochat/"
echo "可以使用以下命令测试:"
echo "  python -m scripts.chat_cli"
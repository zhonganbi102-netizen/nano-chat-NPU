#!/bin/bash

# 华为昇腾NPU版本的nanochat训练脚本
# Huawei Ascend NPU version of nanochat training pipeline

set -e  # 出错时退出

echo "=== NanoChat NPU Training Pipeline ==="
echo "华为昇腾NPU训练管道开始..."

# 1. 环境检查
echo "1. 检查NPU环境..."
python3 -c "
import torch
import torch_npu
assert torch_npu.npu.is_available(), 'NPU不可用'
print(f'NPU设备数量: {torch_npu.npu.device_count()}')
print(f'当前设备: {torch_npu.npu.current_device()}')
"

# 2. 设置NPU相关环境变量 (适配5个NPU)
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4}
export WORLD_SIZE=${WORLD_SIZE:-5}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

echo "NPU配置:"
echo "  ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
echo "  WORLD_SIZE: $WORLD_SIZE"

# 3. 训练步骤
echo "2. 开始训练tokenizer..."
python -m scripts.tok_train

echo "3. 训练base model (depth=12)..."
torchrun --standalone --nproc_per_node=$WORLD_SIZE -m scripts.base_train -- \
    --run=npu_base_d12 \
    --depth=12 \
    --device_batch_size=16 \
    --total_batch_size=262144

echo "4. 运行midtraining..."
torchrun --standalone --nproc_per_node=$WORLD_SIZE -m scripts.mid_train -- \
    --run=npu_mid_d12 \
    --device_batch_size=16 \
    --total_batch_size=262144

echo "5. Chat SFT..."
torchrun --standalone --nproc_per_node=$WORLD_SIZE -m scripts.chat_sft -- \
    --run=npu_sft_d12 \
    --device_batch_size=8 \
    --target_examples_per_step=32

echo "6. Chat RL (可选)..."
read -p "是否运行RL训练? (y/N): " run_rl
if [[ $run_rl =~ ^[Yy]$ ]]; then
    torchrun --standalone --nproc_per_node=$WORLD_SIZE -m scripts.chat_rl -- \
        --run=npu_rl_d12 \
        --device_batch_size=4 \
        --examples_per_step=16
else
    echo "跳过RL训练"
fi

echo "7. 启动web服务..."
read -p "是否启动web服务? (y/N): " start_web
if [[ $start_web =~ ^[Yy]$ ]]; then
    python -m scripts.chat_web
else
    echo "跳过web服务启动"
fi

echo "=== NPU训练管道完成! ==="
echo "模型已保存在 ~/.cache/nanochat/ 目录中"
echo "可以使用以下命令测试模型:"
echo "  python -m scripts.chat_cli"

#!/bin/bash

# 简化版NPU训练脚本 - 跳过tokenizer训练，直接使用预训练模型
# Simplified NPU training script - skip tokenizer training

set -e

echo "=== 简化版 NanoChat NPU Training Pipeline ==="
echo "华为昇腾NPU训练管道开始 (简化版)..."

# 1. 环境检查
echo "1. 检查NPU环境..."
python3 -c "
import torch
import torch_npu
assert torch_npu.npu.is_available(), 'NPU不可用'
npu_count = torch_npu.npu.device_count()
print(f'NPU设备数量: {npu_count}')
print(f'当前设备: {torch_npu.npu.current_device()}')
# 设置适合的WORLD_SIZE
if npu_count >= 8:
    world_size = 8
elif npu_count >= 4:
    world_size = 4
else:
    world_size = min(npu_count, 5)
print(f'建议WORLD_SIZE: {world_size}')
"

# 2. 自动设置NPU配置
NPU_COUNT=$(python3 -c "import torch_npu; print(torch_npu.npu.device_count())")
if [ "$NPU_COUNT" -ge 8 ]; then
    export WORLD_SIZE=8
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ "$NPU_COUNT" -ge 4 ]; then
    export WORLD_SIZE=4
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
else
    export WORLD_SIZE=$NPU_COUNT
    if [ "$NPU_COUNT" -eq 5 ]; then
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4
    elif [ "$NPU_COUNT" -eq 3 ]; then
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2
    elif [ "$NPU_COUNT" -eq 2 ]; then
        export ASCEND_RT_VISIBLE_DEVICES=0,1
    else
        export ASCEND_RT_VISIBLE_DEVICES=0
    fi
fi

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "NPU配置 (自动检测):"
echo "  NPU数量: $NPU_COUNT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"

# 3. 检查是否需要构建tokenizer
if [ ! -f ~/.cache/nanochat/tokenizer_65536.json ]; then
    echo "2. 需要训练tokenizer..."
    echo "提示: 如果遇到rustbpe问题，请运行: bash build_rustbpe.sh"
    
    # 尝试训练tokenizer，如果失败则提供指导
    if ! python -m scripts.tok_train; then
        echo ""
        echo "❌ Tokenizer训练失败！"
        echo "解决方案:"
        echo "1. 运行: bash build_rustbpe.sh"
        echo "2. 或者下载预训练的tokenizer"
        echo "3. 或者使用更小的vocab_size重新训练"
        exit 1
    fi
else
    echo "2. 使用现有tokenizer..."
fi

# 4. 根据NPU数量调整训练参数
if [ "$WORLD_SIZE" -le 2 ]; then
    # 小规模训练（1-2 NPU）
    DEPTH=8
    DEVICE_BATCH_SIZE=8
    TOTAL_BATCH_SIZE=32768
    TARGET_EXAMPLES=16
    RL_EXAMPLES=8
    echo "使用小规模训练配置 (适用于 1-2 NPU)"
elif [ "$WORLD_SIZE" -le 4 ]; then
    # 中等规模训练（3-4 NPU）
    DEPTH=10
    DEVICE_BATCH_SIZE=12
    TOTAL_BATCH_SIZE=131072
    TARGET_EXAMPLES=24
    RL_EXAMPLES=12
    echo "使用中等规模训练配置 (适用于 3-4 NPU)"
else
    # 大规模训练（5+ NPU）
    DEPTH=12
    DEVICE_BATCH_SIZE=16
    TOTAL_BATCH_SIZE=262144
    TARGET_EXAMPLES=32
    RL_EXAMPLES=16
    echo "使用大规模训练配置 (适用于 5+ NPU)"
fi

echo "训练参数:"
echo "  模型深度: $DEPTH"
echo "  设备批次大小: $DEVICE_BATCH_SIZE"
echo "  总批次大小: $TOTAL_BATCH_SIZE"

# 5. 开始训练流程
echo "3. 训练base model (depth=$DEPTH)..."
torchrun --standalone --nproc_per_node=$WORLD_SIZE -m scripts.base_train -- \
    --run=npu_base_d${DEPTH}_${WORLD_SIZE}npu \
    --depth=$DEPTH \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --num_iterations=1000

echo "4. 运行midtraining..."
torchrun --standalone --nproc_per_node=$WORLD_SIZE -m scripts.mid_train -- \
    --run=npu_mid_d${DEPTH}_${WORLD_SIZE}npu \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE

echo "5. Chat SFT..."
torchrun --standalone --nproc_per_node=$WORLD_SIZE -m scripts.chat_sft -- \
    --run=npu_sft_d${DEPTH}_${WORLD_SIZE}npu \
    --device_batch_size=$(($DEVICE_BATCH_SIZE / 2)) \
    --target_examples_per_step=$TARGET_EXAMPLES

echo "6. Chat RL (可选)..."
read -p "是否运行RL训练? 这可能需要很长时间 (y/N): " run_rl
if [[ $run_rl =~ ^[Yy]$ ]]; then
    torchrun --standalone --nproc_per_node=$WORLD_SIZE -m scripts.chat_rl -- \
        --run=npu_rl_d${DEPTH}_${WORLD_SIZE}npu \
        --device_batch_size=$(($DEVICE_BATCH_SIZE / 4)) \
        --examples_per_step=$RL_EXAMPLES
else
    echo "跳过RL训练"
fi

echo "7. 启动web服务..."
read -p "是否启动web服务进行测试? (y/N): " start_web
if [[ $start_web =~ ^[Yy]$ ]]; then
    echo "启动web服务，访问 http://localhost:8000"
    python -m scripts.chat_web
else
    echo "跳过web服务启动"
    echo ""
    echo "可以使用以下命令测试模型:"
    echo "  python -m scripts.chat_cli"
    echo "  python -m scripts.chat_web"
fi

echo ""
echo "=== NPU训练管道完成! ==="
echo "配置总结:"
echo "  使用了 $WORLD_SIZE 个NPU"
echo "  模型深度: $DEPTH"
echo "  模型保存在: ~/.cache/nanochat/"
echo ""
echo "模型文件位置:"
echo "  Base model: ~/.cache/nanochat/base_checkpoints/d${DEPTH}/"
echo "  Mid model:  ~/.cache/nanochat/mid_checkpoints/d${DEPTH}/"
echo "  SFT model:  ~/.cache/nanochat/chatsft_checkpoints/d${DEPTH}/"
if [[ $run_rl =~ ^[Yy]$ ]]; then
echo "  RL model:   ~/.cache/nanochat/chatrl_checkpoints/d${DEPTH}/"
fi
#!/bin/bash

echo "🔄 渐进式NPU分布式训练测试..."
echo "⏰ 开始时间: $(date)"

# 基础环境配置
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export ASCEND_RT_DEBUG_LEVEL=INFO

# 清理进程
cleanup_processes() {
    echo "🧹 清理进程..."
    pkill -f "python.*base_train" || true
    pkill -f "torchrun" || true
    sleep 3
}

# 测试单NPU
test_single_npu() {
    echo "🔥 测试单NPU训练..."
    cleanup_processes
    
    export HCCL_CONNECT_TIMEOUT=60
    export HCCL_EXEC_TIMEOUT=60
    
    python scripts/base_train.py \
        --device_batch_size=4 \
        --total_batch_size=32 \
        --max_seq_len=512 \
        --model_size=124M \
        --learning_rate=0.0006 \
        --warmup_iters=10 \
        --max_iters=50 \
        --eval_every=25 \
        --eval_tokens=5120 \
        --save_every=100 \
        --generate_every=100 \
        --overwrite_output_dir=True \
        --optimizer=adamw \
        --output_dir=./logs/test_1npu
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ 单NPU测试成功"
        return 0
    else
        echo "❌ 单NPU测试失败"
        return 1
    fi
}

# 测试2NPU
test_2npu() {
    echo "🔥 测试2NPU分布式训练..."
    cleanup_processes
    
    export HCCL_CONNECT_TIMEOUT=120
    export HCCL_EXEC_TIMEOUT=120
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=12358
    
    torchrun \
        --nproc_per_node=2 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        scripts/base_train.py \
        --device_batch_size=2 \
        --total_batch_size=32 \
        --max_seq_len=512 \
        --model_size=124M \
        --learning_rate=0.0006 \
        --warmup_iters=10 \
        --max_iters=50 \
        --eval_every=25 \
        --eval_tokens=5120 \
        --save_every=100 \
        --generate_every=100 \
        --overwrite_output_dir=True \
        --use_ddp=True \
        --optimizer=adamw \
        --output_dir=./logs/test_2npu
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ 2NPU测试成功"
        return 0
    else
        echo "❌ 2NPU测试失败"
        return 1
    fi
}

# 测试4NPU
test_4npu() {
    echo "🔥 测试4NPU分布式训练..."
    cleanup_processes
    
    export HCCL_CONNECT_TIMEOUT=300
    export HCCL_EXEC_TIMEOUT=300
    export HCCL_HEARTBEAT_TIMEOUT=300
    export HCCL_REDUCE_OP_SYNC=1
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=12359
    
    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        scripts/base_train.py \
        --device_batch_size=2 \
        --total_batch_size=32 \
        --max_seq_len=512 \
        --model_size=124M \
        --learning_rate=0.0006 \
        --warmup_iters=10 \
        --max_iters=50 \
        --eval_every=25 \
        --eval_tokens=5120 \
        --save_every=100 \
        --generate_every=100 \
        --overwrite_output_dir=True \
        --use_ddp=True \
        --optimizer=adamw \
        --output_dir=./logs/test_4npu
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ 4NPU测试成功"
        return 0
    else
        echo "❌ 4NPU测试失败"
        return 1
    fi
}

# 执行渐进式测试
echo "📊 NPU设备状态:"
npu-smi info | head -10

echo "🚀 开始渐进式测试..."

echo "════════════════════════════════════════"
echo "第1步: 单NPU基准测试"
echo "════════════════════════════════════════"
if test_single_npu; then
    echo "📈 单NPU基准测试通过，继续2NPU测试"
else
    echo "💥 单NPU基准测试失败，停止测试"
    exit 1
fi

echo "════════════════════════════════════════"
echo "第2步: 2NPU分布式测试"
echo "════════════════════════════════════════"
if test_2npu; then
    echo "📈 2NPU分布式测试通过，继续4NPU测试"
else
    echo "💥 2NPU分布式测试失败，停止测试"
    exit 1
fi

echo "════════════════════════════════════════"
echo "第3步: 4NPU分布式测试"
echo "════════════════════════════════════════"
if test_4npu; then
    echo "🎉 4NPU分布式测试成功！"
    echo "🏁 所有测试都通过了！"
else
    echo "💥 4NPU分布式测试失败"
    exit 1
fi

cleanup_processes
echo "⏰ 结束时间: $(date)"
echo "🏁 渐进式测试完成"
#!/bin/bash

echo "📈 渐进式多GPU测试..."

# 基础设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:128"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

function cleanup() {
    pkill -f "python.*base_train.py" 2>/dev/null || true
    pkill -f "torchrun" 2>/dev/null || true
    sleep 2
}

function test_config() {
    local npus=$1
    local depth=$2
    local batch=$3
    local seq_len=$4
    local steps=$5
    local total_batch=$6
    
    echo ""
    echo "=== 测试 ${npus}NPU 配置 ==="
    echo "depth=$depth, batch=$batch, seq_len=$seq_len, steps=$steps"
    
    cleanup
    
    # 设置NPU设备
    if [ $npus -eq 2 ]; then
        export ASCEND_RT_VISIBLE_DEVICES=0,1
        export WORLD_SIZE=2
    else
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
        export WORLD_SIZE=4
    fi
    
    torchrun --standalone --nproc_per_node=$npus -- scripts/base_train.py \
        --depth=$depth \
        --device_batch_size=$batch \
        --total_batch_size=$total_batch \
        --max_seq_len=$seq_len \
        --num_iterations=$steps \
        --eval_every=999999 \
        --core_metric_every=999999 \
        --sample_every=999999 \
        --run="test_${npus}npu_d${depth}_$(date +%H%M%S)"
    
    return $?
}

echo "开始渐进式测试..."

# 测试1: 超小2NPU
echo "🧪 第1轮：超小2NPU测试"
if test_config 2 2 1 128 3 1024; then
    echo "✅ 超小2NPU成功"
else
    echo "❌ 超小2NPU失败，停止测试"
    exit 1
fi

# 测试2: 小2NPU
echo "🧪 第2轮：小2NPU测试"
if test_config 2 4 1 256 5 2048; then
    echo "✅ 小2NPU成功"
else
    echo "❌ 小2NPU失败，但2NPU基础功能正常"
fi

# 测试3: 超小4NPU
echo "🧪 第3轮：超小4NPU测试"
if test_config 4 2 1 128 3 2048; then
    echo "✅ 超小4NPU成功！"
    
    # 测试4: 小4NPU
    echo "🧪 第4轮：小4NPU测试"
    if test_config 4 3 1 256 5 4096; then
        echo "✅ 小4NPU成功！"
        
        # 测试5: 中等4NPU
        echo "🧪 第5轮：中等4NPU测试"
        if test_config 4 6 2 512 10 16384; then
            echo "🎉 中等4NPU成功！多GPU训练完全正常"
        else
            echo "⚠️ 中等4NPU失败，但小配置可用"
        fi
    else
        echo "⚠️ 小4NPU失败，但超小配置可用"
    fi
else
    echo "❌ 超小4NPU失败，4NPU可能有问题"
fi

cleanup
echo ""
echo "🏁 渐进式测试完成: $(date)"
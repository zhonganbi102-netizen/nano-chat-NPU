#!/bin/bash

echo "🔬 4NPU渐进式内存测试..."

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512"

function cleanup_and_test() {
    echo "清理进程和内存..."
    pkill -f "python.*base_train.py" 2>/dev/null || true
    pkill -f "torchrun" 2>/dev/null || true
    sleep 3
    
    python3 -c "
import torch_npu
for i in range(4):
    try:
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
    except:
        pass
"
}

echo "=== 测试1: 超小配置 ==="
cleanup_and_test
echo "配置: depth=4, batch=2, seq_len=512"

torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
    --depth=4 \
    --device_batch_size=2 \
    --total_batch_size=32768 \
    --max_seq_len=512 \
    --num_iterations=5 \
    --run="4npu_tiny_test"

if [ $? -eq 0 ]; then
    echo "✅ 超小配置成功!"
    
    echo ""
    echo "=== 测试2: 小配置 ==="
    cleanup_and_test
    echo "配置: depth=6, batch=3, seq_len=1024"
    
    torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
        --depth=6 \
        --device_batch_size=3 \
        --total_batch_size=65536 \
        --max_seq_len=1024 \
        --num_iterations=5 \
        --run="4npu_small_test"
    
    if [ $? -eq 0 ]; then
        echo "✅ 小配置成功!"
        
        echo ""
        echo "=== 测试3: 中配置 ==="
        cleanup_and_test
        echo "配置: depth=8, batch=4, seq_len=1024"
        
        torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
            --depth=8 \
            --device_batch_size=4 \
            --total_batch_size=131072 \
            --max_seq_len=1024 \
            --num_iterations=5 \
            --run="4npu_medium_test"
    else
        echo "❌ 小配置失败"
    fi
else
    echo "❌ 超小配置失败，内存问题严重"
fi

echo ""
echo "测试完成: $(date)"
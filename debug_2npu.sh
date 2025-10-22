#!/bin/bash

echo "🔍 详细调试2NPU训练..."

# 设置调试环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 启用详细调试
export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 内存设置
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:256"

# 清理
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 3

echo "=== 启动前检查 ==="
python3 -c "
import torch_npu
import torch

print('检查NPU状态...')
for i in range(2):
    try:
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize()
        
        # 测试基本操作
        x = torch.randn(10, 10, device=f'npu:{i}')
        y = x @ x.T
        print(f'NPU {i}: 基本操作测试通过')
        del x, y
        torch_npu.npu.empty_cache()
    except Exception as e:
        print(f'NPU {i}: 测试失败 - {e}')
        exit(1)

print('所有NPU基本功能正常')
"

if [ $? -ne 0 ]; then
    echo "❌ NPU基本功能测试失败"
    exit 1
fi

echo ""
echo "=== 启动详细调试的2NPU训练 ==="
echo "配置: depth=4, batch=1, 仅5步"

# 使用最小配置和详细日志
torchrun \
    --standalone \
    --nproc_per_node=2 \
    --log_dir=./debug_logs \
    -- \
    scripts/base_train.py \
    --depth=4 \
    --device_batch_size=1 \
    --total_batch_size=8192 \
    --max_seq_len=512 \
    --num_iterations=5 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="debug_2npu_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=== 查看错误日志 ==="
if [ -d "./debug_logs" ]; then
    echo "Torchrun 日志:"
    ls -la ./debug_logs/ || echo "无日志文件"
    for log_file in ./debug_logs/*.log; do
        if [ -f "$log_file" ]; then
            echo "--- $log_file ---"
            tail -20 "$log_file"
        fi
    done
else
    echo "未生成调试日志目录"
fi

echo ""
echo "调试完成: $(date)"
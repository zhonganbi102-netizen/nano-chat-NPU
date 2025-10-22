#!/bin/bash

echo "🔧 修复HCCL通信问题的4NPU训练..."

# 设置基础环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL通信优化设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export HCCL_CONNECT_TIMEOUT=600  # 增加连接超时时间到10分钟
export HCCL_EXEC_TIMEOUT=600     # 增加执行超时时间
export HCCL_BUFFSIZE=512         # 设置缓冲区大小

# NPU同步和调试设置
export ASCEND_LAUNCH_BLOCKING=1   # 同步模式，便于调试
export ASCEND_GLOBAL_LOG_LEVEL=1  # 详细日志
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 内存优化设置
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512"

# 清理进程和内存
echo "清理之前的进程..."
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 5

# 清理NPU内存和状态
echo "重置NPU状态..."
python3 -c "
import torch_npu
import time

print('清理NPU内存...')
for i in range(4):
    try:
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize()
        print(f'NPU {i} 清理完成')
    except Exception as e:
        print(f'NPU {i} 清理失败: {e}')

time.sleep(2)
print('NPU状态重置完成')
"

# 检查NPU通信能力
echo "检查NPU设备状态..."
python3 -c "
import torch_npu
import torch
import torch.distributed as dist

try:
    device_count = torch_npu.npu.device_count()
    print(f'可用NPU数量: {device_count}')
    
    for i in range(min(4, device_count)):
        torch_npu.npu.set_device(i)
        print(f'NPU {i}: {torch_npu.npu.get_device_name(i)}')
        
        # 测试内存分配
        x = torch.randn(100, 100, device=f'npu:{i}')
        print(f'NPU {i} 内存测试通过')
        del x
        torch_npu.npu.empty_cache()
        
except Exception as e:
    print(f'NPU检查失败: {e}')
"

echo ""
echo "启动HCCL优化的4NPU训练..."
echo "配置: 同步模式，延长超时，优化通信"

# 启动训练，跳过初始评估
torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
    --depth=8 \
    --device_batch_size=4 \
    --total_batch_size=131072 \
    --max_seq_len=1024 \
    --num_iterations=50 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="4npu_hccl_fix_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "训练完成: $(date)"
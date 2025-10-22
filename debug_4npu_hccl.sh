#!/bin/bash

echo "🔍 调试4NPU HCCL通信问题..."

# 1. 检查NPU设备状态
echo "📊 检查NPU设备状态:"
npu-smi info

# 2. 检查HCCL通信环境
echo "🔗 检查HCCL通信环境:"
export HCCL_CONNECT_TIMEOUT=300  # 增加超时时间到5分钟
export HCCL_EXEC_TIMEOUT=300
export ASCEND_LAUNCH_BLOCKING=1  # 同步模式，便于调试
export PYTHONFAULTHANDLER=1

# 3. 测试基础HCCL通信
echo "🧪 测试基础HCCL通信..."
cat > test_hccl_simple.py << 'EOF'
import os
import torch
import torch_npu
import torch.distributed as dist
from datetime import timedelta

def init_process():
    try:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        print(f"[Rank {rank}] 初始化进程组...")
        
        # 设置设备
        torch_npu.npu.set_device(local_rank)
        
        # 初始化进程组，使用timedelta设置超时
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5)  # 5分钟超时
        )
        
        print(f"[Rank {rank}] 进程组初始化成功")
        
        # 同步所有进程
        dist.barrier()
        print(f"[Rank {rank}] 所有进程已同步")
        
        # 测试简单的all_reduce
        device = f'npu:{local_rank}'
        tensor = torch.ones(10, device=device) * rank
        print(f"[Rank {rank}] 发送张量: {tensor[:5]}")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] 接收张量: {tensor[:5]}")
        
        # 再次同步
        dist.barrier()
        print(f"[Rank {rank}] all_reduce操作完成")
        
        # 测试expected结果 (0+1+2+3=6 for each element)
        expected = torch.ones(10, device=device) * 6
        if torch.allclose(tensor[:5], expected[:5]):
            print(f"[Rank {rank}] ✅ all_reduce结果正确!")
        else:
            print(f"[Rank {rank}] ❌ all_reduce结果错误! 期望:{expected[:5]} 实际:{tensor[:5]}")
        
        # 清理
        dist.destroy_process_group()
        print(f"[Rank {rank}] 测试完成!")
        
    except Exception as e:
        print(f"[Rank {rank}] 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    init_process()
EOF

# 4. 运行HCCL通信测试
echo "🚀 运行HCCL通信测试..."
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12355 \
    test_hccl_simple.py

echo "✅ HCCL通信测试完成"
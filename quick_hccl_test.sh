#!/bin/bash

echo "🔧 快速HCCL通信测试..."

# 清理之前的进程
pkill -f "test_hccl" || true
sleep 2

# 设置基础环境
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=120
export PYTHONFAULTHANDLER=1

# 创建简化测试
cat > simple_hccl_test.py << 'EOF'
import os
import torch
import torch_npu
import torch.distributed as dist
from datetime import timedelta

def main():
    # 获取分布式参数
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"[Rank {rank}] 开始测试，world_size={world_size}, local_rank={local_rank}")
    
    try:
        # 设置NPU设备
        torch_npu.npu.set_device(local_rank)
        device = f'npu:{local_rank}'
        print(f"[Rank {rank}] 设备设置完成: {device}")
        
        # 初始化分布式进程组
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=120)
        )
        print(f"[Rank {rank}] 分布式进程组初始化成功")
        
        # 创建测试张量
        test_tensor = torch.ones(5, device=device) * rank
        print(f"[Rank {rank}] 原始张量: {test_tensor}")
        
        # 执行all_reduce
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] all_reduce后: {test_tensor}")
        
        # 验证结果
        expected_sum = sum(range(world_size))  # 0+1+2+3=6 for 4 ranks
        if torch.allclose(test_tensor, torch.ones(5, device=device) * expected_sum):
            print(f"[Rank {rank}] ✅ all_reduce测试通过!")
        else:
            print(f"[Rank {rank}] ❌ all_reduce测试失败!")
            
        # 同步所有进程
        dist.barrier()
        print(f"[Rank {rank}] 所有进程同步完成")
        
        # 清理
        dist.destroy_process_group()
        print(f"[Rank {rank}] 测试完成，清理成功")
        
    except Exception as e:
        print(f"[Rank {rank}] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
EOF

echo "🚀 运行4NPU HCCL通信测试..."

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12356 \
    simple_hccl_test.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ HCCL通信测试成功！"
else
    echo "❌ HCCL通信测试失败，退出码: $exit_code"
fi

# 清理临时文件
rm -f simple_hccl_test.py

echo "🏁 测试完成"
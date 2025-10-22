#!/bin/bash

echo "🔬 基础分布式通信测试..."

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 清理
pkill -f "python.*test_dist.py" 2>/dev/null || true
sleep 2

# 创建简单的分布式测试脚本
cat > test_dist.py << 'EOF'
import os
import torch
import torch.distributed as dist
import torch_npu

def main():
    try:
        print(f"进程开始，环境变量:")
        print(f"RANK: {os.environ.get('RANK', 'None')}")
        print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'None')}")
        print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'None')}")
        
        # 设置设备
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch_npu.npu.set_device(local_rank)
        device = f'npu:{local_rank}'
        
        print(f"设置设备: {device}")
        
        # 初始化分布式
        dist.init_process_group(backend='hccl')
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        print(f"分布式初始化成功! Rank: {rank}, World Size: {world_size}")
        
        # 测试简单的all_reduce
        tensor = torch.ones(2, device=device) * rank
        print(f"Rank {rank}: 初始tensor = {tensor}")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {rank}: all_reduce后 = {tensor}")
        
        # 同步
        dist.barrier()
        print(f"Rank {rank}: barrier同步完成")
        
        # 清理
        dist.destroy_process_group()
        print(f"Rank {rank}: 进程组清理完成")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
EOF

echo "启动基础分布式测试..."

torchrun --standalone --nproc_per_node=2 test_dist.py

if [ $? -eq 0 ]; then
    echo "✅ 基础分布式通信测试成功!"
    echo "可以继续尝试训练脚本"
else
    echo "❌ 基础分布式通信测试失败"
    echo "需要检查HCCL环境配置"
fi

# 清理测试文件
rm -f test_dist.py

echo "基础测试完成: $(date)"
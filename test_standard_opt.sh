#!/bin/bash

echo "🔧 使用标准优化器的2NPU测试..."

# 环境变量设置
export ASCEND_RT_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 内存设置
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:128"

# 清理
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 2

echo "创建临时训练脚本，使用标准AdamW优化器..."

# 创建使用标准优化器的测试脚本
cat > test_standard_optimizer.py << 'EOF'
import os
import torch
import torch.distributed as dist
import torch_npu

def setup_distributed():
    dist.init_process_group(backend='hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch_npu.npu.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = setup_distributed()
    device = f'npu:{local_rank}'
    
    print(f"Rank {rank}/{world_size} on device {device}")
    
    # 创建简单模型
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32)
    ).to(device)
    
    # 使用DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )
    
    # 使用标准AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    print(f"Rank {rank}: 模型和优化器初始化完成")
    
    # 简单训练循环
    for step in range(5):
        # 创建随机数据
        inputs = torch.randn(4, 128, device=device)
        targets = torch.randn(4, 32, device=device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Rank {rank}: Step {step}, Loss: {loss.item():.4f}")
    
    dist.destroy_process_group()
    print(f"Rank {rank}: 训练完成")

if __name__ == "__main__":
    main()
EOF

echo "启动标准优化器2NPU测试..."

torchrun --standalone --nproc_per_node=2 test_standard_optimizer.py

if [ $? -eq 0 ]; then
    echo "✅ 标准优化器2NPU测试成功！"
    echo "问题确实在自定义的DistAdamW优化器"
else
    echo "❌ 标准优化器2NPU测试失败"
fi

# 清理测试文件
rm -f test_standard_optimizer.py

echo "标准优化器测试完成: $(date)"
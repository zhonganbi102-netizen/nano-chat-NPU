#!/bin/bash

echo "=== 快速NPU训练验证 ==="

# 清理环境
source clean_npu_environment.sh

# 设置单NPU环境
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export ASCEND_RT_VISIBLE_DEVICES=0

cd /mnt/linxid615/bza/nanochat-npu

echo "运行快速NPU训练验证..."

PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

import torch
try:
    import torch_npu
    print('✅ torch_npu导入成功')
except ImportError:
    print('❌ torch_npu导入失败')
    exit(1)

import time

# 禁用torch.compile
torch._dynamo.config.disable = True
print('⚠️  已禁用torch.compile')

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, print_banner

print_banner()

print('=== 快速NPU训练验证 ===')

# 最简设置
depth = 2
max_seq_len = 128
num_iterations = 2
device_batch_size = 1
vocab_size = 1000

# 分布式初始化
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
device_type = 'npu' if device.type == 'npu' else 'cuda'

print0(f'设备: {device}, 类型: {device_type}')

# 模型配置
model_config = GPTConfig(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=2,
    n_head=2,
    n_kv_head=2,
    n_embd=128
)

print0('创建模型...')
with torch.device('meta'):
    model = GPT(model_config)

model.to_empty(device=device)
model.init_weights()

# 跳过torch.compile
print0('跳过模型编译（NPU兼容）')

# 使用简化优化器
optimizer = model.configure_optimizers(
    weight_decay=0.01,
    learning_rate=1e-4,
    device_type=device_type
)

print0('✅ 模型和优化器准备完成')

# 训练循环
print0('\\n=== 开始训练 ===')
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

for step in range(num_iterations):
    print0(f'\\n步骤 {step + 1}/{num_iterations}')
    
    # 同步
    if device_type == 'npu':
        torch_npu.npu.synchronize()
    else:
        torch.cuda.synchronize()
    t0 = time.time()
    
    # 创建数据
    x = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
    y = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
    
    # 训练步骤
    optimizer.zero_grad()
    
    with autocast_ctx:
        loss = model(x, y)
    
    loss.backward()
    optimizer.step()
    
    # 同步
    if device_type == 'npu':
        torch_npu.npu.synchronize()
    else:
        torch.cuda.synchronize()
    t1 = time.time()
    
    dt = t1 - t0
    tokens_per_sec = int(device_batch_size * max_seq_len / dt)
    
    print0(f'  损失: {loss.item():.4f}')
    print0(f'  时间: {dt*1000:.1f}ms')
    print0(f'  速度: {tokens_per_sec:,} tokens/sec')
    
    if device_type == 'npu':
        mem_mb = torch_npu.npu.memory_allocated(0) / 1024**2
    else:
        mem_mb = torch.cuda.memory_allocated(0) / 1024**2
    print0(f'  内存: {mem_mb:.1f}MB')

# 推理测试
print0('\\n=== 推理测试 ===')
model.eval()
with torch.no_grad():
    test_x = torch.randint(0, vocab_size, (1, 16), device=device)
    with autocast_ctx:
        logits = model(test_x)
    print0(f'✅ 推理成功: {logits.shape}')

print0('\\n=== 验证完成 ===')
print0('✅ NPU训练验证成功！')

# 清理
compute_cleanup()
"

echo "快速验证完成"
#!/bin/bash

echo "=== 完整NPU训练测试（无需数据文件）==="

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

echo "测试完整的NPU训练流程（模拟数据）..."

PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

import torch
import torch_npu
import time
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import get_dist_info, print0

# 设置随机种子
torch.manual_seed(1337)

# 分布式设置
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
device = torch.device('npu:0')
device_type = 'npu'

print('=== 测试完整训练设置 ===')

# 模型配置（小规模测试）
depth = 2
max_seq_len = 256
device_batch_size = 2
total_batch_size = 1024
num_iterations = 3

# 计算模型参数
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads

print(f'模型配置: layers={num_layers}, dim={model_dim}, heads={num_heads}')

# 创建模型
model_config_kwargs = dict(
    sequence_len=max_seq_len, 
    vocab_size=1000, 
    n_layer=num_layers, 
    n_head=num_heads, 
    n_kv_head=num_kv_heads, 
    n_embd=model_dim
)

with torch.device('meta'):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

print('✅ Meta模型创建成功')

# 移动到NPU并初始化
model.to_empty(device=device)
model.init_weights()
print('✅ 模型初始化到NPU成功')

# 设置优化器
optimizers = model.setup_optimizers(
    unembedding_lr=0.002, 
    embedding_lr=0.01, 
    matrix_lr=0.01, 
    weight_decay=0.01
)
adamw_optimizer, muon_optimizer = optimizers
print('✅ 优化器设置成功')

# 计算梯度累积步数
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

print(f'批次配置: device_batch={device_batch_size}, seq_len={max_seq_len}')
print(f'梯度累积步数: {grad_accum_steps}')

# 模拟训练循环
print('\\n=== 开始模拟训练 ===')
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

for step in range(num_iterations):
    print(f'\\n步骤 {step + 1}/{num_iterations}:')
    
    # 同步和计时
    torch_npu.npu.synchronize()
    t0 = time.time()
    
    # 梯度累积循环
    total_loss = 0.0
    for micro_step in range(grad_accum_steps):
        # 创建模拟数据
        x = torch.randint(0, 1000, (device_batch_size, max_seq_len), device=device)
        y = torch.randint(0, 1000, (device_batch_size, max_seq_len), device=device)
        
        with autocast_ctx:
            loss = model(x, y)
        
        train_loss = loss.detach()
        total_loss += train_loss.item()
        loss = loss / grad_accum_steps
        loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 优化器步骤
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    
    # 同步和计时
    torch_npu.npu.synchronize()
    t1 = time.time()
    dt = t1 - t0
    
    avg_loss = total_loss / grad_accum_steps
    tokens_per_sec = int(world_tokens_per_fwdbwd / dt)
    
    print(f'  损失: {avg_loss:.4f}')
    print(f'  时间: {dt*1000:.1f}ms')
    print(f'  速度: {tokens_per_sec:,} tokens/sec')
    print(f'  内存: {torch_npu.npu.memory_allocated(0) / 1024**2:.1f}MB')

print('\\n=== 测试推理模式 ===')
model.eval()
with torch.no_grad():
    test_x = torch.randint(0, 1000, (1, 32), device=device)
    with autocast_ctx:
        logits = model(test_x)
    print(f'✅ 推理成功，输出形状: {logits.shape}')

print('\\n=== 测试检查点保存 ===')
checkpoint = {
    'model': model.state_dict(),
    'config': model_config_kwargs,
    'step': num_iterations,
    'optimizers': [opt.state_dict() for opt in optimizers]
}

torch.save(checkpoint, 'test_full_checkpoint.pt')
print('✅ 检查点保存成功')

# 清理
if os.path.exists('test_full_checkpoint.pt'):
    os.remove('test_full_checkpoint.pt')
    print('✅ 测试文件清理完成')

print(f'\\n=== 测试完成 ===')
print(f'总计训练{num_iterations}步')
print(f'最大内存使用: {torch_npu.npu.max_memory_allocated(0) / 1024**2:.1f}MB')
print('✅ 完整NPU训练流程测试成功！')
"

echo "完整训练测试完成"
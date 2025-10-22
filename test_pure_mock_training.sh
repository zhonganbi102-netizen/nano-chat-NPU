#!/bin/bash

echo "=== 纯模拟数据训练测试 ==="

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

echo "运行纯模拟数据训练（完全跳过数据加载器）..."

PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

import torch
import torch_npu
import time

# 禁用torch.compile
torch._dynamo.config.disable = True
print('⚠️  已禁用torch.compile以避免NPU编译器问题')

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner

print_banner()

# 训练设置
run = 'pure_mock_data'
depth = 2
max_seq_len = 256
num_iterations = 3
device_batch_size = 2
total_batch_size = 1024
embedding_lr = 0.01
unembedding_lr = 0.002
weight_decay = 0.01
matrix_lr = 0.01
grad_clip = 1.0

print('=== 纯模拟数据训练设置 ===')
print(f'深度: {depth}, 序列长度: {max_seq_len}')
print(f'批次大小: {device_batch_size}, 总批次: {total_batch_size}')
print(f'训练步数: {num_iterations}')

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
device_type = 'npu' if device.type == 'npu' else 'cuda'
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

print0(f'设备类型: {device_type}, 设备: {device}')

# 使用固定词汇表大小（避免tokenizer依赖）
vocab_size = 1000
print0(f'使用固定词汇表大小: {vocab_size:,}')

# Model kwargs
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads
print0(f'模型层数: {num_layers}, 维度: {model_dim}, 头数: {num_heads}')

# 计算梯度累积
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f'梯度累积步数: {grad_accum_steps}')

# 初始化模型
model_config_kwargs = dict(
    sequence_len=max_seq_len, 
    vocab_size=vocab_size, 
    n_layer=num_layers, 
    n_head=num_heads, 
    n_kv_head=num_kv_heads, 
    n_embd=model_dim
)

with torch.device('meta'):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

model.to_empty(device=device)
model.init_weights()
orig_model = model

# NPU兼容性：跳过编译
if device_type == 'npu':
    print0('NPU环境：跳过torch.compile')
else:
    model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
print0(f'参数数量: {num_params:,}')

# 初始化优化器
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr, 
    embedding_lr=embedding_lr, 
    matrix_lr=matrix_lr, 
    weight_decay=weight_decay
)
adamw_optimizer, muon_optimizer = optimizers
print0('✅ 优化器初始化成功')

# 完全跳过数据加载器，使用纯模拟数据
print0('✅ 跳过数据加载器，使用纯模拟数据')

print0('\\n=== 开始纯模拟数据训练 ===')

# 训练循环
for step in range(num_iterations + 1):
    last_step = step == num_iterations
    
    if last_step:
        print0(f'✅ 纯模拟数据训练完成！总共{num_iterations}步')
        break
    
    print0(f'\\n步骤 {step + 1}/{num_iterations}')
    
    # 同步计时
    if device_type == 'npu':
        torch_npu.npu.synchronize()
    else:
        torch.cuda.synchronize()
    t0 = time.time()
    
    # 梯度累积
    total_loss = 0.0
    for micro_step in range(grad_accum_steps):
        # 纯模拟数据
        x = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
        y = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
        
        with autocast_ctx:
            loss = model(x, y)
        
        train_loss = loss.detach()
        total_loss += train_loss.item()
        loss = loss / grad_accum_steps
        loss.backward()
    
    # 梯度裁剪
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
    
    # 优化器步骤
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    
    # 计时
    if device_type == 'npu':
        torch_npu.npu.synchronize()
    else:
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    
    # 统计
    avg_loss = total_loss / grad_accum_steps
    tokens_per_sec = int(world_tokens_per_fwdbwd / dt)
    
    print0(f'  损失: {avg_loss:.4f}')
    print0(f'  时间: {dt*1000:.1f}ms')
    print0(f'  速度: {tokens_per_sec:,} tokens/sec')
    
    if device_type == 'npu':
        mem_mb = torch_npu.npu.memory_allocated(0) / 1024**2
    else:
        mem_mb = torch.cuda.memory_allocated(0) / 1024**2
    print0(f'  内存: {mem_mb:.1f}MB')

# 测试推理
print0('\\n=== 测试推理模式 ===')
model.eval()
with torch.no_grad():
    test_x = torch.randint(0, vocab_size, (1, 32), device=device)
    with autocast_ctx:
        logits = model(test_x)
    print0(f'✅ 推理成功，输出形状: {logits.shape}')

# 测试检查点保存
print0('\\n=== 测试检查点保存 ===')
checkpoint = {
    'model': orig_model.state_dict(),
    'config': model_config_kwargs,
    'step': num_iterations,
    'optimizers': [opt.state_dict() for opt in optimizers]
}

torch.save(checkpoint, 'test_pure_mock_checkpoint.pt')
print0('✅ 检查点保存成功')

# 清理
if os.path.exists('test_pure_mock_checkpoint.pt'):
    os.remove('test_pure_mock_checkpoint.pt')
    print0('✅ 测试文件清理完成')

# 统计
if device_type == 'npu':
    max_mem_mb = torch_npu.npu.max_memory_allocated(0) / 1024**2
else:
    max_mem_mb = torch.cuda.max_memory_allocated(0) / 1024**2

print0(f'\\n=== 纯模拟数据训练完成 ===')
print0(f'总计训练{num_iterations}步')
print0(f'最大内存使用: {max_mem_mb:.1f}MB')
print0('✅ 纯模拟数据训练流程测试成功！')
print0('\\n注意: 完全跳过了数据加载器，使用随机生成的训练数据')

# 清理
compute_cleanup()
"

echo "纯模拟数据训练测试完成"
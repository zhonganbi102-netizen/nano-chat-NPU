#!/bin/bash

echo "=== 简化NPU训练测试（无分布式）==="

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

echo "测试简化的NPU训练流程（模拟数据，无分布式）..."

PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

import torch
import torch_npu
import time
from nanochat.gpt import GPT, GPTConfig

# 设置随机种子
torch.manual_seed(1337)

# 设备设置
device = torch.device('npu:0')
device_type = 'npu'

print('=== 测试简化训练设置 ===')

# 模型配置（小规模测试）
max_seq_len = 256
device_batch_size = 2
num_iterations = 3

# 创建模型
model_config_kwargs = dict(
    sequence_len=max_seq_len, 
    vocab_size=1000, 
    n_layer=2, 
    n_head=4, 
    n_kv_head=4, 
    n_embd=128
)

with torch.device('meta'):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

print('✅ Meta模型创建成功')

# 移动到NPU并初始化
model.to_empty(device=device)
model.init_weights()
print('✅ 模型初始化到NPU成功')

# 使用简化的优化器（避免分布式问题）
optimizer = model.configure_optimizers(
    weight_decay=0.01, 
    learning_rate=1e-4, 
    device_type='npu'
)
print('✅ 简化优化器设置成功')

# 模拟训练循环
print('\\n=== 开始简化训练 ===')
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

for step in range(num_iterations):
    print(f'\\n步骤 {step + 1}/{num_iterations}:')
    
    # 同步和计时
    torch_npu.npu.synchronize()
    t0 = time.time()
    
    # 创建模拟数据
    x = torch.randint(0, 1000, (device_batch_size, max_seq_len), device=device)
    y = torch.randint(0, 1000, (device_batch_size, max_seq_len), device=device)
    
    # 训练步骤
    optimizer.zero_grad()
    
    with autocast_ctx:
        loss = model(x, y)
    
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 优化器步骤
    optimizer.step()
    
    # 同步和计时
    torch_npu.npu.synchronize()
    t1 = time.time()
    dt = t1 - t0
    
    tokens_per_sec = int(device_batch_size * max_seq_len / dt)
    
    print(f'  损失: {loss.item():.4f}')
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
    'optimizer': optimizer.state_dict()
}

torch.save(checkpoint, 'test_simple_checkpoint.pt')
print('✅ 检查点保存成功')

# 清理
if os.path.exists('test_simple_checkpoint.pt'):
    os.remove('test_simple_checkpoint.pt')
    print('✅ 测试文件清理完成')

print(f'\\n=== 简化测试完成 ===')
print(f'总计训练{num_iterations}步')
print(f'最大内存使用: {torch_npu.npu.max_memory_allocated(0) / 1024**2:.1f}MB')
print('✅ 简化NPU训练流程测试成功！')
print('\\n注意: 这是简化版本，不包含分布式训练和高级优化器')
"

echo "简化训练测试完成"
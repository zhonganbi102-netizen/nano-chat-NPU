#!/bin/bash

echo "=== 极简NPU测试 ==="

# 只使用一张NPU
export ASCEND_RT_VISIBLE_DEVICES=0

echo "停止现有训练进程..."
pkill -f "python.*base_train"
sleep 5

echo "开始极简测试..."
PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

import torch
import torch_npu
from nanochat.gpt import GPT, GPTConfig
import time

print('1. 基础环境检查...')
print(f'NPU可用: {torch_npu.npu.is_available()}')
torch_npu.npu.set_device(0)
print(f'当前设备: {torch_npu.npu.current_device()}')

print('\\n2. 创建小模型...')
config = GPTConfig(
    sequence_len=512,
    vocab_size=1000, 
    n_layer=2,
    n_head=2,
    n_kv_head=2,
    n_embd=128
)

print('\\n3. 在NPU上初始化模型...')
start_time = time.time()
device = torch.device('npu:0')

with torch.device('meta'):
    model = GPT(config)

print(f'Meta模型创建时间: {time.time() - start_time:.2f}s')

print('\\n4. 移动到NPU...')
start_time = time.time()
model.to_empty(device=device)
model.init_weights()
print(f'NPU移动时间: {time.time() - start_time:.2f}s')

print('\\n5. 测试前向传播...')
start_time = time.time()
batch_size = 2
seq_len = 64
x = torch.randint(0, 1000, (batch_size, seq_len), device=device)

with torch.no_grad():
    logits = model(x)

print(f'前向传播时间: {time.time() - start_time:.2f}s')
print(f'输出形状: {logits.shape}')

print('\\n6. 检查NPU内存使用...')
allocated = torch_npu.npu.memory_allocated(0)
reserved = torch_npu.npu.memory_reserved(0)
print(f'NPU内存分配: {allocated / 1024**2:.1f} MB')
print(f'NPU内存预留: {reserved / 1024**2:.1f} MB')

print('\\n✅ 极简测试完成！NPU基本功能正常。')
print('问题可能在于完整训练脚本的某个特定部分。')
"

echo "极简测试完成"
#!/bin/bash

echo "=== 使用本地模拟数据训练测试 ==="

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

echo "检查或创建本地模拟数据..."

PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

# 确保有模拟数据
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nanochat.dataset import DATA_DIR, list_parquet_files

os.makedirs(DATA_DIR, exist_ok=True)

# 检查是否已有数据
existing_files = list_parquet_files()
if len(existing_files) < 2:
    print('创建快速模拟数据...')
    
    # 创建训练数据
    train_texts = [
        'The quick brown fox jumps over the lazy dog.',
        'Machine learning is transforming the world of technology.',
        'Deep neural networks can learn complex patterns in data.',
        'Natural language processing enables computers to understand text.',
        'Artificial intelligence is the future of computing.',
        'Python is a popular programming language for AI development.',
        'Data science combines statistics and computer science.',
        'Large language models have revolutionized NLP tasks.',
    ] * 50  # 400条记录
    
    train_df = pd.DataFrame({'text': train_texts})
    train_table = pa.Table.from_pandas(train_df)
    train_file = os.path.join(DATA_DIR, 'shard_00000.parquet')
    pq.write_table(train_table, train_file)
    print(f'✅ 创建训练文件: {len(train_texts)} 条记录')
    
    # 创建验证数据
    val_texts = [
        'This is validation text for model evaluation.',
        'Testing helps ensure model quality and performance.',
        'Validation data should be separate from training data.',
    ] * 20  # 60条记录
    
    val_df = pd.DataFrame({'text': val_texts})
    val_table = pa.Table.from_pandas(val_df)
    val_file = os.path.join(DATA_DIR, 'shard_00001.parquet')
    pq.write_table(val_table, val_file)
    print(f'✅ 创建验证文件: {len(val_texts)} 条记录')
else:
    print(f'✅ 已有{len(existing_files)}个数据文件')

print('\\n开始本地数据训练测试...')
"

echo "运行本地数据训练..."

PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

import torch
import torch_npu
import time

# 禁用torch.compile
torch._dynamo.config.disable = True

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, print_banner
from nanochat.tokenizer import get_tokenizer

print_banner()

print('=== 本地数据训练测试 ===')

# 训练设置
depth = 2
max_seq_len = 128
num_iterations = 3
device_batch_size = 2
total_batch_size = 512

# 分布式初始化
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
device_type = 'npu' if device.type == 'npu' else 'cuda'

print0(f'设备: {device}, 类型: {device_type}')

# 获取tokenizer和词汇表
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f'词汇表大小: {vocab_size}')

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

# 尝试真实数据加载器
print0('\\n尝试使用真实数据加载器...')
try:
    train_loader = tokenizing_distributed_data_loader(
        device_batch_size, max_seq_len, split='train'
    )
    
    # 测试获取一批数据
    print0('获取第一批数据...')
    x, y = next(train_loader)
    print0(f'✅ 真实数据加载成功: {x.shape}, {y.shape}')
    use_real_data = True
    
except Exception as e:
    print0(f'❌ 真实数据加载失败: {e}')
    print0('切换到模拟数据')
    use_real_data = False

# 训练循环
print0('\\n=== 开始训练 ===')
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

# 计算梯度累积
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f'梯度累积步数: {grad_accum_steps}')

for step in range(num_iterations):
    print0(f'\\n步骤 {step + 1}/{num_iterations}')
    
    # 同步
    if device_type == 'npu':
        torch_npu.npu.synchronize()
    else:
        torch.cuda.synchronize()
    t0 = time.time()
    
    # 梯度累积循环
    total_loss = 0.0
    for micro_step in range(grad_accum_steps):
        if use_real_data:
            try:
                if micro_step == 0 and step == 0:
                    # 第一步已经有数据了
                    pass
                else:
                    x, y = next(train_loader)
            except Exception as e:
                print0(f'  数据获取失败，使用模拟数据: {e}')
                x = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
                y = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
        else:
            # 模拟数据
            x = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
            y = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
        
        with autocast_ctx:
            loss = model(x, y)
        
        train_loss = loss.detach()
        total_loss += train_loss.item()
        loss = loss / grad_accum_steps
        loss.backward()
    
    # 优化器步骤
    optimizer.zero_grad()
    optimizer.step()
    
    # 同步
    if device_type == 'npu':
        torch_npu.npu.synchronize()
    else:
        torch.cuda.synchronize()
    t1 = time.time()
    
    dt = t1 - t0
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

# 推理测试
print0('\\n=== 推理测试 ===')
model.eval()
with torch.no_grad():
    test_x = torch.randint(0, vocab_size, (1, 16), device=device)
    with autocast_ctx:
        logits = model(test_x)
    print0(f'✅ 推理成功: {logits.shape}')

print0('\\n=== 本地数据训练完成 ===')
data_type = '真实数据' if use_real_data else '模拟数据'
print0(f'✅ 使用{data_type}训练成功！')

# 清理
compute_cleanup()
"

echo "本地数据训练测试完成"
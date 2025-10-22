#!/bin/bash

echo "=== 渐进式NPU训练测试 ==="

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

echo "1. 测试数据加载..."
python3 -c "
import torch
from nanochat.dataloader import tokenizing_distributed_data_loader
try:
    train_loader = tokenizing_distributed_data_loader(B=2, T=32, split='train')
    print('✅ 数据加载器创建成功')
    # 注意: 实际使用需要parquet数据文件，这里只测试创建
    print('数据加载器函数可用')
except Exception as e:
    print(f'❌ 数据加载失败: {e}')
    exit(1)
"

echo "2. 测试模型+优化器..."
python3 -c "
import torch
import torch_npu
from nanochat.gpt import GPT, GPTConfig

torch.manual_seed(1337)
# torch_npu不需要单独设置seed，torch.manual_seed已经够了

device = 'npu:0'
config = GPTConfig(sequence_len=64, vocab_size=1000, n_layer=2, n_head=4, n_kv_head=4, n_embd=128)

try:
    model = GPT(config)
    model.to(device)
    print('✅ 模型创建成功')
    
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-4, device_type='npu')
    print('✅ 优化器创建成功')
    
    # 测试一步训练
    x = torch.randint(0, 1000, (2, 64), device=device)
    y = torch.randint(0, 1000, (2, 64), device=device)
    
    # 测试训练模式（with targets）
    loss = model(x, y)
    print(f'✅ 前向传播成功，损失: {loss.item():.4f}')
    
    # 测试推理模式（without targets）
    with torch.no_grad():
        logits = model(x)
        print(f'✅ 推理模式成功，输出形状: {logits.shape}')
    
    optimizer.zero_grad()
    loss.backward()
    print('✅ 反向传播成功')
    
    optimizer.step()
    print('✅ 优化器步骤成功')
    
except Exception as e:
    print(f'❌ 模型/优化器测试失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo "3. 测试简化训练循环..."
python3 -c "
import torch
import torch_npu
import time
from nanochat.gpt import GPT, GPTConfig

torch.manual_seed(1337)

device = 'npu:0'
config = GPTConfig(sequence_len=32, vocab_size=1000, n_layer=2, n_head=4, n_kv_head=4, n_embd=128)

try:
    model = GPT(config)
    model.to(device)
    model.train()
    
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-4, device_type='npu')
    
    print('开始训练循环测试...')
    for step in range(3):  # 只测试3步
        t0 = time.time()
        
        # 创建假数据（实际应该用数据加载器）
        x = torch.randint(0, 1000, (2, 32), device=device)
        y = torch.randint(0, 1000, (2, 32), device=device)
        
        optimizer.zero_grad()
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        dt = time.time() - t0
        print(f'步骤 {step}: 损失={loss.item():.4f}, 时间={dt:.3f}s')
    
    print('✅ 简化训练循环成功！')
    
except Exception as e:
    print(f'❌ 训练循环失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo "4. 测试检查点保存..."
python3 -c "
import torch
import torch_npu
from nanochat.gpt import GPT, GPTConfig
import os

device = 'npu:0'
config = GPTConfig(sequence_len=32, vocab_size=1000, n_layer=2, n_head=4, n_kv_head=4, n_embd=128)

try:
    model = GPT(config)
    model.to(device)
    
    checkpoint = {
        'model': model.state_dict(),
        'config': config,
        'step': 1,
        'val_loss': 3.0
    }
    
    torch.save(checkpoint, 'test_checkpoint.pt')
    print('✅ 检查点保存成功')
    
    # 清理
    if os.path.exists('test_checkpoint.pt'):
        os.remove('test_checkpoint.pt')
        print('✅ 测试文件清理完成')
    
except Exception as e:
    print(f'❌ 检查点测试失败: {e}')
    exit(1)
"

echo "=== 渐进式测试完成 ==="
echo "如果所有测试都通过，问题可能在于："
echo "1. 分布式训练设置"
echo "2. 大批次大小配置" 
echo "3. 长时间运行的内存管理"
echo "4. 特定的训练参数组合"
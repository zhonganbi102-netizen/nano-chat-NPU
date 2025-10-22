#!/bin/bash

echo "=== 直接在项目目录中测试 ==="

# 确保在正确的目录
cd /mnt/linxid615/bza/nanochat-npu

# 只使用一张NPU
export ASCEND_RT_VISIBLE_DEVICES=0

echo "停止现有训练进程..."
pkill -f "python.*base_train"
sleep 5

echo "开始测试NanoChat模型..."
PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

print('导入检查:')
try:
    import torch
    print('✅ torch导入成功')
    
    import torch_npu  
    print(f'✅ torch_npu导入成功, NPU可用: {torch_npu.npu.is_available()}')
    
    from nanochat.gpt import GPT, GPTConfig
    print('✅ nanochat.gpt导入成功')
    
    # 测试模型创建
    config = GPTConfig(sequence_len=128, vocab_size=1000, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)
    model = GPT(config)
    print('✅ 模型创建成功')
    
    # 移动到NPU
    model.to('npu:0')
    print('✅ 模型移动到NPU成功')
    
    # 测试前向传播
    x = torch.randint(0, 1000, (1, 64), device='npu:0')
    with torch.no_grad():
        output = model(x)
    print(f'✅ 前向传播成功，输出形状: {output.shape}')
    
except Exception as e:
    import traceback
    print(f'❌ 错误: {e}')
    traceback.print_exc()
"
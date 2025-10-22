#!/bin/bash

echo "=== MQA配置测试 ==="

echo "测试不同的Multi-Query Attention配置..."

PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

import torch
import torch_npu
from nanochat.gpt import GPT, GPTConfig

# 测试各种MQA配置
test_configs = [
    # (n_head, n_kv_head, 期望结果)
    (4, 4, '✅ 标准多头注意力'),
    (4, 2, '✅ Multi-Query Attention'), 
    (4, 1, '✅ 极简MQA'),
    (6, 6, '✅ 默认配置'),
    (6, 3, '✅ 6头MQA'),
    (6, 2, '✅ 6头MQA-2'),
    (4, 6, '❌ 应该失败: kv_head > head'),
    (5, 2, '❌ 应该失败: head不能被kv_head整除'),
]

device = 'npu:0'

for n_head, n_kv_head, expected in test_configs:
    try:
        config = GPTConfig(
            sequence_len=64, 
            vocab_size=1000, 
            n_layer=1, 
            n_head=n_head, 
            n_kv_head=n_kv_head, 
            n_embd=128
        )
        model = GPT(config)
        model.to(device)
        
        # 简单前向传播测试
        x = torch.randint(0, 1000, (1, 32), device=device)
        with torch.no_grad():
            output = model(x)
        
        print(f'n_head={n_head}, n_kv_head={n_kv_head}: {expected}')
        
    except Exception as e:
        if '❌' in expected:
            print(f'n_head={n_head}, n_kv_head={n_kv_head}: {expected} - 正确失败')
        else:
            print(f'n_head={n_head}, n_kv_head={n_kv_head}: ❌ 意外失败: {e}')

print('\\n=== MQA约束说明 ===')
print('1. n_kv_head <= n_head (KV头数不能超过查询头数)')
print('2. n_head % n_kv_head == 0 (查询头数必须能被KV头数整除)')
print('3. 当 n_kv_head == n_head 时，这是标准的多头注意力')
print('4. 当 n_kv_head < n_head 时，这是Multi-Query Attention')
"

echo "MQA配置测试完成"
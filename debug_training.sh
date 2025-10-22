#!/bin/bash

echo "=== 🔍 NPU训练调试脚本 ==="

# 设置调试环境变量
export ASCEND_GLOBAL_LOG_LEVEL=3  # 详细日志
export ASCEND_SLOG_PRINT_TO_STDOUT=1  # 输出到控制台
export PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH

echo "1. 检查NPU状态..."
python3 -c "
try:
    import torch_npu
    print(f'✅ torch_npu版本: {torch_npu.__version__}')
    print(f'✅ NPU设备数量: {torch_npu.npu.device_count()}')
    for i in range(torch_npu.npu.device_count()):
        print(f'   NPU {i}: {torch_npu.npu.get_device_name(i)}')
except Exception as e:
    print(f'❌ NPU检查失败: {e}')
"

echo "2. 检查数据文件..."
python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

try:
    from nanochat.dataset import list_parquet_files
    files = list_parquet_files()
    print(f'✅ 找到 {len(files)} 个数据文件')
    if len(files) > 0:
        print(f'   第一个文件: {files[0]}')
        print(f'   最后一个文件: {files[-1]}')
    else:
        print('❌ 没有找到数据文件！')
        exit(1)
except Exception as e:
    print(f'❌ 数据检查失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo "3. 测试tokenizer..."
python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

try:
    from nanochat.tokenizer import get_tokenizer
    print('正在加载tokenizer...')
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f'✅ Tokenizer加载成功，词汇表大小: {vocab_size}')
    
    # 测试编码
    test_text = 'Hello world'
    tokens = tokenizer.encode([test_text])
    print(f'✅ 编码测试成功: \"{test_text}\" -> {len(tokens[0])} tokens')
except Exception as e:
    print(f'❌ Tokenizer测试失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo "4. 测试数据加载器（这里通常会卡住）..."
timeout 30 python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

try:
    print('导入必要模块...')
    import torch
    import torch_npu
    from nanochat.dataloader import tokenizing_distributed_data_loader
    from nanochat.common import get_dist_info
    
    print('✅ 模块导入成功')
    
    print('获取分布式信息...')
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    print(f'✅ 分布式设置: rank={ddp_rank}, world_size={ddp_world_size}')
    
    print('创建数据加载器...')
    train_loader = tokenizing_distributed_data_loader(
        B=2,  # 很小的batch size
        T=128,  # 很小的序列长度
        split='train'
    )
    print('✅ 数据加载器创建成功')
    
    print('获取第一批数据...')
    x, y = next(train_loader)
    print(f'✅ 数据加载成功: x.shape={x.shape}, y.shape={y.shape}')
    print(f'   数据设备: {x.device}')
    
except Exception as e:
    print(f'❌ 数据加载器测试失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" || echo "❌ 数据加载器测试超时（30秒）"

echo "5. 如果上面都成功，开始调试版训练..."

if [ $? -eq 0 ]; then
    echo "开始带调试信息的训练..."
    
    python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

print('=== 🚀 开始调试版训练 ===')

try:
    import os
    import time
    import torch
    import torch_npu
    
    print('Step 1: 导入nanochat模块...')
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.dataloader import tokenizing_distributed_data_loader
    from nanochat.common import compute_init, print0
    from nanochat.tokenizer import get_tokenizer
    print('✅ 模块导入完成')
    
    print('Step 2: 计算初始化...')
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    print(f'✅ 计算初始化完成: device={device}')
    
    print('Step 3: 设置参数...')
    depth = 12
    max_seq_len = 256  # 减小序列长度
    device_batch_size = 4  # 减小batch size
    
    print('Step 4: 加载tokenizer...')
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f'✅ Tokenizer加载完成: vocab_size={vocab_size}')
    
    print('Step 5: 创建模型配置...')
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    num_kv_heads = num_heads
    
    model_config_kwargs = dict(
        sequence_len=max_seq_len, 
        vocab_size=vocab_size, 
        n_layer=num_layers, 
        n_head=num_heads, 
        n_kv_head=num_kv_heads, 
        n_embd=model_dim
    )
    print(f'✅ 模型配置: layers={num_layers}, dim={model_dim}, heads={num_heads}')
    
    print('Step 6: 创建模型...')
    with torch.device('meta'):
        model_config = GPTConfig(**model_config_kwargs)
        model = GPT(model_config)
    
    model.to_empty(device=device)
    model.init_weights()
    num_params = sum(p.numel() for p in model.parameters())
    print(f'✅ 模型创建完成: {num_params:,} 参数')
    
    print('Step 7: 创建数据加载器（关键步骤）...')
    train_loader = tokenizing_distributed_data_loader(
        device_batch_size, max_seq_len, split='train'
    )
    print('✅ 数据加载器创建成功')
    
    print('Step 8: 获取第一批数据...')
    x, y = next(train_loader)
    print(f'✅ 第一批数据获取成功: {x.shape}, {y.shape}')
    
    print('Step 9: 测试前向传播...')
    model.train()
    with torch.amp.autocast(device_type='npu', dtype=torch.bfloat16):
        loss = model(x, y)
    print(f'✅ 前向传播成功: loss={loss.item():.4f}')
    
    print('\\n🎉 所有测试通过！现在可以开始正式训练了！')
    
except Exception as e:
    print(f'❌ 调试训练失败: {e}')
    import traceback
    traceback.print_exc()
"
else
    echo "❌ 前面的测试失败，无法继续训练"
    echo ""
    echo "🔍 故障排除建议:"
    echo "1. 检查数据文件是否存在: ls -la base_data/*.parquet"
    echo "2. 检查NPU状态: npu-smi info"
    echo "3. 重启NPU环境: source clean_npu_environment.sh"
    echo "4. 检查内存使用: free -h"
fi
#!/bin/bash

echo "=== 简化调试训练（修改脚本参数）==="

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

echo "创建临时调试训练脚本..."

# 创建临时的调试训练脚本
cat > debug_train_temp.py << 'EOF'
"""
临时调试训练脚本 - 基于base_train.py的简化版本
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
import torch_npu  # NPU支持

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes

print_banner()

# 调试设置（直接修改，避免configurator问题）
run = "debug_npu"
depth = 2  # 小模型
max_seq_len = 256  # 短序列
num_iterations = 3  # 只训练3步
device_batch_size = 2  # 小批次
total_batch_size = 1024  # 小总批次
embedding_lr = 0.01
unembedding_lr = 0.002
weight_decay = 0.01
matrix_lr = 0.01
grad_clip = 1.0
eval_every = 3  # 每3步评估
eval_tokens = 2048  # 少量评估token
core_metric_every = 10  # 跳过core metric
core_metric_max_per_task = 10
sample_every = 10  # 跳过sampling
model_tag = "debug_npu"

print("=== 调试训练设置 ===")
print(f"深度: {depth}, 序列长度: {max_seq_len}")
print(f"批次大小: {device_batch_size}, 总批次: {total_batch_size}")
print(f"训练步数: {num_iterations}")

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
device_type = "npu" if device.type == "npu" else "cuda"
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

print0(f"设备类型: {device_type}, 设备: {device}")

# wandb logging init
use_dummy_wandb = True  # 调试时不用wandb
wandb_run = DummyWandb()

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"词汇表大小: {vocab_size:,}")

# Model kwargs
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads
print0(f"模型层数: {num_layers}, 维度: {model_dim}, 头数: {num_heads}")

# 计算梯度累积
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"梯度累积步数: {grad_accum_steps}")

# 初始化模型
model_config_kwargs = dict(
    sequence_len=max_seq_len, 
    vocab_size=vocab_size, 
    n_layer=num_layers, 
    n_head=num_heads, 
    n_kv_head=num_kv_heads, 
    n_embd=model_dim
)

with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

model.to_empty(device=device)
model.init_weights()
orig_model = model

# NPU兼容性：跳过编译
if device_type == "npu":
    print0("NPU环境：跳过torch.compile")
else:
    model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
print0(f"参数数量: {num_params:,}")

# 初始化优化器
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr, 
    embedding_lr=embedding_lr, 
    matrix_lr=matrix_lr, 
    weight_decay=weight_decay
)
adamw_optimizer, muon_optimizer = optimizers
print0("✅ 优化器初始化成功")

# 数据加载器
base_dir = get_base_dir()
print0(f"Base目录: {base_dir}")

try:
    print0("尝试初始化数据加载器...")
    train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train")
    print0("数据加载器创建成功，尝试获取第一批数据...")
    
    # 使用超时机制避免无限等待
    import signal
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("数据加载超时")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10秒超时
    
    try:
        x, y = next(train_loader)
        signal.alarm(0)  # 取消超时
        print0("✅ 数据加载器成功")
        use_real_data = True
    except TimeoutError:
        signal.alarm(0)
        print0("⚠️  数据加载超时，切换到模拟数据")
        use_real_data = False
    except Exception as e:
        signal.alarm(0)
        print0(f"⚠️  数据加载器异常: {e}")
        use_real_data = False
        
except Exception as e:
    print0(f"⚠️  数据加载器初始化失败: {e}")
    print0("使用模拟数据")
    use_real_data = False

print0("\n=== 开始调试训练 ===")

# 训练循环
for step in range(num_iterations + 1):
    last_step = step == num_iterations
    
    if last_step:
        print0(f"✅ 调试训练完成！总共{num_iterations}步")
        break
    
    print0(f"\n步骤 {step + 1}/{num_iterations}")
    
    # 同步计时
    if device_type == "npu":
        torch_npu.npu.synchronize()
    else:
        torch.cuda.synchronize()
    t0 = time.time()
    
    # 梯度累积
    for micro_step in range(grad_accum_steps):
        if use_real_data:
            try:
                if micro_step == 0:  # 只在第一个micro step获取数据
                    pass  # x, y 已经从之前加载
                else:
                    x, y = next(train_loader)
            except:
                # 如果数据耗尽，使用模拟数据
                x = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
                y = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
        else:
            # 模拟数据
            x = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
            y = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device=device)
        
        with autocast_ctx:
            loss = model(x, y)
        
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        
        if use_real_data and micro_step < grad_accum_steps - 1:
            try:
                x, y = next(train_loader)  # 预取下一批
            except:
                pass
    
    # 梯度裁剪
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
    
    # 优化器步骤
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    
    # 计时
    if device_type == "npu":
        torch_npu.npu.synchronize()
    else:
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    
    # 统计
    tokens_per_sec = int(world_tokens_per_fwdbwd / dt)
    print0(f"  损失: {train_loss.item():.4f}")
    print0(f"  时间: {dt*1000:.1f}ms")
    print0(f"  速度: {tokens_per_sec:,} tokens/sec")
    
    if device_type == "npu":
        mem_mb = torch_npu.npu.memory_allocated(0) / 1024**2
    else:
        mem_mb = torch.cuda.memory_allocated(0) / 1024**2
    print0(f"  内存: {mem_mb:.1f}MB")

# 清理
wandb_run.finish()
compute_cleanup()

print0("\n🎉 调试训练成功完成！")
EOF

echo "运行临时调试训练脚本..."
python3 debug_train_temp.py

echo "清理临时文件..."
rm -f debug_train_temp.py

echo "调试完成"
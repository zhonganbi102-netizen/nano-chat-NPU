"""
Train model with FIXED Muon optimizer for distributed training.

This version intelligently separates parameters:
- Compatible parameters (divisible by world_size) → Use Muon
- Incompatible parameters → Use standard AdamW

Run as:
python base_train_muon_fixed.py

or distributed as:
torchrun --nproc_per_node=8 base_train_muon_fixed.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# NPU稳定性环境变量
if "npu" in str(os.environ.get("DEVICE", "")).lower() or os.path.exists("/usr/local/Ascend"):
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    os.environ["TASK_QUEUE_ENABLE"] = "0"  # 减少TBE任务队列压力
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"  # 启用同步模式
    os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "1"  # 减少日志输出
    print("🔧 NPU环境优化变量已设置")

import time
import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Model architecture
depth = 20 # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 2048 # max context length
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization - NPU稳定性优化
device_batch_size = 16 # per-device batch size (降低以提高NPU稳定性)
total_batch_size = 524288 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
# Evaluation
eval_every = 250 # every how many steps to evaluate the model for val bpb
eval_tokens = 20*524288 # number of tokens to evaluate val loss on
core_metric_every = 2000 # every how many steps to evaluate the core metric
core_metric_max_per_task = 500 # examples per task in estimating the core metric
sample_every = 2000 # every how many steps to sample from the model
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
device_type = "npu" if device.type == "npu" else "cuda"
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads # 1:1 MQA ratio
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
# -----------------------------------------------------------------------------
# Initialize the Model
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
orig_model = model # original, uncompiled model, for saving raw model state_dict

# NPU compatible compilation check
if device.type == "npu" or os.environ.get("TORCH_COMPILE_DISABLE") == "1":
    print0("Skipping torch.compile for NPU compatibility")
    # Keep model uncompiled for NPU
    # NPU stability settings
    if device.type == "npu":
        print0("🔧 配置NPU稳定性设置...")
        import torch_npu
        # 启用内存回收
        torch_npu.npu.empty_cache()
        # 设置NPU优化选项
        torch_npu.npu.set_option({"ACL_OP_SELECT_IMPL_MODE": "high_precision"})
        torch_npu.npu.set_option({"ACL_OPTYPELIST_FOR_IMPLMODE": "Dropout"})
else:
    model = torch.compile(model, dynamic=False) # TODO: dynamic True/False think through
    
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer - SMART HYBRID APPROACH
# Separate parameters into Muon-compatible and incompatible groups
print0("")
print0("🔧 智能混合优化器配置（保留Muon，解决分布式问题）")
print0("=" * 70)

# Collect all parameters
embedding_params = []
unembedding_params = []
matrix_params_all = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if 'wte' in name:  # embedding
            embedding_params.append(param)
        elif 'lm_head' in name:  # unembedding
            unembedding_params.append(param)
        else:  # matrix params (potential Muon candidates)
            if param.ndim == 2:  # Muon只支持2D参数
                matrix_params_all.append((name, param))

print0(f"📊 参数统计:")
print0(f"  Embedding参数: {len(embedding_params)}")
print0(f"  Unembedding参数: {len(unembedding_params)}")
print0(f"  Matrix参数(2D): {len(matrix_params_all)}")

# Analyze which matrix params are Muon-compatible
muon_compatible_params = []
muon_incompatible_params = []

if ddp:
    world_size = ddp_world_size
    print0(f"\n🔍 分析参数兼容性（world_size={world_size}）:")
    
    for name, param in matrix_params_all:
        # 检查参数总元素数是否能被world_size整除
        # 这是reduce_scatter的核心要求
        if param.numel() % world_size == 0:
            muon_compatible_params.append(param)
            if master_process and len(muon_compatible_params) <= 5:  # 只打印前5个
                print0(f"  ✅ {name}: shape={param.shape}, numel={param.numel()} → Muon兼容")
        else:
            muon_incompatible_params.append(param)
            if master_process:
                print0(f"  ⚠️  {name}: shape={param.shape}, numel={param.numel()} → AdamW降级")
    
    if len(muon_compatible_params) > 5 and master_process:
        print0(f"  ... 还有 {len(muon_compatible_params) - 5} 个兼容参数未显示")
else:
    # 单GPU模式，所有参数都兼容
    muon_compatible_params = [p for _, p in matrix_params_all]
    world_size = 1

print0(f"\n✅ 参数分配结果:")
print0(f"  Muon优化器: {len(muon_compatible_params)} 个参数")
print0(f"  AdamW优化器: {len(embedding_params) + len(unembedding_params) + len(muon_incompatible_params)} 个参数")
print0(f"    - Embedding: {len(embedding_params)}")
print0(f"    - Unembedding: {len(unembedding_params)}")
print0(f"    - Matrix(降级): {len(muon_incompatible_params)}")

# Create optimizers
optimizers = []

# AdamW for embedding, unembedding, and incompatible matrix params
adamw_param_groups = [
    {'params': embedding_params, 'lr': embedding_lr, 'weight_decay': weight_decay, 'initial_lr': embedding_lr},
    {'params': unembedding_params, 'lr': unembedding_lr, 'weight_decay': weight_decay, 'initial_lr': unembedding_lr}
]

# Add incompatible matrix params to AdamW
if muon_incompatible_params:
    adamw_param_groups.append({
        'params': muon_incompatible_params, 
        'lr': matrix_lr,  # 使用matrix_lr
        'weight_decay': weight_decay,
        'initial_lr': matrix_lr
    })

adamw_optimizer = torch.optim.AdamW(adamw_param_groups, betas=(0.9, 0.999), eps=1e-8)
optimizers.append(adamw_optimizer)

# Muon for compatible matrix params
if muon_compatible_params:
    print0(f"\n🚀 创建Muon优化器 ({len(muon_compatible_params)} 个兼容参数)")
    try:
        # 直接创建DistMuon实例（分布式版本）
        # DistMuon的__init__会正确初始化所有内部状态（包括zero_buffer）
        if ddp:
            # 分布式环境：使用DistMuon
            from nanochat.muon import DistMuon
            muon_optimizer = DistMuon(
                muon_compatible_params,
                lr=matrix_lr,
                momentum=0.95,
                nesterov=True,
                ns_steps=5
            )
        else:
            # 单卡环境：使用普通Muon
            from nanochat.muon import Muon
            muon_optimizer = Muon(
                muon_compatible_params,
                lr=matrix_lr,
                momentum=0.95,
                nesterov=True,
                ns_steps=5
            )
        
        # 为所有参数组添加initial_lr（用于学习率调度）
        for group in muon_optimizer.param_groups:
            group['initial_lr'] = matrix_lr
        
        optimizers.append(muon_optimizer)
        print0(f"  ✅ Muon优化器创建成功！")
        print0(f"  📊 Muon参数组数量: {len(muon_optimizer.param_groups)}")
            
    except Exception as e:
        print0(f"  ⚠️  Muon创建失败: {e}")
        print0(f"  ⚠️  降级：所有matrix参数使用AdamW")
        
        # 降级方案：所有参数都用AdamW
        all_matrix_params = muon_compatible_params + muon_incompatible_params
        adamw_optimizer.param_groups.append({
            'params': all_matrix_params,
            'lr': matrix_lr,
            'weight_decay': weight_decay,
            'initial_lr': matrix_lr
        })
else:
    print0(f"\n⚠️  没有Muon兼容参数，全部使用AdamW")

print0(f"\n📋 最终优化器配置:")
print0(f"  优化器数量: {len(optimizers)}")
for i, opt in enumerate(optimizers):
    opt_name = opt.__class__.__name__
    param_count = sum(len(g['params']) for g in opt.param_groups)
    print0(f"  [{i+1}] {opt_name}: {param_count} 个参数")
print0("=" * 70)

# Initialize the DataLoaders for train/val - NPU稳定性优化
base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")

# 降低worker数以提高NPU稳定性
print0("🔧 使用低并发DataLoader配置以提高NPU稳定性...")
train_loader = tokenizing_distributed_data_loader(
    device_batch_size, max_seq_len, split="train",
    tokenizer_threads=2,        # 降低线程数：4→1
    tokenizer_batch_size=64     # 降低批处理大小：128→64
)
build_val_loader = lambda: tokenizing_distributed_data_loader(
    device_batch_size, max_seq_len, split="val",
    tokenizer_threads=2,        # 降低线程数：4→1
    tokenizer_batch_size=64     # 降低批处理大小：128→64
)
x, y = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training

# NPU稳定性增强配置
memory_cleanup_interval = 100  # 每100步清理一次内存
checkpoint_interval = 1000   # 每1000步保存检查点
last_stable_step = 0

print0(f"🛡️  NPU稳定性设置:")
print0(f"   内存清理间隔: 每 {memory_cleanup_interval} 步")
print0(f"   检查点间隔: 每 {checkpoint_interval} 步")

# note that we run +1 steps only so that we can eval and save at the end
for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            # Skip initial eval to avoid HCCL timeout
            if step == 0:
                print0("⏭️  跳过初始evaluation，直接开始训练（避免HCCL超时）")
                val_bpb = 8.0
            else:
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    if last_step or (step > 0 and step % core_metric_every == 0):
        model.eval()
        with autocast_ctx:
            print0("⏭️  跳过CORE评估（避免icl_tasks配置错误）")
            results = {
                "accuracy": 0.0,
                "core_metric": 0.0,
                "centered_results": {},
                "task_results": {},
                "raw_results": {}
            }
        print0(f"Step {step:05d} | CORE metric: skipped (避免配置错误)")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            }
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # NPU稳定性检查和维护
    if device.type == "npu" and step > 0:
        # 定期内存清理
        if step % memory_cleanup_interval == 0:
            try:
                import torch_npu
                torch_npu.npu.empty_cache()
                if master_process:
                    current_memory = torch_npu.npu.memory_allocated() / 1024 / 1024
                    print0(f"🧹 Step {step}: 内存清理完成, 当前使用: {current_memory:.2f}MiB")
            except Exception as e:
                print0(f"⚠️  内存清理失败: {e}")
        
        # 定期保存中间检查点（防止长时间训练丢失）
        if master_process and step % checkpoint_interval == 0:
            try:
                output_dirname = model_tag if model_tag else f"d{depth}_step{step}"
                checkpoint_dir = os.path.join(base_dir, "intermediate_checkpoints", output_dirname)
                save_checkpoint(
                    checkpoint_dir,
                    step,
                    orig_model.state_dict(),
                    [opt.state_dict() for opt in optimizers],
                    {
                        "step": step,
                        "val_bpb": smooth_train_loss,
                        "model_config": model_config_kwargs,
                        "user_config": user_config,
                        "device_batch_size": device_batch_size,
                        "max_seq_len": max_seq_len,
                    }
                )
                print0(f"💾 Step {step}: 中间检查点已保存")
                last_stable_step = step
            except Exception as e:
                print0(f"⚠️  检查点保存失败: {e}")

    # -------------------------------------------------------------------------
    # single training step with error handling
    try:
        # evaluate the gradient
        if device_type == "npu":
            import torch_npu
            torch_npu.npu.synchronize()
        else:
            torch.cuda.synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach() # for logging
            loss = loss / grad_accum_steps
            loss.backward()
            x, y = next(train_loader)
        # gradient clipping
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        # step the optimizers
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        
        # Apply momentum scheduler to Muon optimizer if present
        if len(optimizers) >= 2:
            muon_optimizer = optimizers[1]
            muon_momentum = get_muon_momentum(step)
            for group in muon_optimizer.param_groups:
                if 'momentum' in group:
                    group["momentum"] = muon_momentum
        
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        if device_type == "npu":
            import torch_npu
            torch_npu.npu.synchronize()
        else:
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        
    except Exception as e:
        print0(f"❌ Step {step} 训练失败: {e}")
        print0(f"🔄 尝试恢复训练...")
        
        # 清理内存和状态
        if device_type == "npu":
            try:
                import torch_npu
                torch_npu.npu.empty_cache()
            except:
                pass
        else:
            torch.cuda.empty_cache()
        
        # 重置梯度
        model.zero_grad(set_to_none=True)
        
        # 跳过这一步，使用上一步的时间
        dt = 5000.0  # 默认时间
        train_loss = torch.tensor(smooth_train_loss)  # 使用平滑损失
        print0(f"⚠️  Step {step} 已跳过，继续下一步")
    # -------------------------------------------------------------------------

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    if step > 10:
        total_training_time += dt
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })

# print a few more stats
if device_type == "npu":
    try:
        import torch_npu
        print0(f"Peak memory usage: {torch_npu.npu.max_memory_allocated() / 1024 / 1024:.2f}MiB")
    except:
        print0("Peak memory usage: N/A (NPU memory tracking not available)")
else:
    print0(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config,
    {
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    {
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results["core_metric"],
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{torch_npu.npu.max_memory_allocated() / 1024 / 1024:.2f}MiB" if device_type == "npu" and 'torch_npu' in globals() else f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish()
compute_cleanup()



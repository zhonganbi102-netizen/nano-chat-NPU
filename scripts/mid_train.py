"""
Midtrain the model. Same as pretraining but simpler.
Run as:

python -m scripts.mid_train

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
"""

from collections import deque
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# NPU稳定性环境变量 + 内存优化
if "npu" in str(os.environ.get("DEVICE", "")).lower() or os.path.exists("/usr/local/Ascend"):
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    os.environ["TASK_QUEUE_ENABLE"] = "0"  # 减少TBE任务队列压力
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"  # 启用同步模式
    os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "1"  # 减少日志输出
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["NPU_CALCULATE_DEVICE"] = "0,1,2,3,4,5,6,7"
    os.environ["ASCEND_GLOBAL_EVENT_ENABLE"] = "0"  # 减少事件开销
    print("🔧 NPU环境优化变量已设置（含内存优化）")

import time
import wandb
import torch
import gc

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.checkpoint_manager import load_model
import torch.distributed as dist

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
run = "mid_train_8npu" # wandb run name default 
model_tag = "d18" # 加载d18模型 (350M参数)
step = 13351 # 加载完整训练的step 13351检查点
dtype = "bfloat16"
max_seq_len = 2048
# NPU内存优化配置（8NPU分布式）
device_batch_size = 8  # 内存优化：8NPU × 8 = 64 total
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
init_lr_frac = 1.0 # initial learning rate is this fraction of the base learning rate
weight_decay = 0.0
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
eval_every = 150
eval_tokens = 20*524288
total_batch_size = 262144  # 8NPU优化：8×8×2048×2 = 262144
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
device_type = "npu" if device.type == "npu" else "cuda"
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-mid", name=run, config=user_config)

# Load the model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=model_tag, step=step)
pretrain_batch_size = meta.get("device_batch_size", None)
if pretrain_batch_size is not None and device_batch_size > pretrain_batch_size:
    print0(f"FOOTGUN WARNING: base model training used device_batch_size {pretrain_batch_size}, did you pass in a good --device_batch_size to this script?")
orig_model = model

# NPU compatible compilation check
if device.type == "npu" or os.environ.get("TORCH_COMPILE_DISABLE") == "1":
    print0("Skipping torch.compile for NPU compatibility")
    # Keep model uncompiled for NPU
    if device.type == "npu":
        print0("🔧 配置NPU稳定性设置...")
        import torch_npu
        # 启用内存回收
        torch_npu.npu.empty_cache()
        # 设置NPU优化选项
        torch_npu.npu.set_option({"ACL_OP_SELECT_IMPL_MODE": "high_precision"})
        torch_npu.npu.set_option({"ACL_OPTYPELIST_FOR_IMPLMODE": "Dropout"})
else:
    model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer - SMART HYBRID APPROACH (与base_train一致)
# Separate parameters into Muon-compatible and incompatible groups
print0("")
print0("🔧 智能混合优化器配置（保留Muon，解决分布式问题）")
print0("=" * 70)

# Collect all parameters
embedding_params = []
unembedding_params = []
matrix_params_all = []

for name, param in orig_model.named_parameters():
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
        if param.numel() % world_size == 0:
            muon_compatible_params.append(param)
            if master_process and len(muon_compatible_params) <= 5:
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

# Create optimizers
optimizers = []

# AdamW for embedding, unembedding, and incompatible matrix params
adamw_param_groups = [
    {'params': embedding_params, 'lr': embedding_lr * init_lr_frac, 'weight_decay': weight_decay, 'initial_lr': embedding_lr * init_lr_frac},
    {'params': unembedding_params, 'lr': unembedding_lr * init_lr_frac, 'weight_decay': weight_decay, 'initial_lr': unembedding_lr * init_lr_frac}
]

# Add incompatible matrix params to AdamW
if muon_incompatible_params:
    adamw_param_groups.append({
        'params': muon_incompatible_params, 
        'lr': matrix_lr * init_lr_frac,
        'weight_decay': weight_decay,
        'initial_lr': matrix_lr * init_lr_frac
    })

adamw_optimizer = torch.optim.AdamW(adamw_param_groups, betas=(0.9, 0.999), eps=1e-8)
optimizers.append(adamw_optimizer)

# Muon for compatible matrix params
if muon_compatible_params:
    print0(f"\n🚀 创建Muon优化器 ({len(muon_compatible_params)} 个兼容参数)")
    try:
        if ddp:
            # 分布式环境：使用DistMuon
            from nanochat.muon import DistMuon
            muon_optimizer = DistMuon(
                muon_compatible_params,
                lr=matrix_lr * init_lr_frac,
                momentum=0.95,
                nesterov=True,
                ns_steps=5
            )
        else:
            # 单卡环境：使用普通Muon
            from nanochat.muon import Muon
            muon_optimizer = Muon(
                muon_compatible_params,
                lr=matrix_lr * init_lr_frac,
                momentum=0.95,
                nesterov=True,
                ns_steps=5
            )
        
        # 为所有参数组添加initial_lr
        for group in muon_optimizer.param_groups:
            group['initial_lr'] = matrix_lr * init_lr_frac
        
        optimizers.append(muon_optimizer)
        print0(f"  ✅ Muon优化器创建成功！")
            
    except Exception as e:
        print0(f"  ⚠️  Muon创建失败: {e}")
        print0(f"  ⚠️  降级：所有matrix参数使用AdamW")
        
        # 降级方案：所有参数都用AdamW
        all_matrix_params = muon_compatible_params + muon_incompatible_params
        adamw_optimizer.param_groups.append({
            'params': all_matrix_params,
            'lr': matrix_lr * init_lr_frac,
            'weight_decay': weight_decay,
            'initial_lr': matrix_lr * init_lr_frac
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

# Midtraining data mixture and DataLoader
base_dir = get_base_dir()
train_dataset = TaskMixture([
    SmolTalk(split="train"), # 460K rows of general conversations
    MMLU(subset="auxiliary_train", split="train"), # 100K rows of multiple choice problems drawn from ARC, MC_TEST, OBQA, RACE
    GSM8K(subset="main", split="train"), # 8K rows teaching simple math and (calculator) tool use
]) # total: 460K + 100K + 8K = 568K rows
val_dataset = TaskMixture([
    SmolTalk(split="test"), # 24K rows in test set
    MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
    GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
]) # total: 24K + 14K + 1.32K ~= 39K rows
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = False # we will toggle this to True when we reach the end of the dataset
approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
def mid_data_generator(split):
    global last_step, approx_progress
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    needed_tokens = device_batch_size * max_seq_len + 1 # to form one training batch of inputs,targets
    token_buffer = deque()
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    cursor = ddp_rank # increments by ddp_world_size each time, so each rank processes unique documents
    while True:
        # Accumulate enough tokens for one iteration before yielding
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            token_buffer.extend(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor -= dataset_size # wrap around for another epoch
                if split == "train":
                    last_step = True # toggle last_step to True, which will terminate the training loop
        # Build up inputs/targets and yield
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
        if split == "train":
            approx_progress = cursor / dataset_size # approximate progress as a fraction of the dataset
        yield inputs, targets

train_loader = mid_data_generator("train")
build_val_loader = lambda: mid_data_generator("val")
progress = 0 # will go from 0 to 1 over the course of the epoch

# Learning rate scheduler
def get_lr_multiplier(progress):
    return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# NPU稳定性增强配置（内存优化）
memory_cleanup_interval = 20   # 紧急：每20步清理内存
checkpoint_interval = 500    # 更频繁检查点
gc_interval = 10  # 每10步强制Python垃圾回收

print0(f"🛡️  NPU内存优化设置:")
print0(f"   batch_size: {device_batch_size} (8NPU优化)")
print0(f"   内存清理间隔: 每 {memory_cleanup_interval} 步")
print0(f"   垃圾回收间隔: 每 {gc_interval} 步")

# -----------------------------------------------------------------------------
# Training loop
x, y = next(train_loader) # prefetch the very first batch of data
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
step = 0
while True:
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
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

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step:
        output_dirname = f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers], # TODO: make sure saving across ranks is done correctly
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": {
                    "sequence_len": max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                },
                "user_config": user_config, # inputs to the training script
            }
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
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
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        progress = max(progress, approx_progress) # only increase progress monotonically
    # step the optimizers
    lrm = get_lr_multiplier(progress)
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
    
    # NPU内存优化和维护（紧急模式）
    if device_type == "npu" and step > 0:
        # 🔥 每步都进行轻度内存清理（紧急模式）
        if step % gc_interval == 0:
            try:
                # Python垃圾回收
                gc.collect()
                # NPU缓存清理
                import torch_npu
                torch_npu.npu.empty_cache()
                if master_process and step % (gc_interval * 2) == 0:  # 减少日志频率
                    current_memory = torch_npu.npu.memory_allocated() / 1024 / 1024
                    reserved_memory = torch_npu.npu.memory_reserved() / 1024 / 1024
                    print0(f"🧹 Step {step}: 内存清理 - 使用: {current_memory:.0f}MB, 保留: {reserved_memory:.0f}MB")
            except Exception as e:
                print0(f"⚠️  内存清理失败: {e}")
        
        # 深度内存清理
        if step % memory_cleanup_interval == 0:
            try:
                import torch_npu
                # 强制同步和清理
                torch_npu.npu.synchronize()
                torch_npu.npu.empty_cache()
                # 强制Python垃圾回收
                for i in range(3):  # 多次回收
                    gc.collect()
                if master_process:
                    current_memory = torch_npu.npu.memory_allocated() / 1024 / 1024
                    print0(f"🔥 Step {step}: 深度内存清理完成, 当前使用: {current_memory:.2f}MiB")
            except Exception as e:
                print0(f"⚠️  深度内存清理失败: {e}")
    if device_type == "npu":
        import torch_npu
        torch_npu.npu.synchronize()
    else:
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
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
get_report().log(section="Midtraining", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of iterations": step,
        "DDP world size": ddp_world_size,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()

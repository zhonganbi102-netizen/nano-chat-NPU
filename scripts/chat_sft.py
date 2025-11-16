"""
Finetune a base model to be a chat model.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
"""

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

import copy
import gc

import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb
from nanochat.checkpoint_manager import load_model
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture, TaskSequence
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# SFT Hyperparameters
run = "chat_sft_8npu" # wandb run name default 
# input model options
source = "mid" # base|mid , which checkpoint to load the model from (base model or midtrained model)
model_tag = "d18" # 加载d18模型 (350M参数)
step = 1532 # 加载mid_train完成的step 1532检查点
# compute/precision
dtype = "bfloat16"
# NPU内存优化配置（8NPU分布式）
device_batch_size = 4  # 内存优化：8NPU × 4 = 32 total
# optimization
num_epochs = 1
max_iterations = -1 # override number of iterations (-1 = use num_epochs * num_iterations)
target_examples_per_step = 64  # 8NPU优化：8×4×2 = 64
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
# evaluation and logging there of
eval_every = 100
eval_steps = 100
eval_metrics_every = 200
# now allow CLI to override the settings via the configurator lol
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
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=run, config=user_config, save_code=True)

# Load the model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
orig_model = model # original, uncompiled model

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
    # model = torch.compile(model, dynamic=True) # doesn't work super well because of variable lengths of inputs
    pass

engine = Engine(model, tokenizer) # will be used for inline model evaluation only

# -----------------------------------------------------------------------------
# Task data mixture we'll train on

train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"), # 2.3K rows
    ARC(subset="ARC-Challenge", split="train"), # 1.1K rows
    GSM8K(subset="main", split="train"), # 8K rows
    SmolTalk(split="train", stop=10_000), # 10K rows of smoltalk
]) # 2.3K + 1.1K + 8K + 10K = 21.4K rows
val_ds = SmolTalk(split="test") # general conversations, 24K rows (though we don't actually use all of it)

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>") # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss
    # prepares a list of tokenized conversations into a batch and yields
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1 # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long) # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            # recall -1 is the ignore index, so mask out targets where mask is 0
            row_targets = ids_tensor[1:]
            # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1 # mask out targets where mask is 0
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device) # move to device
        targets = targets.to(device)
        return inputs, targets
    # iterates over the dataset in epochs, tokenizes
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
if max_iterations >= 0 and num_iterations > max_iterations:
    print0(f"Number of iterations is too high: {num_iterations}, capping to {max_iterations}")
    num_iterations = max_iterations
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer - SMART HYBRID APPROACH (与base_train/mid_train一致)
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

# NPU稳定性增强配置（内存优化）
memory_cleanup_interval = 20   # 紧急：每20步清理内存
gc_interval = 10  # 每10步强制Python垃圾回收

print0(f"🛡️  NPU内存优化设置:")
print0(f"   batch_size: {device_batch_size} (8NPU优化)")
print0(f"   内存清理间隔: 每 {memory_cleanup_interval} 步")
print0(f"   垃圾回收间隔: 每 {gc_interval} 步")

# -----------------------------------------------------------------------------
# Training loop

# Learning rate scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# Go!
step = 0
train_iter = iter(train_loader)
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    # evaluate the validation loss
    if last_step or step % eval_every == 0:
        model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean() # average over eval_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) # average over ranks
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        model.train()

    # evlauate MMLU accuracy
    if last_step or (step > 0 and step % eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            # note that because these are inside no_grad, we can usually afford to at least ~2X the batch size
            metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=1024)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=1024)
            metrics["gsm8k_acc"] = run_chat_eval("GSM8K", model, tokenizer, engine, max_problems=64)
            metrics["humaneval_acc"] = run_chat_eval("HumanEval", model, tokenizer, engine, max_problems=64)
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            "step": step,
            **metrics,
        })
        model.train()

    if last_step:
        break

    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device) # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward() # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # sum over ranks

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # Apply momentum scheduler to Muon optimizer if present
    if len(optimizers) >= 2:
        muon_optimizer = optimizers[1]
        # Simple momentum for SFT (less aggressive than pre-training)
        muon_momentum = 0.95
        for group in muon_optimizer.param_groups:
            if 'momentum' in group:
                group["momentum"] = muon_momentum

    # step the optimizers
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

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    })
    step += 1

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag = f"d{depth}" # base the model tag on the depth of the base model
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
    model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None, # note: we don't bother to save the optimizer state
        {
            "step": step,
            "val_loss": val_loss,
            **metrics,
            "model_config": model_config_kwargs,
        }
    )
    print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat SFT", data=[
    user_config, # CLI args
    {
        "Training rows": len(train_ds),
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()

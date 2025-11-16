"""
Reinforcement learning on GSM8K via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use GAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import os
import itertools
import re
import wandb
import torch
import torch.distributed as dist

# NPU兼容性配置（torch导入后）
try:
    if torch.cuda.is_available() and hasattr(torch.cuda, 'current_device'):
        # GPU环境
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        # NPU环境
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:512"
except:
    # 默认NPU配置
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:512"

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# RL hyperparameters - 8NPU配置
run = "npu_chat_rl_8npu" # wandb run name
source = "sft" # mid|sft - 使用已完成的SFT模型
dtype = "bfloat16"
device_batch_size = 4 # 8NPU配置：每个NPU处理4个样本，避免OOM
examples_per_step = 32 # 8NPU总计：8 * 4 = 32 examples per step
num_samples = 16 # number of samples per example (/question)
max_new_tokens = 256
temperature = 1.0
top_k = 50 # TODO: try None?
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05
num_epochs = 1 # how many epochs of gsm8k to train on
save_every = 60 # every how many steps to save the model
eval_every = 60 # every how many steps to evaluate the model for val pass@k
eval_examples = 400 # number of examples used for evaluating pass@k
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Init compute/precision
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
device_type = "npu" if device.type == "npu" else "cuda"
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=run, config=user_config)

# Init model and tokenizer from SFT checkpoint
print0(f"🔄 正在从source='{source}'加载SFT模型...")
model, tokenizer, meta = load_model(source, device, phase="eval")
print0(f"✅ SFT模型加载成功:")
print0(f"  - 模型深度: {model.config.n_layer} layers")
print0(f"  - 模型维度: {model.config.n_embd}")
print0(f"  - 参数量: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print0(f"  - 设备类型: {device.type}")

engine = Engine(model, tokenizer) # for sampling rollouts
print0(f"✅ Engine初始化完成")

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // examples_per_step) * num_epochs
print0(f"Calculated number of steps: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # ok to use this token, it's only for padding and isn't used in the loss.
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # each rank is responsible for different examples in the training data
    for example_idx in itertools.cycle(rank_indices):

        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval() # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed, # must make sure to change the seed for each sampling step
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        mu = rewards.mean()
        advantages = rewards - mu
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= device_batch_size # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer - Use standard PyTorch AdamW for NPU compatibility

print0("🔧 使用Muon分布式优化器（NPU分布式兼容，避免reduce_scatter错误）")

# 智能参数分组（参考base_train_muon_fixed.py）
embedding_params = []
unembedding_params = []
matrix_params_all = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if 'wte' in name:
            embedding_params.append(param)
        elif 'lm_head' in name:
            unembedding_params.append(param)
        else:
            matrix_params_all.append((name, param))

print0(f"📊 参数统计:")
print0(f"  Embedding参数: {len(embedding_params)}")
print0(f"  Unembedding参数: {len(unembedding_params)}")
print0(f"  Matrix参数(2D): {len(matrix_params_all)}")

# 分析哪些matrix参数是Muon兼容的
muon_compatible_params = []
muon_incompatible_params = []
if ddp:
    world_size = ddp_world_size
    print0(f"\n🔍 分析参数兼容性（world_size={world_size}）:")
    for name, param in matrix_params_all:
        if param.numel() % world_size == 0:
            muon_compatible_params.append(param)
        else:
            muon_incompatible_params.append(param)
    if len(muon_compatible_params) > 5 and master_process:
        print0(f"  ... 还有 {len(muon_compatible_params) - 5} 个兼容参数未显示")
else:
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
            # 分布式环境：使用DistMuon（参考chat_sft.py）
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
        print0(f"  📊 Muon参数组数量: {len(muon_optimizer.param_groups)}")
            
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

# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {examples_per_step * num_samples}") # total batch size in sequences/step
assert examples_per_step % ddp_world_size == 0, f"Desired examples per step ({examples_per_step}) must be divisible by the number of ranks ({ddp_world_size})"
examples_per_rank = examples_per_step // ddp_world_size # per NPU
print0(f"Calculated examples per rank: {examples_per_rank}")
print0(f"8NPU配置验证: {ddp_world_size} NPUs × {examples_per_rank} examples/NPU = {examples_per_step} total examples")

# Kick off the training loop
batch_iterator = get_batch()
for step in range(num_steps):

    # Evaluate the model once in a while and log to wandb
    if step % eval_every == 0:
        model.eval()
        passk = torch.zeros(device_batch_size, device=device) # pass@k for k=1..device_batch_size
        with autocast_ctx:
            records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=device_batch_size, max_examples=eval_examples, temperature=1.0)
            records = list(records_iter) # collect all records
        for k in range(1, device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # normalize by the total number of records
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        # Evaluate the loss and gradients
        model.train() # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % device_batch_size == 0
        num_passes = inputs_all.size(0) // device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
            loss.backward()
            progress_percent = (step / num_steps) * 100
            print0(f"Step {step}/{num_steps} ({progress_percent:.1f}%) | Example {example_step}/{examples_per_rank} | Pass {pass_idx}/{num_passes} | loss: {loss.item():.6f} | reward: {rewards.mean().item():.4f}")
        # For logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)
        
        # NPU内存清理（每处理完一个example后）
        if device_type == "npu" and example_step % 5 == 0:  # 每5个examples清理一次
            try:
                import torch_npu
                torch_npu.npu.empty_cache()
            except:
                pass

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp: # aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })
    
    # 定期显示训练进度
    if step % 20 == 0 or step < 10:
        progress_pct = (step / num_steps) * 100
        print0(f"🚀 训练进度: {step}/{num_steps} ({progress_pct:.1f}%) | 奖励: {mean_reward:.3f} | 序列长度: {mean_sequence_length:.1f}")

    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for opt in optimizers: # first set the learning rate
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # Apply momentum scheduler to Muon optimizer if present (参考chat_sft.py)
    if len(optimizers) >= 2:
        muon_optimizer = optimizers[1]
        # Simple momentum for RL (类似SFT)
        muon_momentum = 0.95
        for group in muon_optimizer.param_groups:
            if 'momentum' in group:
                group["momentum"] = muon_momentum

    for opt in optimizers: # then step the optimizers
        opt.step()
    model.zero_grad(set_to_none=True)
    
    # NPU同步（确保优化器步骤完成）
    if device_type == "npu":
        try:
            import torch_npu
            torch_npu.npu.synchronize()
        except ImportError:
            # torch_npu未安装，跳过同步
            pass
        except Exception as e:
            print0(f"⚠️ NPU同步警告: {e}")
            pass
    
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}" # base the model tag on the depth of the base model (d18)
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
        print0(f"💾 保存RL模型到: {checkpoint_dir} (模型: {model_tag})")
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[
    user_config, # CLI args
])

wandb_run.finish() # wandb run finish
compute_cleanup()

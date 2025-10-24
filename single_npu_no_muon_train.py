"""
NPU兼容的GPT训练脚本 - 不使用Muon优化器
NPU compatible GPT training script - without Muon optimizer
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import wandb
import torch

# 添加当前目录到路径
sys.path.append('.')

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model

print_banner()

def setup_adamw_only_optimizers(model, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
    """
    设置只使用AdamW的优化器，避免Muon在NPU上的兼容性问题
    """
    from nanochat.common import get_dist_info
    from nanochat.adamw import DistAdamW
    from functools import partial
    
    model_dim = model.config.n_embd
    ddp, rank, local_rank, world_size = get_dist_info()
    
    # 获取所有参数
    matrix_params = list(model.transformer.h.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    
    # 学习率缩放
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    if rank == 0:
        print(f"[无Muon模式] 缩放AdamW学习率 ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
    
    # 创建AdamW优化器组，包含所有参数
    adam_groups = [
        dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        dict(params=matrix_params, lr=matrix_lr * dmodel_lr_scale),  # 矩阵参数也用AdamW
    ]
    
    adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
    
    # NPU兼容的AdamW
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            # NPU环境使用标准AdamW
            if ddp:
                AdamWFactory = DistAdamW
            else:
                AdamWFactory = torch.optim.AdamW
            print0("[无Muon模式] 使用NPU兼容的AdamW优化器")
        else:
            AdamWFactory = partial(torch.optim.AdamW, fused=True)
    except ImportError:
        AdamWFactory = partial(torch.optim.AdamW, fused=True)
    
    adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
    
    # 只返回AdamW优化器
    optimizers = [adamw_optimizer]
    
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    print0(f"[无Muon模式] 优化器设置完成，参数组数: {len(adam_groups)}")
    return optimizers

def main():
    """
    主训练函数 - 单NPU无Muon版本
    """
    # 检查NPU环境
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            torch_npu.npu.set_device(0)
            print(f"✅ 使用NPU设备: npu:{torch_npu.npu.current_device()}")
        else:
            print("⚠️  NPU不可用，使用CPU")
    except ImportError:
        print("⚠️  torch_npu未安装，使用CPU")
    
    # 训练配置（保守参数）
    config = {
        'run': 'single_npu_no_muon',
        'depth': 6,
        'device_batch_size': 4,
        'total_batch_size': 8192,
        'num_iterations': 500,
        'embedding_lr': 0.001,
        'unembedding_lr': 0.0001,
        'matrix_lr': 0.0005,
        'grad_clip': 1.0,
        'eval_every': 100,
        'sample_every': 500,
        'core_metric_every': 999999,
    }
    
    print("🚀 开始单NPU训练（无Muon优化器）...")
    print(f"配置: {config}")
    
    # 初始化分布式（单GPU模式）
    compute_init()
    
    # 创建模型
    model_config = GPTConfig(
        vocab_size=265,  # 简化词汇表
        seq_len=2048,
        depth=config['depth'],
        model_dim=768,
        num_heads=6,
        num_kv_heads=6
    )
    
    model = GPT(model_config)
    print0(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 移动到设备
    device = model.get_device()
    print0(f"模型设备: {device}")
    
    # 设置优化器（只使用AdamW）
    optimizers = setup_adamw_only_optimizers(
        model,
        unembedding_lr=config['unembedding_lr'],
        embedding_lr=config['embedding_lr'],
        matrix_lr=config['matrix_lr'],
        weight_decay=0.0
    )
    
    print0("✅ 优化器设置完成")
    
    # 获取tokenizer
    tokenizer = get_tokenizer()
    
    # 创建数据加载器
    loader = tokenizing_distributed_data_loader(
        files="fineweb/*.parquet",
        text_key="text",
        num_workers=1,
        num_epochs=1,
        seed=42,
        verbose=True,
        shuffle=True,
        batch_size=config['device_batch_size'],
        seq_len=model_config.seq_len,
        header_len=1,
        tokenizer=tokenizer,
        rank=0,
        world_size=1
    )
    
    print0("✅ 数据加载器创建完成")
    
    # 训练循环
    for step in range(config['num_iterations']):
        model.train()
        
        # 获取批次数据
        batch = next(loader)
        x, y = batch["input_ids"], batch["labels"]
        
        # 前向传播
        outputs = model(x, targets=y)
        loss = outputs['loss']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if config['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        # 优化器步骤（只有一个AdamW优化器）
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
        
        # 打印进度
        if step % 10 == 0:
            print0(f"步骤 {step}/{config['num_iterations']}, 损失: {loss.item():.4f}")
        
        # 评估
        if step > 0 and step % config['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                print0(f"评估步骤 {step}...")
    
    print0("🎉 训练完成！")
    compute_cleanup()

if __name__ == "__main__":
    main()
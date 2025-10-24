"""
NPU兼容的setup_optimizers方法修改
为GPT类添加NPU友好的优化器设置选项
"""

def setup_optimizers_npu_compatible(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, use_muon=False):
    """
    设置优化器，支持NPU兼容模式
    
    Args:
        unembedding_lr: unembedding层学习率
        embedding_lr: embedding层学习率  
        matrix_lr: 矩阵层学习率
        weight_decay: 权重衰减
        use_muon: 是否使用Muon优化器（NPU设为False）
    
    Returns:
        优化器列表
    """
    from nanochat.common import get_dist_info
    from nanochat.adamw import DistAdamW
    from functools import partial
    
    ddp, rank, local_rank, world_size = get_dist_info()
    dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
    
    if rank == 0:
        print(f"Learning rate scale ∝1/√({self.config.n_embd}/768) = {dmodel_lr_scale:.6f}")
    
    # 获取参数组
    matrix_params = list(self.transformer.h.parameters())
    embedding_params = list(self.transformer.wte.parameters())
    lm_head_params = list(self.lm_head.parameters())
    
    if use_muon:
        # 原始Muon模式（GPU/CPU）
        print(f"[原始模式] 使用AdamW + Muon优化器")
        from nanochat.muon import Muon, DistMuon
        
        # AdamW for embeddings and unembedding
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        
        # 选择AdamW工厂
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                AdamWFactory = DistAdamW if ddp else torch.optim.AdamW
            else:
                AdamWFactory = partial(torch.optim.AdamW, fused=True)
        except ImportError:
            AdamWFactory = partial(torch.optim.AdamW, fused=True)
        
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        # Muon for matrix parameters
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        
        optimizers = [adamw_optimizer, muon_optimizer]
        
    else:
        # NPU兼容模式（只使用AdamW）
        print(f"[NPU兼容模式] 只使用AdamW优化器，避免Muon")
        
        # 所有参数都用AdamW
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
                AdamWFactory = DistAdamW if ddp else torch.optim.AdamW
                print(f"[NPU兼容] 使用torch_npu兼容的AdamW")
            else:
                AdamWFactory = partial(torch.optim.AdamW, fused=True)
        except ImportError:
            AdamWFactory = partial(torch.optim.AdamW, fused=True)
        
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        optimizers = [adamw_optimizer]
    
    # 设置初始学习率
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    print(f"✅ 优化器设置完成: {len(optimizers)} 个优化器, use_muon={use_muon}")
    return optimizers

# 使用示例：
# 在NPU环境中：
# optimizers = model.setup_optimizers_npu_compatible(use_muon=False)
# 在GPU/CPU环境中：  
# optimizers = model.setup_optimizers_npu_compatible(use_muon=True)
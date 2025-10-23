#!/usr/bin/env python3
"""
NPU与Muon优化器兼容性补丁
通过修改GPT类的setup_optimizers方法来避免使用Muon优化器
"""

import os
import sys
import torch
from nanochat.gpt import GPT

# 保存原始的setup_optimizers方法
original_setup_optimizers = GPT.setup_optimizers

def patched_setup_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    """
    为NPU环境打补丁的优化器设置方法，全部使用AdamW替代Muon
    """
    # 检查是否为NPU环境
    is_npu_env = torch.device(torch.cuda.current_device()).type == "npu" or os.environ.get("NPU_USE_ADAM_ONLY") == "1"
    
    if not is_npu_env:
        # 非NPU环境，使用原始方法
        return original_setup_optimizers(self, unembedding_lr, embedding_lr, matrix_lr, weight_decay)
    
    print("应用NPU优化器兼容性补丁: 全部使用AdamW")
    
    # 将所有参数分组为两类
    params_dict = {n: p for n, p in self.named_parameters()}
    
    # 第一组：词嵌入和输出层权重
    embedding_params = []
    embedding_params.extend([params_dict['emb_tok.weight']])
    embedding_names = ['emb_tok.weight']
    
    # 第二组：其他所有参数（原本使用Muon的参数）
    matrix_params = []
    matrix_names = []
    for n, p in params_dict.items():
        if n not in embedding_names and n != 'lm_head.weight':
            matrix_params.append(p)
            matrix_names.append(n)
    
    # 第三组：lm_head权重
    if 'lm_head.weight' in params_dict:
        unembedding_params = [params_dict['lm_head.weight']]
    else:
        unembedding_params = []
    
    # 创建优化器，全部使用AdamW
    adamw_embedding = torch.optim.AdamW(
        [{'params': embedding_params, 'lr': embedding_lr, 'initial_lr': embedding_lr}],
        lr=embedding_lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    
    adamw_matrix = torch.optim.AdamW(
        [{'params': matrix_params, 'lr': matrix_lr, 'initial_lr': matrix_lr}],
        lr=matrix_lr, weight_decay=0.0, betas=(0.9, 0.95)
    )
    
    adamw_unembedding = torch.optim.AdamW(
        [{'params': unembedding_params, 'lr': unembedding_lr, 'initial_lr': unembedding_lr}],
        lr=unembedding_lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    
    return [adamw_embedding, adamw_matrix, adamw_unembedding]

# 应用补丁
def apply_patch():
    GPT.setup_optimizers = patched_setup_optimizers
    print("✅ NPU优化器兼容性补丁已应用")
    
if __name__ == "__main__":
    apply_patch()
    print("运行此脚本来应用NPU优化器补丁")
    print("在训练脚本中添加: import npu_adam_patch; npu_adam_patch.apply_patch()")

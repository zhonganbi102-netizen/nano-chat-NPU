"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
Fixed for NPU distributed training compatibility.
"""
import torch
import torch.distributed as dist
from torch import Tensor


class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    Fixed tensor size alignment for reduce_scatter operations.
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()  # 移除torch.compile，在NPU上可能有问题
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # 检查是否是分布式环境
        if world_size == 1:
            # 单GPU情况，使用标准AdamW
            self._single_gpu_step()
            return
            
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for param in params:
                if param.grad is None:
                    continue
                    
                grad = param.grad
                # 确保grad的第0维能被world_size整除
                if grad.shape[0] % world_size != 0:
                    # 如果不能整除，使用all_reduce而不是reduce_scatter
                    all_reduce_futures.append(dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                    grad_slices.append(grad)
                else:
                    rank_size = grad.shape[0] // world_size
                    grad_slice = torch.empty(rank_size, *grad.shape[1:], dtype=grad.dtype, device=grad.device)
                    reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                    grad_slices.append(grad_slice)

        # 等待所有通信完成
        if reduce_scatter_futures:
            torch.futures.collect_all(reduce_scatter_futures).wait()
        if all_reduce_futures:
            torch.futures.collect_all(all_reduce_futures).wait()

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            
            for param in params:
                if param.grad is None:
                    continue
                    
                lr = group['lr'] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]
                g = grad_slices[idx]
                
                # 确定工作的参数切片
                if param.shape[0] % world_size != 0:
                    # 全参数更新（all_reduce情况）
                    p_slice = param
                    g_slice = g
                else:
                    # 分片更新（reduce_scatter情况）
                    rank_size = param.shape[0] // world_size
                    p_slice = param[rank * rank_size:(rank + 1) * rank_size]
                    g_slice = g
                
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=param.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                
                idx += 1
                
                # 如果使用了分片，需要同步回完整参数
                if param.shape[0] % world_size == 0:
                    all_reduce_futures.append(dist.all_gather_into_tensor(param, p_slice, async_op=True).get_future())
        
        # 等待参数同步完成
        if all_reduce_futures:
            torch.futures.collect_all(all_reduce_futures).wait()

    def _single_gpu_step(self):
        """单GPU时的标准AdamW实现"""
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            
            for param in params:
                if param.grad is None:
                    continue
                    
                lr = group['lr'] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]
                grad = param.grad
                
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=param.device)
                    state['exp_avg'] = torch.zeros_like(param)
                    state['exp_avg_sq'] = torch.zeros_like(param)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    param.mul_(1 - eff_weight_decay)
                
                # update running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                param.add_(other=update, alpha=-1.0)
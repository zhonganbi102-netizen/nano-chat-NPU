#!/usr/bin/env python3
"""
NanoChat NPU批次大小计算器
用于验证训练参数配置是否正确
"""

def calculate_batch_config(device_batch_size, max_seq_len, total_batch_size, world_size):
    """
    计算批次配置参数
    """
    tokens_per_fwdbwd = device_batch_size * max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * world_size
    
    print(f"配置参数:")
    print(f"  device_batch_size: {device_batch_size}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  total_batch_size: {total_batch_size:,}")
    print(f"  world_size: {world_size}")
    print()
    
    print(f"计算结果:")
    print(f"  tokens_per_fwdbwd (单设备): {tokens_per_fwdbwd:,}")
    print(f"  world_tokens_per_fwdbwd (所有设备): {world_tokens_per_fwdbwd:,}")
    
    if total_batch_size % world_tokens_per_fwdbwd == 0:
        grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
        print(f"  ✅ 配置有效!")
        print(f"  gradient_accumulation_steps: {grad_accum_steps}")
        
        # 计算内存使用估计
        model_size_gb = estimate_model_memory(12)  # depth=12
        batch_memory_gb = estimate_batch_memory(device_batch_size, max_seq_len, 12)
        total_memory_gb = model_size_gb + batch_memory_gb
        
        print(f"  预估内存使用:")
        print(f"    模型参数: ~{model_size_gb:.1f} GB")
        print(f"    批次数据: ~{batch_memory_gb:.1f} GB")
        print(f"    总计: ~{total_memory_gb:.1f} GB/NPU")
        
        return True, grad_accum_steps
    else:
        remainder = total_batch_size % world_tokens_per_fwdbwd
        print(f"  ❌ 配置无效!")
        print(f"  余数: {remainder:,}")
        print(f"  建议调整:")
        
        # 建议新的device_batch_size
        suggested_sizes = []
        for size in [8, 12, 16, 20, 24, 32]:
            test_world_tokens = size * max_seq_len * world_size
            if total_batch_size % test_world_tokens == 0:
                grad_steps = total_batch_size // test_world_tokens
                suggested_sizes.append((size, grad_steps))
        
        if suggested_sizes:
            print(f"  可用的device_batch_size选项:")
            for size, grad_steps in suggested_sizes:
                print(f"    device_batch_size={size} => grad_accum_steps={grad_steps}")
        
        return False, 0

def estimate_model_memory(depth):
    """
    估计模型内存使用 (GB)
    """
    model_dim = depth * 64
    num_layers = depth
    vocab_size = 65536
    
    # 参数数量估计
    embedding_params = vocab_size * model_dim
    layer_params = num_layers * (
        4 * model_dim * model_dim +  # attention weights
        8 * model_dim * model_dim    # MLP weights
    )
    total_params = embedding_params + layer_params
    
    # 每个参数2字节 (bfloat16) + 优化器状态
    memory_gb = total_params * 2 * 3 / (1024**3)  # 参数 + 梯度 + 优化器状态
    return memory_gb

def estimate_batch_memory(device_batch_size, max_seq_len, depth):
    """
    估计批次内存使用 (GB)
    """
    model_dim = depth * 64
    
    # 激活值内存估计
    # 主要是attention和MLP的中间结果
    activation_memory = (
        device_batch_size * max_seq_len * model_dim * depth * 4 * 2  # bfloat16
    ) / (1024**3)
    
    return activation_memory

def main():
    print("=== NanoChat NPU批次大小计算器 ===\n")
    
    # 当前4张NPU配置
    world_size = 4
    max_seq_len = 2048
    total_batch_size = 262144
    
    print("检查当前配置:")
    print("=" * 50)
    
    # 检查base training配置
    print("\n1. Base Training:")
    device_batch_size = 16
    valid, grad_steps = calculate_batch_config(device_batch_size, max_seq_len, total_batch_size, world_size)
    
    # 检查其他配置
    print("\n2. SFT Training:")
    device_batch_size = 8
    target_examples_per_step = 32
    world_examples_per_step = device_batch_size * world_size
    
    print(f"配置参数:")
    print(f"  device_batch_size: {device_batch_size}")
    print(f"  target_examples_per_step: {target_examples_per_step}")
    print(f"  world_size: {world_size}")
    print(f"  world_examples_per_step: {world_examples_per_step}")
    
    if target_examples_per_step % world_examples_per_step == 0:
        sft_grad_steps = target_examples_per_step // world_examples_per_step
        print(f"  ✅ SFT配置有效!")
        print(f"  gradient_accumulation_steps: {sft_grad_steps}")
    else:
        print(f"  ❌ SFT配置可能有问题")
    
    print("\n3. RL Training:")
    device_batch_size = 4
    examples_per_step = 16
    world_examples_per_step = device_batch_size * world_size
    
    print(f"配置参数:")
    print(f"  device_batch_size: {device_batch_size}")
    print(f"  examples_per_step: {examples_per_step}")
    print(f"  world_examples_per_step: {world_examples_per_step}")
    
    if examples_per_step % world_examples_per_step == 0:
        rl_grad_steps = examples_per_step // world_examples_per_step
        print(f"  ✅ RL配置有效!")
        print(f"  gradient_accumulation_steps: {rl_grad_steps}")
    else:
        print(f"  ❌ RL配置可能有问题")
    
    print("\n" + "=" * 50)
    print("配置建议:")
    print("- 确保所有批次大小都能被world_size整除")
    print("- 监控NPU内存使用，调整device_batch_size")
    print("- 如果内存不足，减小device_batch_size")
    print("- 如果内存有余，可以适当增加以提高效率")

if __name__ == "__main__":
    main()
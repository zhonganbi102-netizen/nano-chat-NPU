#!/bin/bash

echo "=== 基础NPU功能测试 ==="

# 只使用一张NPU
export ASCEND_RT_VISIBLE_DEVICES=0

echo "1. 测试基础NPU环境..."
python3 -c "
import torch
import torch_npu
import time

print('=== NPU基础环境测试 ===')
print(f'PyTorch版本: {torch.__version__}')
print(f'torch_npu可用: {torch_npu.npu.is_available()}')
print(f'NPU设备数量: {torch_npu.npu.device_count()}')

if torch_npu.npu.device_count() > 0:
    torch_npu.npu.set_device(0)
    print(f'当前NPU设备: {torch_npu.npu.current_device()}')
    print(f'设备名称: {torch_npu.npu.get_device_name(0)}')

print('\\n=== 基础张量操作测试 ===')
# 创建NPU张量
x = torch.randn(1000, 1000).npu()
print(f'✅ 创建NPU张量成功: {x.shape}')
print(f'张量设备: {x.device}')

# 基础运算
y = x + 1
print(f'✅ NPU加法运算成功: {y.shape}')

# 矩阵乘法
z = torch.matmul(x, x.T)
print(f'✅ NPU矩阵乘法成功: {z.shape}')

print('\\n=== 内存管理测试 ===')
allocated = torch_npu.npu.memory_allocated(0)
reserved = torch_npu.npu.memory_reserved(0)
print(f'NPU内存分配: {allocated / 1024**2:.1f} MB')
print(f'NPU内存预留: {reserved / 1024**2:.1f} MB')

# 清理内存
del x, y, z
torch_npu.npu.empty_cache()
print('✅ 内存清理完成')

allocated_after = torch_npu.npu.memory_allocated(0)
print(f'清理后内存分配: {allocated_after / 1024**2:.1f} MB')

print('\\n🎉 基础NPU功能测试全部通过！')
"

echo ""
echo "2. 测试简单神经网络..."
python3 -c "
import torch
import torch.nn as nn
import torch_npu

print('=== 简单神经网络测试 ===')

# 创建简单的线性层
device = torch.device('npu:0')
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
).to(device)

print(f'✅ 模型创建并移动到NPU成功')
print(f'模型设备: {next(model.parameters()).device}')

# 创建输入数据
x = torch.randn(32, 100).to(device)
print(f'✅ 输入数据创建成功: {x.shape}')

# 前向传播
with torch.no_grad():
    output = model(x)

print(f'✅ 前向传播成功: {output.shape}')

# 检查输出
print(f'输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]')

print('\\n🎉 简单神经网络测试通过！')
"

echo ""
echo "=== 基础测试完成 ==="
echo "如果这些测试都通过，说明NPU基础功能正常。"
echo "问题可能在于复杂的模型初始化或优化器设置。"
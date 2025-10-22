#!/bin/bash

echo "=== 直接在项目目录中测试 ==="

# 确保在正确的目录
cd /mnt/linxid615/bza/nanochat-npu

# 只使用一张NPU
export ASCEND_RT_VISIBLE_DEVICES=0

echo "停止现有训练进程..."
pkill -f "python.*base_train"
sleep 5

echo "开始nanochat模型测试..."

# 直接运行Python，而不是通过-c传递代码
cat > /tmp/test_nanochat_npu.py << 'EOF'
import sys
import os
import time

# 添加当前目录到路径
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

print("=== NanoChat NPU模型测试 ===")

try:
    import torch
    import torch_npu
    print(f"1. ✅ PyTorch NPU导入成功")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   NPU可用: {torch_npu.npu.is_available()}")
    
    torch_npu.npu.set_device(0)
    print(f"   当前设备: {torch_npu.npu.current_device()}")
    
except Exception as e:
    print(f"1. ❌ PyTorch NPU导入失败: {e}")
    exit(1)

try:
    from nanochat.gpt import GPT, GPTConfig
    print("2. ✅ NanoChat模型导入成功")
except Exception as e:
    print(f"2. ❌ NanoChat模型导入失败: {e}")
    print("   检查模块路径...")
    print(f"   当前工作目录: {os.getcwd()}")
    print(f"   sys.path: {sys.path[:3]}...")
    exit(1)

try:
    print("3. 创建小模型配置...")
    config = GPTConfig(
        sequence_len=512,
        vocab_size=1000, 
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128
    )
    print("   ✅ 配置创建成功")
except Exception as e:
    print(f"3. ❌ 配置创建失败: {e}")
    exit(1)

try:
    print("4. 在meta设备上创建模型...")
    start_time = time.time()
    with torch.device("meta"):
        model = GPT(config)
    print(f"   ✅ Meta模型创建成功 ({time.time() - start_time:.2f}s)")
    print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"4. ❌ Meta模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    print("5. 移动模型到NPU...")
    start_time = time.time()
    device = torch.device('npu:0')
    model.to_empty(device=device)
    model.init_weights()
    print(f"   ✅ NPU移动成功 ({time.time() - start_time:.2f}s)")
    
    # 验证模型在NPU上
    first_param = next(model.parameters())
    print(f"   模型设备: {first_param.device}")
    
except Exception as e:
    print(f"5. ❌ NPU移动失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    print("6. 测试前向传播...")
    start_time = time.time()
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"   ✅ 前向传播成功 ({time.time() - start_time:.2f}s)")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {logits.shape}")
    
except Exception as e:
    print(f"6. ❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    print("7. 检查NPU内存使用...")
    allocated = torch_npu.npu.memory_allocated(0)
    reserved = torch_npu.npu.memory_reserved(0)
    print(f"   NPU内存分配: {allocated / 1024**2:.1f} MB")
    print(f"   NPU内存预留: {reserved / 1024**2:.1f} MB")
    
    if allocated > 0:
        print("   ✅ NPU内存正常使用")
    else:
        print("   ⚠️  NPU内存使用为0，可能有问题")
        
except Exception as e:
    print(f"7. ❌ 内存检查失败: {e}")

print("\n🎉 NanoChat NPU模型测试完成！")
print("这证明NPU可以正常运行NanoChat模型。")
print("训练卡住的问题可能在优化器初始化或数据加载部分。")
EOF

python3 /tmp/test_nanochat_npu.py

echo ""
echo "测试完成，清理临时文件..."
rm -f /tmp/test_nanochat_npu.py
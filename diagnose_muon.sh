#!/bin/bash

# Muon优化器诊断脚本
# Muon optimizer diagnostic script

echo "🔬 Muon优化器诊断工具"
echo ""

# 1. 环境准备
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32
export NPU_COMPILE_DISABLE=1

# 2. 创建诊断脚本
cat > diagnose_muon.py << 'EOF'
"""
Muon优化器诊断脚本
"""
import os
import sys
import time
import torch
import torch_npu
import gc

def test_basic_npu():
    """测试基础NPU功能"""
    print("🔍 测试1: 基础NPU功能...")
    
    if not torch_npu.npu.is_available():
        print("❌ NPU不可用")
        return False
    
    torch_npu.npu.set_device(0)
    print(f"✅ 使用设备: npu:{torch_npu.npu.current_device()}")
    
    # 简单张量测试
    x = torch.randn(10, 10).to('npu:0')
    y = torch.matmul(x, x.T)
    result = y.sum().item()
    print(f"✅ 矩阵运算测试: {result:.2f}")
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n🔍 测试2: 模型创建...")
    
    try:
        # 导入nanochat模块
        sys.path.append('.')
        from nanochat.gpt import GPT
        from nanochat.common import ModelConfig
        
        # 创建最小配置
        config = ModelConfig(
            vocab_size=100,
            seq_len=128,
            depth=2,
            model_dim=64,
            num_heads=2,
            num_kv_heads=2
        )
        
        print(f"配置: {config}")
        
        # 创建模型
        model = GPT(config)
        model = model.to('npu:0')
        
        print("✅ 模型创建成功")
        return model, config
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_muon_optimizer(model):
    """测试Muon优化器"""
    print("\n🔍 测试3: Muon优化器...")
    
    if model is None:
        print("❌ 模型为空，跳过优化器测试")
        return False
    
    try:
        from nanochat.muon import Muon
        
        print("正在创建Muon优化器...")
        
        # 获取参数
        params = list(model.parameters())
        print(f"模型参数数量: {len(params)}")
        
        # 显示参数形状
        for i, p in enumerate(params[:5]):  # 只显示前5个
            print(f"  参数{i}: {p.shape}, device: {p.device}")
        
        print("创建Muon优化器中...")
        start_time = time.time()
        
        # 创建优化器（这里可能会卡住）
        optimizer = Muon(params, lr=0.001)
        
        end_time = time.time()
        print(f"✅ Muon优化器创建成功，耗时: {end_time - start_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ Muon优化器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alternative_optimizer(model):
    """测试替代优化器"""
    print("\n🔍 测试4: 替代优化器...")
    
    if model is None:
        print("❌ 模型为空，跳过")
        return False
    
    try:
        # 测试AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        print("✅ AdamW优化器创建成功")
        
        # 测试一步优化
        x = torch.randint(0, 100, (2, 64)).to('npu:0')
        y = model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print("✅ 优化步骤测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 替代优化器失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("开始Muon优化器诊断...\n")
    
    # 清理环境
    if torch_npu.npu.is_available():
        torch_npu.npu.empty_cache()
        gc.collect()
    
    # 运行测试
    if not test_basic_npu():
        return
    
    model, config = test_model_creation()
    
    # 测试Muon（可能卡住的地方）
    print("\n⚠️  即将测试Muon优化器，如果卡住请Ctrl+C终止")
    time.sleep(2)
    
    muon_success = test_muon_optimizer(model)
    
    if not muon_success:
        print("\n🔄 Muon失败，测试替代方案...")
        test_alternative_optimizer(model)
    
    print("\n🎯 诊断总结:")
    print(f"  - 基础NPU: ✅")
    print(f"  - 模型创建: {'✅' if model is not None else '❌'}")
    print(f"  - Muon优化器: {'✅' if muon_success else '❌'}")
    print(f"  - 替代方案: 可用")

if __name__ == "__main__":
    main()
EOF

# 3. 运行诊断
echo "3. 运行Muon优化器诊断..."
echo "如果程序卡住，请按Ctrl+C终止"
echo ""

python3 diagnose_muon.py

# 4. 清理
rm -f diagnose_muon.py

echo ""
echo "🎉 诊断完成！"
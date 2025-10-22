#!/usr/bin/env python3
"""
简单的NPU功能测试脚本
Simple NPU functionality test script
"""

import torch
import sys

def test_basic_npu():
    """测试基本NPU功能"""
    print("=== 基本NPU功能测试 ===")
    
    try:
        import torch_npu
        print("✅ torch_npu 导入成功")
    except ImportError:
        print("❌ torch_npu 导入失败")
        return False
    
    # 检查NPU可用性
    if not torch_npu.npu.is_available():
        print("❌ NPU不可用")
        return False
    
    print(f"✅ NPU可用，设备数: {torch_npu.npu.device_count()}")
    
    # 基本tensor操作
    try:
        x = torch.randn(4, 4)
        x_npu = x.to('npu')
        y_npu = x_npu * 2
        result = y_npu.cpu()
        print("✅ 基本tensor操作成功")
    except Exception as e:
        print(f"❌ tensor操作失败: {e}")
        return False
    
    return True

def test_nanochat_imports():
    """测试nanochat模块导入"""
    print("\n=== NanoChat模块测试 ===")
    
    try:
        from nanochat.common import compute_init
        print("✅ nanochat.common 导入成功")
        
        from nanochat.gpt import GPT, GPTConfig
        print("✅ nanochat.gpt 导入成功")
        
        from nanochat.engine import Engine
        print("✅ nanochat.engine 导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_mixed_precision():
    """测试混合精度"""
    print("\n=== 混合精度测试 ===")
    
    try:
        import torch_npu
        device = torch.device('npu:0')
        
        with torch.amp.autocast(device_type='npu', dtype=torch.bfloat16):
            x = torch.randn(32, 64, device=device)
            linear = torch.nn.Linear(64, 32).to(device)
            y = linear(x)
            loss = y.sum()
            
        print("✅ 混合精度支持正常")
        return True
    except Exception as e:
        print(f"❌ 混合精度测试失败: {e}")
        return False

if __name__ == "__main__":
    print("NanoChat NPU 兼容性测试")
    print("=" * 40)
    
    success = True
    
    success &= test_basic_npu()
    success &= test_nanochat_imports() 
    success &= test_mixed_precision()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 所有测试通过! 可以开始使用NPU训练")
    else:
        print("❌ 某些测试失败，请检查环境配置")
        sys.exit(1)

#!/usr/bin/env python3
"""
华为昇腾NPU环境检查脚本
Huawei Ascend NPU Environment Check Script
"""

import sys
import os

def check_npu_environment():
    """检查NPU环境配置"""
    print("=== 华为昇腾NPU环境检查 ===")
    
    # 1. 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 2. 检查torch安装
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 3. 检查torch_npu
    try:
        import torch_npu
        print(f"torch_npu版本: {torch_npu.__version__}")
    except ImportError:
        print("❌ torch_npu未安装")
        print("请运行: pip install torch-npu")
        return False
    
    # 4. 检查NPU可用性
    try:
        if torch_npu.npu.is_available():
            print(f"✅ NPU可用，设备数量: {torch_npu.npu.device_count()}")
            
            # 检查每个NPU设备
            for i in range(torch_npu.npu.device_count()):
                try:
                    device_name = torch_npu.npu.get_device_name(i)
                    print(f"  设备 {i}: {device_name}")
                except:
                    print(f"  设备 {i}: 信息获取失败")
        else:
            print("❌ NPU不可用")
            return False
    except Exception as e:
        print(f"❌ NPU检查失败: {e}")
        return False
    
    # 5. 检查环境变量
    print("\n=== 环境变量检查 ===")
    env_vars = [
        'ASCEND_HOME',
        'ASCEND_OPP_PATH',
        'TOOLCHAIN_HOME',
        'ASCEND_AICPU_PATH',
        'ASCEND_RT_VISIBLE_DEVICES'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not Set')
        print(f"{var}: {value}")
    
    # 6. 测试NPU基本操作
    print("\n=== NPU基本操作测试 ===")
    try:
        # 创建tensor并移到NPU
        x = torch.randn(10, 10)
        x_npu = x.to('npu:0')
        print("✅ 创建NPU tensor成功")
        
        # 基本运算
        y_npu = x_npu + x_npu
        print("✅ NPU tensor运算成功")
        
        # 内存检查
        torch_npu.npu.empty_cache()
        memory_allocated = torch_npu.npu.memory_allocated() / 1024 / 1024
        print(f"✅ NPU内存使用: {memory_allocated:.2f} MB")
        
    except Exception as e:
        print(f"❌ NPU操作测试失败: {e}")
        return False
    
    # 7. 检查分布式训练支持
    print("\n=== 分布式训练检查 ===")
    try:
        import torch.distributed as dist
        print("✅ 分布式训练模块可用")
        
        # 检查HCCL后端
        if hasattr(torch.distributed, 'Backend'):
            if hasattr(torch.distributed.Backend, 'HCCL'):
                print("✅ HCCL后端可用")
            else:
                print("⚠️  HCCL后端不可用，将使用NCCL")
        
    except Exception as e:
        print(f"⚠️  分布式检查失败: {e}")
    
    print("\n=== 检查完成 ===")
    print("✅ NPU环境配置正常")
    return True

def test_training_compatibility():
    """测试训练兼容性"""
    print("\n=== 训练兼容性测试 ===")
    
    try:
        import torch
        import torch_npu
        
        # 测试自动混合精度
        device = torch.device('npu:0')
        with torch.amp.autocast(device_type='npu', dtype=torch.bfloat16):
            x = torch.randn(32, 512, device=device)
            linear = torch.nn.Linear(512, 256).to(device)
            y = linear(x)
            print("✅ 自动混合精度支持")
        
        # 测试梯度计算
        loss = y.sum()
        loss.backward()
        print("✅ 梯度计算支持")
        
        # 测试优化器
        optimizer = torch.optim.AdamW(linear.parameters())
        optimizer.step()
        print("✅ 优化器支持")
        
    except Exception as e:
        print(f"❌ 训练兼容性测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = check_npu_environment()
    if success:
        test_training_compatibility()
        print("\n🎉 环境检查完成，可以开始训练!")
    else:
        print("\n❌ 环境配置有问题，请检查安装")
        sys.exit(1)

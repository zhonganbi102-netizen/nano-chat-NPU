#!/usr/bin/env python3
"""
NPU API兼容性测试脚本
测试不同版本的torch_npu API
"""

import torch
import torch_npu

def test_npu_basic():
    """测试基本NPU功能"""
    print("=== NPU基本功能测试 ===")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"torch_npu可用: {torch_npu.npu.is_available()}")
    print(f"NPU设备数量: {torch_npu.npu.device_count()}")
    
    if torch_npu.npu.device_count() == 0:
        print("❌ 没有可用的NPU设备")
        return False
    
    # 设置设备
    device_id = 0
    torch_npu.npu.set_device(device_id)
    print(f"当前NPU设备: {torch_npu.npu.current_device()}")
    print(f"设备名称: {torch_npu.npu.get_device_name(device_id)}")
    
    return True

def test_tensor_creation():
    """测试张量创建和操作"""
    print("\n=== 张量创建测试 ===")
    
    try:
        # 方法1: 使用torch.tensor().npu()
        print("测试方法1: torch.tensor().npu()")
        x1 = torch.tensor([1.0, 2.0, 3.0]).npu()
        print(f"✅ 成功: {x1}")
        print(f"设备: {x1.device}")
        
        # 方法2: 使用torch.zeros().npu()
        print("\n测试方法2: torch.zeros().npu()")
        x2 = torch.zeros(3, 3).npu()
        print(f"✅ 成功: {x2.shape}")
        print(f"设备: {x2.device}")
        
        # 方法3: 直接在NPU上创建
        print("\n测试方法3: 直接在NPU设备上创建")
        device = torch.device('npu:0')
        x3 = torch.randn(2, 2, device=device)
        print(f"✅ 成功: {x3.shape}")
        print(f"设备: {x3.device}")
        
        return True, [x1, x2, x3]
        
    except Exception as e:
        print(f"❌ 张量创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_npu_operations(tensors):
    """测试NPU计算操作"""
    print("\n=== NPU计算测试 ===")
    
    if not tensors:
        print("没有可用的测试张量")
        return False
    
    try:
        x = tensors[1]  # 使用3x3的张量
        
        # 基本运算
        print("测试基本运算...")
        y = x + 1
        z = torch.matmul(x, x.T)
        print(f"✅ 加法成功: {y.shape}")
        print(f"✅ 矩阵乘法成功: {z.shape}")
        
        # 内存管理
        print("测试内存管理...")
        del y, z
        torch_npu.npu.empty_cache()
        print("✅ 内存清理成功")
        
        return True
        
    except Exception as e:
        print(f"❌ NPU计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_info():
    """测试内存信息"""
    print("\n=== 内存信息测试 ===")
    
    try:
        # 获取内存信息
        if hasattr(torch_npu.npu, 'memory_allocated'):
            allocated = torch_npu.npu.memory_allocated(0)
            print(f"已分配内存: {allocated / 1024**2:.2f} MB")
        
        if hasattr(torch_npu.npu, 'memory_reserved'):
            reserved = torch_npu.npu.memory_reserved(0)
            print(f"预留内存: {reserved / 1024**2:.2f} MB")
        
        # 内存统计
        if hasattr(torch_npu.npu, 'memory_stats'):
            stats = torch_npu.npu.memory_stats(0)
            print(f"内存统计: {len(stats)} 项")
        
        return True
        
    except Exception as e:
        print(f"内存信息获取失败: {e}")
        return False

def cleanup_tensors(tensors):
    """清理测试张量"""
    print("\n=== 清理资源 ===")
    
    for i, tensor in enumerate(tensors):
        try:
            del tensor
            print(f"✅ 清理张量 {i+1}")
        except:
            pass
    
    try:
        torch_npu.npu.empty_cache()
        print("✅ 清理NPU缓存")
    except Exception as e:
        print(f"缓存清理失败: {e}")

def main():
    """主测试函数"""
    print("NPU API兼容性测试开始...\n")
    
    # 基本功能测试
    if not test_npu_basic():
        return
    
    # 张量创建测试
    success, tensors = test_tensor_creation()
    if not success:
        return
    
    # 计算测试
    test_npu_operations(tensors)
    
    # 内存信息测试
    test_memory_info()
    
    # 清理
    cleanup_tensors(tensors)
    
    print("\n=== 测试完成 ===")
    print("如果所有测试都通过，NPU环境配置正确！")

if __name__ == "__main__":
    main()
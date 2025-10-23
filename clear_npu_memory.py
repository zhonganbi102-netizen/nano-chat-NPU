#!/usr/bin/env python3
"""
清理NPU内存和缓存的脚本
"""

import torch
import torch_npu
import gc
import os
import sys

def clear_npu_memory():
    """清理NPU内存"""
    print("=== 开始清理NPU内存 ===")
    
    try:
        # 检查NPU是否可用
        if not torch_npu.npu.is_available():
            print("❌ NPU不可用")
            return False
            
        # 获取NPU设备数量
        device_count = torch_npu.npu.device_count()
        print(f"检测到 {device_count} 个NPU设备")
        
        # 清理每个NPU的内存
        for i in range(device_count):
            print(f"清理NPU {i}...")
            torch_npu.npu.set_device(i)
            
            # 清空缓存
            torch_npu.npu.empty_cache()
            
            # 强制垃圾回收
            gc.collect()
            
            # 获取内存使用情况
            if hasattr(torch_npu.npu, 'memory_allocated'):
                allocated = torch_npu.npu.memory_allocated(i) / 1024**3
                cached = torch_npu.npu.memory_reserved(i) / 1024**3
                print(f"  NPU {i}: 已分配 {allocated:.2f} GiB, 缓存 {cached:.2f} GiB")
        
        print("✅ NPU内存清理完成")
        return True
        
    except Exception as e:
        print(f"❌ 清理NPU内存时出错: {e}")
        return False

def reset_npu_environment():
    """重置NPU环境"""
    print("=== 重置NPU环境 ===")
    
    try:
        # 重新初始化NPU
        torch_npu.npu.init()
        print("✅ NPU环境重置完成")
        return True
        
    except Exception as e:
        print(f"❌ 重置NPU环境时出错: {e}")
        return False

def main():
    print("NPU内存清理工具")
    print("-" * 50)
    
    # 清理内存
    if clear_npu_memory():
        print("内存清理成功")
    else:
        print("内存清理失败")
        
    # 重置环境
    if reset_npu_environment():
        print("环境重置成功")
    else:
        print("环境重置失败")
    
    print("\n建议执行以下命令进一步清理:")
    print("1. 杀死所有Python进程: pkill -f python")
    print("2. 重启NPU驱动: /usr/local/Ascend/driver/tools/docker_start.sh")
    print("3. 检查内存: npu-smi info")

if __name__ == "__main__":
    main()
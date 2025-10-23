#!/usr/bin/env python3
"""
单NPU训练的Python脚本
直接调用训练逻辑，避免torchrun的开销
"""

import os
import sys
import torch
import torch_npu
import gc

def setup_single_npu():
    """设置单NPU环境"""
    print("=== 设置单NPU环境 ===")
    
    # 设置环境变量
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # NPU内存优化
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:64'
    os.environ['NPU_COMPILE_DISABLE'] = '1'
    
    # 检查NPU
    if not torch_npu.npu.is_available():
        raise RuntimeError("NPU不可用")
    
    # 清理内存
    torch_npu.npu.empty_cache()
    gc.collect()
    
    device = torch_npu.npu.current_device()
    print(f"使用NPU设备: {device}")
    
    return device

def main():
    """主训练函数"""
    try:
        # 设置NPU
        device = setup_single_npu()
        
        print("开始导入训练模块...")
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # 导入训练脚本
        from scripts import base_train
        
        # 设置训练参数
        sys.argv = [
            'base_train.py',
            '--run=direct_single_npu',
            '--depth=6',
            '--device_batch_size=8',
            '--total_batch_size=16384',
            '--num_iterations=2000'
        ]
        
        print("开始训练...")
        base_train.main()
        
    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理内存
        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()
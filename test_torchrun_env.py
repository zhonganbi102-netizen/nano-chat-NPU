#!/usr/bin/env python3
"""
测试torchrun子进程中的TBE模块可用性
"""

import os
import sys

print(f"进程 {os.getenv('RANK', 'main')} 开始环境测试...")

# 显示环境变量
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '未设置')[:100]}...")
print(f"ASCEND_HOME: {os.environ.get('ASCEND_HOME', '未设置')}")

# 显示Python路径
print("Python sys.path 前5项:")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i}: {path}")

# 测试TBE模块
print("\n测试TBE模块...")
try:
    import tbe
    print("✅ TBE模块加载成功")
except ImportError as e:
    print(f"❌ TBE模块加载失败: {e}")
    
    # 尝试手动添加路径
    print("尝试手动添加TBE路径...")
    tbe_paths = [
        '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages',
        '/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe'
    ]
    
    for path in tbe_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"  添加路径: {path}")
    
    try:
        import tbe
        print("✅ 手动添加路径后TBE模块加载成功")
    except ImportError as e2:
        print(f"❌ 手动添加路径后仍然失败: {e2}")

# 测试torch_npu
print("\n测试torch_npu...")
try:
    import torch_npu
    print(f"✅ torch_npu版本: {torch_npu.__version__}")
    print(f"✅ NPU可用: {torch_npu.npu.is_available()}")
    print(f"✅ NPU数量: {torch_npu.npu.device_count()}")
except ImportError as e:
    print(f"❌ torch_npu导入失败: {e}")

print(f"进程 {os.getenv('RANK', 'main')} 环境测试完成\n")

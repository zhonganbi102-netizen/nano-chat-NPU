#!/usr/bin/env python3
"""
测试 nanochat 参数格式
"""

import sys
import os

# 添加路径
sys.path.insert(0, '.')

print("测试参数格式...")
print(f"参数: {sys.argv[1:]}")

try:
    # 模拟 base_train.py 的配置加载
    run = "dummy"
    depth = 20
    device_batch_size = 32
    total_batch_size = 524288
    num_iterations = -1
    
    print(f"默认参数:")
    print(f"  run = {run}")
    print(f"  depth = {depth}")
    print(f"  device_batch_size = {device_batch_size}")
    
    # 获取配置keys
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    print(f"可用配置keys: {config_keys}")
    
    # 执行配置器
    exec(open(os.path.join('nanochat', 'configurator.py')).read())
    
    print(f"修改后参数:")
    print(f"  run = {run}")
    print(f"  depth = {depth}")
    print(f"  device_batch_size = {device_batch_size}")
    
    print("✅ 参数格式正确!")
    
except Exception as e:
    print(f"❌ 参数格式错误: {e}")
    import traceback
    traceback.print_exc()
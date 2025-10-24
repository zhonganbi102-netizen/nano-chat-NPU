#!/usr/bin/env python3
"""
调试配置问题
"""

import sys
import os
from ast import literal_eval

# 添加路径
sys.path.insert(0, '.')

print("🔍 调试配置问题...")
print(f"参数: {sys.argv[1:]}")

try:
    # 模拟 base_train.py 的配置加载
    run = "dummy"
    depth = 20
    device_batch_size = 32
    total_batch_size = 524288
    num_iterations = -1
    target_flops = -1.0
    target_param_data_ratio = 20
    embedding_lr = 0.2
    unembedding_lr = 0.004
    weight_decay = 0.0
    matrix_lr = 0.02
    grad_clip = 1.0
    eval_every = 250
    eval_tokens = 20*524288
    core_metric_every = 2000
    core_metric_max_per_task = 500
    sample_every = 2000
    model_tag = ""
    max_seq_len = 2048
    
    print(f"默认参数类型:")
    print(f"  run = {run} ({type(run)})")
    print(f"  depth = {depth} ({type(depth)})")
    print(f"  embedding_lr = {embedding_lr} ({type(embedding_lr)})")
    print(f"  unembedding_lr = {unembedding_lr} ({type(unembedding_lr)})")
    print(f"  matrix_lr = {matrix_lr} ({type(matrix_lr)})")
    
    # 获取配置keys
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    print(f"可用配置keys: {config_keys}")
    
    # 手动解析参数来调试
    def print0(s="",**kwargs):
        ddp_rank = int(os.environ.get('RANK', 0))
        if ddp_rank == 0:
            print(s, **kwargs)

    for arg in sys.argv[1:]:
        print(f"\n处理参数: {arg}")
        if '=' not in arg:
            print(f"  -> 作为配置文件处理")
            assert not arg.startswith('--'), f"配置文件不能以--开头: {arg}"
        else:
            print(f"  -> 作为key=value处理")
            assert arg.startswith('--'), f"参数必须以--开头: {arg}"
            key, val = arg.split('=')
            key = key[2:]
            print(f"  -> key={key}, val={val}")
            
            if key in globals():
                print(f"  -> key存在于globals")
                try:
                    attempt = literal_eval(val)
                    print(f"  -> literal_eval成功: {attempt} ({type(attempt)})")
                except (SyntaxError, ValueError) as e:
                    print(f"  -> literal_eval失败: {e}")
                    attempt = val
                    print(f"  -> 使用字符串: {attempt} ({type(attempt)})")
                
                if globals()[key] is not None:
                    attempt_type = type(attempt)
                    default_type = type(globals()[key])
                    print(f"  -> 类型检查: {attempt_type} vs {default_type}")
                    if attempt_type != default_type:
                        print(f"  -> ❌ 类型不匹配!")
                        raise AssertionError(f"Type mismatch: {attempt_type} != {default_type}")
                    else:
                        print(f"  -> ✅ 类型匹配")
                
                print(f"  -> 设置: {key} = {attempt}")
                globals()[key] = attempt
            else:
                print(f"  -> ❌ key不存在: {key}")
                raise ValueError(f"Unknown config key: {key}")
    
    print("✅ 配置解析成功!")
    
except Exception as e:
    print(f"❌ 配置解析失败: {e}")
    import traceback
    traceback.print_exc()
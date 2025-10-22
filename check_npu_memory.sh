#!/bin/bash

echo "📊 NPU内存使用分析..."

python3 -c "
import torch_npu
import torch

print('=== NPU设备信息 ===')
for i in range(torch_npu.npu.device_count()):
    print(f'NPU {i}:')
    
    # 总内存
    total_memory = torch_npu.npu.get_device_properties(i).total_memory / 1024**3
    print(f'  总内存: {total_memory:.2f} GB')
    
    # 当前使用
    try:
        torch_npu.npu.set_device(i)
        allocated = torch_npu.npu.memory_allocated(i) / 1024**3
        reserved = torch_npu.npu.memory_reserved(i) / 1024**3
        free_reserved = reserved - allocated
        
        print(f'  已分配: {allocated:.2f} GB')
        print(f'  已保留: {reserved:.2f} GB')
        print(f'  保留空闲: {free_reserved:.2f} GB')
        print(f'  系统可用: {total_memory - reserved:.2f} GB')
    except Exception as e:
        print(f'  状态: 无法获取 ({e})')
    print()

print('=== 内存优化建议 ===')
print('当前配置占用过高，建议:')
print('1. 减小模型深度: depth=8 (代替12)')
print('2. 减小batch size: device_batch_size=4 (代替8)')
print('3. 减小序列长度: max_seq_len=1024 (代替2048)')
print('4. 启用内存分片: PYTORCH_NPU_ALLOC_CONF=\"max_split_size_mb:512\"')
"

echo ""
echo "💡 推荐运行: ./memory_opt_4npu.sh"
#!/bin/bash

echo "🔍 HCCL通信诊断..."

# 检查HCCL工具
echo "=== 1. 检查HCCL工具 ==="
if command -v hccn_tool &> /dev/null; then
    echo "✅ hccn_tool 可用"
    echo "检查NPU TLS状态..."
    for i in {0..3}; do
        echo "NPU $i TLS状态:"
        hccn_tool -i $i -tls -g 2>/dev/null || echo "  获取失败"
    done
else
    echo "❌ hccn_tool 不可用"
fi

echo ""
echo "=== 2. 检查网络连接 ==="
echo "本地回环测试:"
ping -c 3 127.0.0.1

echo ""
echo "=== 3. 检查端口占用 ==="
echo "检查29500端口:"
netstat -tlnp | grep 29500 || echo "端口29500未被占用"

echo ""
echo "=== 4. NPU设备状态 ==="
python3 -c "
import torch_npu
import torch

try:
    device_count = torch_npu.npu.device_count()
    print(f'NPU设备数: {device_count}')
    
    for i in range(device_count):
        try:
            torch_npu.npu.set_device(i)
            props = torch_npu.npu.get_device_properties(i)
            print(f'NPU {i}:')
            print(f'  名称: {props.name}')
            print(f'  总内存: {props.total_memory/1024**3:.1f}GB')
            
            # 简单计算测试
            x = torch.randn(10, 10, device=f'npu:{i}')
            y = x @ x.T
            print(f'  计算测试: 通过')
            del x, y
            torch_npu.npu.empty_cache()
        except Exception as e:
            print(f'NPU {i}: 错误 - {e}')
            
except Exception as e:
    print(f'NPU检查失败: {e}')
"

echo ""
echo "=== 5. 测试分布式初始化 ==="
echo "启动最小分布式测试..."

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501  # 使用不同端口避免冲突
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

python3 -c "
import torch
import torch.distributed as dist
import torch_npu
import os

def test_distributed():
    try:
        # 初始化分布式
        if not dist.is_initialized():
            dist.init_process_group(
                backend='hccl',
                init_method='env://',
                world_size=int(os.environ.get('WORLD_SIZE', 1)),
                rank=int(os.environ.get('RANK', 0))
            )
        
        print('✅ 分布式初始化成功')
        
        # 测试简单通信
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            print(f'Rank {rank}/{world_size}')
            
            # 清理
            dist.destroy_process_group()
            print('✅ 分布式清理成功')
        
    except Exception as e:
        print(f'❌ 分布式测试失败: {e}')

if __name__ == '__main__':
    test_distributed()
" || echo "❌ 分布式测试失败"

echo ""
echo "=== 诊断完成 ==="
echo "如果看到通信错误，建议:"
echo "1. 重启NPU驱动: sudo systemctl restart npu-driver"
echo "2. 检查HCCL环境变量设置"
echo "3. 使用简化训练脚本: ./simple_4npu.sh"
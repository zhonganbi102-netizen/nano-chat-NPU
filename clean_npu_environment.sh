#!/bin/bash

echo "=== NPU环境清理和重置脚本 ==="

echo "1. 停止所有Python训练进程..."
pkill -f python3
pkill -f torchrun
sleep 2

echo "2. 检查残留进程..."
remaining_procs=$(ps aux | grep -E "(python|torch)" | grep -v grep | wc -l)
if [ $remaining_procs -gt 0 ]; then
    echo "发现残留进程，强制终止..."
    pkill -9 -f python
    pkill -9 -f torch
    sleep 2
fi

echo "3. 清理NPU内存和状态..."
# 尝试重置NPU设备
if command -v npu-smi &> /dev/null; then
    echo "重置NPU设备..."
    npu-smi reset -d 0,1,2,3 2>/dev/null || echo "NPU重置失败（正常情况）"
fi

echo "4. 等待NPU设备稳定..."
sleep 10

echo "5. 清理HCCL相关进程和资源..."
# 清理可能的HCCL残留
ps aux | grep -i hccl | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null

echo "6. 清理共享内存和信号量..."
# 清理可能的共享内存段
ipcs -m | grep $USER | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null
ipcs -s | grep $USER | awk '{print $2}' | xargs -r ipcrm -s 2>/dev/null

echo "7. 重新加载NPU驱动模块（如果需要）..."
# 这通常需要root权限，可能不会成功
if [ "$EUID" -eq 0 ]; then
    echo "尝试重新加载NPU模块..."
    # 注意：这些命令可能因系统而异
    lsmod | grep -i ascend > /dev/null && echo "Ascend模块已加载"
fi

echo "8. 验证NPU状态..."
if command -v npu-smi &> /dev/null; then
    echo "当前NPU状态:"
    npu-smi info
else
    echo "npu-smi命令不可用"
fi

echo "9. 测试基本NPU功能..."
python3 -c "
try:
    import torch_npu
    print(f'✅ torch_npu导入成功')
    print(f'NPU数量: {torch_npu.npu.device_count()}')
    
    # 测试基本张量操作
    if torch_npu.npu.device_count() > 0:
        torch_npu.npu.set_device(0)
        x = torch_npu.FloatTensor([1.0])
        print(f'✅ 基本NPU操作成功: {x}')
        
        # 清理测试张量
        del x
        torch_npu.npu.empty_cache()
    else:
        print('❌ 没有可用的NPU设备')
        
except Exception as e:
    print(f'❌ NPU测试失败: {e}')
"

echo ""
echo "=== 清理完成 ==="
echo "建议:"
echo "1. 等待额外10-30秒让系统完全稳定"
echo "2. 使用单NPU测试: ./test_single_npu.sh"
echo "3. 如果还有问题，可能需要重启服务器"
echo ""
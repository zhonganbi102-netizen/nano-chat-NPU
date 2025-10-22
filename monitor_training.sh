#!/bin/bash

echo "=== NPU性能监控 ==="

while true; do
    clear
    echo "时间: $(date)"
    echo "======================================"
    
    # NPU使用率
    echo "NPU使用率:"
    python3 -c "
import torch_npu
import time
try:
    for i in range(torch_npu.npu.device_count()):
        memory_used = torch_npu.npu.memory_allocated(i) / 1024**3
        memory_total = torch_npu.npu.memory_reserved(i) / 1024**3
        print(f'NPU {i}: 内存 {memory_used:.1f}G/{memory_total:.1f}G')
except Exception as e:
    print(f'获取NPU信息失败: {e}')
"
    echo ""
    
    # 训练进程
    echo "训练进程:"
    ps aux | grep -E "(base_train|torchrun)" | grep -v grep | head -10
    echo ""
    
    # 检查最新日志
    if [ -d "logs" ]; then
        echo "最新训练日志:"
        latest_log=$(ls -t logs/*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            echo "文件: $latest_log"
            tail -3 "$latest_log" 2>/dev/null | grep -E "(step|loss|tok/s)" || echo "暂无训练日志"
        else
            echo "未找到日志文件"
        fi
    else
        echo "日志目录不存在"
    fi
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    sleep 5
done
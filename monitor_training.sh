#!/bin/bash

echo "=== NPU训练进度监控 ==="

while true; do
    echo "--- $(date) ---"
    
    # 检查进程状态
    echo "1. 训练进程状态:"
    pgrep -f "python.*base_train" > /dev/null && echo "✅ 训练进程运行中" || echo "❌ 训练进程未运行"
    
    # 检查NPU使用率
    echo "2. NPU使用情况:"
    if command -v npu-smi &> /dev/null; then
        echo "完整NPU状态:"
        npu-smi info
        echo ""
        echo "NPU利用率汇总:"
        npu-smi info | grep -E "AICore|Process" | head -10
    fi
    
    # 检查内存使用
    echo "3. 系统内存:"
    free -h | head -2
    
    # 检查最新日志
    echo "4. 最新日志输出:"
    if [ -d "wandb" ]; then
        find wandb -name "*.log" -type f -exec tail -n 3 {} \; 2>/dev/null | tail -6
    fi
    
    echo "按Ctrl+C停止监控"
    echo "=========================="
    
    sleep 10
done
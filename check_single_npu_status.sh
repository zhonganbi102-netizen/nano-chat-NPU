#!/bin/bash

echo "=== 单NPU训练状态检查 ==="

echo "1. 检查训练进程详细信息..."
TRAIN_PID=$(pgrep -f "python.*base_train" | head -1)
if [ -n "$TRAIN_PID" ]; then
    echo "✅ 找到训练进程 PID: $TRAIN_PID"
    echo "进程详情:"
    ps -p $TRAIN_PID -o pid,ppid,cmd,etime,pcpu,pmem
    
    echo ""
    echo "进程状态:"
    cat /proc/$TRAIN_PID/status | grep -E "(State|VmRSS|VmPeak)" 2>/dev/null || echo "无法读取进程状态"
    
    echo ""
    echo "打开的文件描述符数量:"
    ls /proc/$TRAIN_PID/fd 2>/dev/null | wc -l || echo "无法访问文件描述符"
else
    echo "❌ 没有找到训练进程"
    exit 1
fi

echo ""
echo "2. 检查NPU 0设备 (单NPU训练使用的设备)..."
if command -v npu-smi &> /dev/null; then
    echo "NPU 0设备状态:"
    npu-smi info | grep -A 5 -B 1 "NPU.*0"
    
    echo ""
    echo "NPU 0进程信息:"
    npu-smi info | grep -A 10 "NPU.*Process"
fi

echo ""
echo "3. 检查Python进程的NPU使用..."
python3 -c "
import torch
import torch_npu
print(f'当前进程NPU设备: {torch_npu.npu.current_device()}')
print(f'NPU 0内存使用: {torch_npu.npu.memory_allocated(0) / 1024**2:.1f} MB')
print(f'NPU 0缓存: {torch_npu.npu.memory_reserved(0) / 1024**2:.1f} MB')
" 2>/dev/null || echo "无法获取NPU使用信息 (可能正在初始化)"

echo ""
echo "4. 检查最近的日志输出..."
if [ -d "wandb" ]; then
    echo "最新wandb日志:"
    find wandb -name "*.log" -type f -exec tail -n 5 {} \; 2>/dev/null
fi

echo ""
echo "5. 估计当前阶段..."
if [ -n "$TRAIN_PID" ]; then
    # 检查进程的系统调用来判断在做什么
    timeout 3 strace -p $TRAIN_PID -c 2>&1 | tail -10 | head -5 2>/dev/null || echo "无法获取系统调用信息"
fi

echo ""
echo "6. 建议操作:"
if [ -n "$TRAIN_PID" ]; then
    runtime=$(ps -p $TRAIN_PID -o etime= 2>/dev/null | tr -d ' ')
    echo "训练已运行: $runtime"
    echo "- 如果运行超过10分钟且NPU利用率为0%，可能卡住了"
    echo "- 正常初始化通常需要2-5分钟"
    echo "- 可以等待或用 kill $TRAIN_PID 停止"
fi
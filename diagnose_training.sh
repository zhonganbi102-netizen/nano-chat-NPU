#!/bin/bash

echo "=== NPU训练状态诊断工具 ==="

echo "1. 检查训练进程状态..."
echo "Python进程:"
ps aux | grep python | grep -v grep

echo ""
echo "2. 检查NPU使用情况..."
npu-smi info

echo ""
echo "3. 检查网络和分布式训练状态..."
echo "HCCL相关进程:"
ps aux | grep -E "(hccl|rank)" | grep -v grep

echo ""
echo "4. 检查系统资源..."
echo "内存使用:"
free -h

echo "CPU使用:"
top -bn1 | head -20

echo ""
echo "5. 检查日志文件..."
if [ -d "/mnt/linxid615/bza/nanochat-npu/wandb" ]; then
    echo "最新的wandb日志:"
    find /mnt/linxid615/bza/nanochat-npu/wandb -name "*.log" -type f -exec ls -la {} \; | tail -5
fi

echo ""
echo "6. 检查可能的错误..."
echo "检查是否有CUDA相关错误:"
dmesg | tail -20 | grep -i error

echo ""
echo "7. 建议的调试步骤:"
echo "- 如果NPU使用率持续为0%，可能是初始化卡住了"
echo "- 检查是否有死锁或等待其他进程"
echo "- 可以尝试杀死进程并重新启动"
echo "- 使用更小的batch size测试"

echo ""
echo "8. 快速重启命令:"
echo "  pkill -f python3"
echo "  ./speedrun_npu.sh"
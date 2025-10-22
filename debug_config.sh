#!/bin/bash

echo "🔍 调试参数类型问题..."

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

echo "测试参数传递..."

# 先测试单个参数
echo "1. 测试run参数..."
python3 scripts/base_train.py --run="debug_test" --num_iterations=10 2>&1 | head -20

echo ""
echo "2. 测试数字参数..."
python3 scripts/base_train.py --num_iterations=10 2>&1 | head -20

echo ""
echo "完成调试"
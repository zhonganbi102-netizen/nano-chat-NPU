#!/bin/bash

# 华为昇腾NPU环境配置脚本
# Setup script for Huawei Ascend NPU environment

echo "=== 配置华为昇腾NPU环境 / Setting up Huawei Ascend NPU Environment ==="

# 1. 检查NPU驱动和固件
echo "1. 检查NPU设备状态..."
if command -v npu-smi info &> /dev/null; then
    echo "NPU驱动已安装，设备状态:"
    npu-smi info
else
    echo "错误: NPU驱动未安装或npu-smi命令不可用"
    echo "请确保已正确安装华为昇腾驱动和固件"
    exit 1
fi

# 2. 设置环境变量
echo "2. 设置环境变量..."
export ASCEND_HOME=/usr/local/Ascend
export PATH=$ASCEND_HOME/ascend-toolkit/latest/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/driver/lib64:$ASCEND_HOME/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/ascend-toolkit/latest/python/site-packages:$PYTHONPATH

# NPU相关环境变量
export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
export TOOLCHAIN_HOME=$ASCEND_HOME/ascend-toolkit/latest/toolkit
export ASCEND_AICPU_PATH=$ASCEND_HOME/ascend-toolkit/latest
export ASCEND_GLOBAL_LOG_LEVEL=3  # 日志级别 (0-ERROR, 1-WARN, 2-INFO, 3-DEBUG)
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_EVENT_ENABLE=0

# 3. 安装Python依赖
echo "3. 安装Python依赖..."
pip install torch-npu -i https://pypi.org/simple/

# 4. 验证安装
echo "4. 验证torch_npu安装..."
python3 -c "
import torch
try:
    import torch_npu
    print(f'torch_npu版本: {torch_npu.__version__}')
    print(f'NPU设备数量: {torch_npu.npu.device_count()}')
    if torch_npu.npu.is_available():
        print('NPU可用: ✓')
        for i in range(torch_npu.npu.device_count()):
            print(f'  设备 {i}: {torch_npu.npu.get_device_name(i)}')
    else:
        print('NPU不可用: ✗')
except ImportError as e:
    print(f'torch_npu导入失败: {e}')
    exit(1)
except Exception as e:
    print(f'NPU检查失败: {e}')
    exit(1)
"

# 5. runtime检查
echo "5. 检查Runtime状态..."
if [ -f "/usr/local/Ascend/driver/version.info" ]; then
    echo "驱动版本信息:"
    cat /usr/local/Ascend/driver/version.info
fi

echo "=== 环境配置完成 / Environment Setup Complete ==="
echo ""
echo "使用说明:"
echo "1. 每次运行前执行: source setup_ascend.sh"
echo "2. 或将环境变量添加到 ~/.bashrc"
echo "3. 运行训练: bash speedrun_npu.sh"

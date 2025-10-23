#!/bin/bash

# 华为昇腾NPU环境修复脚本
# 解决TBE模块缺失和CANN环境问题

set -e

echo "=== 华为昇腾NPU环境修复脚本 ==="

# 1. 检查昇腾安装目录
echo "1. 检查昇腾安装目录..."
ASCEND_PATHS=(
    "/usr/local/Ascend"
    "/opt/Ascend" 
    "/home/ma-user/Ascend"
    "/root/Ascend"
)

ASCEND_HOME=""
for path in "${ASCEND_PATHS[@]}"; do
    if [ -d "$path" ]; then
        ASCEND_HOME="$path"
        echo "✅ 找到昇腾安装目录: $ASCEND_HOME"
        break
    fi
done

if [ -z "$ASCEND_HOME" ]; then
    echo "❌ 未找到昇腾安装目录，请检查CANN是否正确安装"
    exit 1
fi

# 2. 设置环境变量
echo "2. 设置昇腾环境变量..."
export ASCEND_HOME="$ASCEND_HOME"

# 查找set_env.sh文件
SET_ENV_PATHS=(
    "$ASCEND_HOME/ascend-toolkit/set_env.sh"
    "$ASCEND_HOME/ascend-toolkit/latest/set_env.sh" 
    "$ASCEND_HOME/latest/set_env.sh"
    "$ASCEND_HOME/set_env.sh"
)

SET_ENV_FILE=""
for env_file in "${SET_ENV_PATHS[@]}"; do
    if [ -f "$env_file" ]; then
        SET_ENV_FILE="$env_file"
        echo "✅ 找到环境设置文件: $SET_ENV_FILE"
        break
    fi
done

if [ -z "$SET_ENV_FILE" ]; then
    echo "❌ 未找到set_env.sh文件，手动设置环境变量..."
    # 手动设置环境变量
    export PATH="$ASCEND_HOME/ascend-toolkit/latest/bin:$PATH"
    export LD_LIBRARY_PATH="$ASCEND_HOME/driver/lib64:$ASCEND_HOME/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH"
    export PYTHONPATH="$ASCEND_HOME/ascend-toolkit/latest/python/site-packages:$PYTHONPATH"
    export ASCEND_OPP_PATH="$ASCEND_HOME/ascend-toolkit/latest/opp"
    export TOOLCHAIN_HOME="$ASCEND_HOME/ascend-toolkit/latest/toolkit"
    export ASCEND_AICPU_PATH="$ASCEND_HOME/ascend-toolkit/latest"
else
    echo "✅ 运行环境设置脚本..."
    source "$SET_ENV_FILE"
fi

# 3. 检查环境变量
echo "3. 检查环境变量设置..."
echo "ASCEND_HOME: ${ASCEND_HOME:-未设置}"
echo "PYTHONPATH: ${PYTHONPATH:-未设置}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-未设置}"

# 4. 验证NPU驱动
echo "4. 验证NPU驱动状态..."
if command -v npu-smi >/dev/null 2>&1; then
    echo "✅ npu-smi命令可用"
    npu-smi info | head -10
else
    echo "❌ npu-smi命令不可用，请检查驱动安装"
fi

# 5. 验证Python模块
echo "5. 验证Python模块..."
python3 -c "
import sys
print(f'Python路径: {sys.executable}')

# 检查torch_npu
try:
    import torch_npu
    print(f'✅ torch_npu版本: {torch_npu.__version__}')
    print(f'✅ NPU可用: {torch_npu.npu.is_available()}')
    print(f'✅ NPU数量: {torch_npu.npu.device_count()}')
except ImportError as e:
    print(f'❌ torch_npu导入失败: {e}')

# 检查TBE
try:
    import tbe
    print('✅ TBE模块可用')
except ImportError as e:
    print(f'❌ TBE模块缺失: {e}')
    print('建议: 重新安装CANN或检查PYTHONPATH设置')

# 检查ACL
try:
    import acl
    print('✅ ACL模块可用')
except ImportError as e:
    print(f'⚠️  ACL模块缺失: {e}')
"

# 6. 测试简单NPU操作
echo "6. 测试简单NPU操作..."
python3 -c "
try:
    import torch
    import torch_npu
    
    if torch_npu.npu.is_available():
        device = torch.device('npu:0')
        x = torch.randn(10, 10, device=device)
        y = torch.matmul(x, x.t())
        print('✅ NPU基本操作测试通过')
    else:
        print('❌ NPU不可用')
except Exception as e:
    print(f'❌ NPU操作测试失败: {e}')
"

echo "=== 环境检查完成 ==="
echo ""
echo "如果TBE模块仍然缺失，请尝试以下解决方案："
echo "1. 重新安装CANN工具包"
echo "2. 检查Python版本是否与CANN兼容（建议Python 3.8-3.10）"  
echo "3. 重启系统后重新设置环境"
echo "4. 联系系统管理员检查CANN安装完整性"

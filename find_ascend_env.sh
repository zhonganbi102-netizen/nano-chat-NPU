#!/bin/bash

# 自动查找Ascend环境配置脚本

echo "=== 查找Ascend环境配置 ==="

# 可能的set_env.sh路径
POSSIBLE_PATHS=(
    "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    "/usr/local/Ascend/set_env.sh"
    "/opt/ascend/ascend-toolkit/set_env.sh"
    "/usr/local/Ascend/nnae/latest/set_env.sh"
)

echo "搜索可能的set_env.sh位置..."
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "✅ 找到: $path"
        echo "ASCEND_SET_ENV_PATH=\"$path\"" > .ascend_env_path
        exit 0
    else
        echo "❌ 不存在: $path"
    fi
done

echo ""
echo "🔍 使用find命令搜索..."
SET_ENV_PATH=$(find /usr/local/Ascend -name "set_env.sh" 2>/dev/null | head -1)
if [ -n "$SET_ENV_PATH" ]; then
    echo "✅ 找到: $SET_ENV_PATH"
    echo "ASCEND_SET_ENV_PATH=\"$SET_ENV_PATH\"" > .ascend_env_path
    exit 0
fi

SET_ENV_PATH=$(find /opt -name "set_env.sh" 2>/dev/null | grep -i ascend | head -1)
if [ -n "$SET_ENV_PATH" ]; then
    echo "✅ 找到: $SET_ENV_PATH"
    echo "ASCEND_SET_ENV_PATH=\"$SET_ENV_PATH\"" > .ascend_env_path
    exit 0
fi

echo "❌ 未找到set_env.sh文件"
echo "请手动查找并设置环境变量"
exit 1

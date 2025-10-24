#!/bin/bash

# 华为服务器 maturin virtualenv 错误修复
# Fix for maturin virtualenv error on Huawei server

set -e

echo "🔧 修复maturin虚拟环境错误"

# 方案1: 创建临时虚拟环境
echo "1. 创建临时虚拟环境..."
python3 -m venv /tmp/rustbpe_venv
source /tmp/rustbpe_venv/bin/activate

echo "2. 在虚拟环境中安装maturin..."
pip install --upgrade pip setuptools wheel maturin

# 方案2: 使用maturin build + pip install
echo "3. 编译rustbpe..."
if [ -d "rustbpe" ]; then
    cd rustbpe
    echo "清理之前的构建..."
    rm -rf target/ build/ dist/ *.egg-info || true
    
    echo "使用maturin build编译..."
    maturin build --release
    
    echo "安装编译好的wheel..."
    deactivate  # 退出虚拟环境
    pip install target/wheels/*.whl --force-reinstall --root-user-action=ignore
    
    cd ..
else
    echo "❌ rustbpe目录不存在"
    deactivate
    exit 1
fi

# 清理临时虚拟环境
echo "4. 清理临时虚拟环境..."
rm -rf /tmp/rustbpe_venv

# 测试安装
echo "5. 测试rustbpe..."
python3 -c "
try:
    import rustbpe
    print('✅ rustbpe导入成功')
    tokenizer = rustbpe.Tokenizer()
    print('✅ Tokenizer创建成功')
    print('🎉 修复完成！')
except Exception as e:
    print(f'❌ 仍有问题: {e}')
    exit(1)
"

echo "✅ maturin问题已修复！"
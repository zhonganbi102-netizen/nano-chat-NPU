#!/bin/bash

# 完整的RustBPE修复脚本 - 在NPU服务器上执行
# Complete RustBPE fix script - Run on NPU server

set -e

echo "=== 修复RustBPE tokenizer问题 ==="
echo "Fixing RustBPE tokenizer issue"

# 1. 检查当前环境
echo "1. 检查当前环境..."
python3 -c "import sys; print(f'Python: {sys.version}')"
echo "当前目录: $(pwd)"

# 2. 卸载可能存在的错误版本
echo "2. 卸载可能存在的rustbpe包..."
pip uninstall -y rustbpe || echo "没有找到rustbpe包"

# 3. 检查Rust环境
echo "3. 检查Rust环境..."
if ! command -v cargo &> /dev/null; then
    echo "❌ 没有找到Rust，正在安装..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
else
    echo "✅ 找到Rust: $(cargo --version)"
fi

# 4. 安装maturin（Rust-Python绑定工具）
echo "4. 安装maturin..."
pip install maturin

# 5. 构建rustbpe
echo "5. 构建rustbpe..."
cd rustbpe || {
    echo "❌ 没有找到rustbpe目录"
    echo "正在从原始源克隆..."
    git clone https://github.com/karpathy/rustbpe.git
    cd rustbpe
}

# 6. 使用maturin构建和安装
echo "6. 使用maturin构建..."
maturin develop --release

# 7. 测试安装
echo "7. 测试rustbpe安装..."
python3 -c "
import rustbpe
print('✅ rustbpe导入成功')
tokenizer = rustbpe.Tokenizer()
print('✅ Tokenizer创建成功')
print(f'rustbpe版本: {rustbpe.__version__ if hasattr(rustbpe, \"__version__\") else \"unknown\"}')"

echo "✅ RustBPE修复完成！"
echo "现在可以重新运行训练脚本了"
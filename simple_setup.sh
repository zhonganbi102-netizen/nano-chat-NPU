#!/bin/bash

echo "=== 简化版环境安装 ==="

# 检查当前用户
echo "当前用户: $(whoami)"
echo "工作目录: $(pwd)"

# 1. 尝试安装curl和基础工具
echo "1. 安装基础工具..."

# 尝试不同的包管理器
if command -v apt-get &> /dev/null; then
    echo "使用apt-get安装..."
    apt-get update
    apt-get install -y curl wget gcc g++ make build-essential pkg-config libssl-dev
elif command -v yum &> /dev/null; then
    echo "使用yum安装..."
    yum install -y curl wget gcc gcc-c++ make openssl-devel
elif command -v dnf &> /dev/null; then
    echo "使用dnf安装..."
    dnf install -y curl wget gcc gcc-c++ make openssl-devel
else
    echo "❌ 找不到包管理器，尝试手动下载..."
fi

# 2. 手动下载并安装Rust（如果curl还是不可用）
echo "2. 安装Rust..."

if command -v curl &> /dev/null; then
    echo "使用curl安装Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
elif command -v wget &> /dev/null; then
    echo "使用wget下载Rust安装脚本..."
    wget -O rustup-init.sh https://sh.rustup.rs
    chmod +x rustup-init.sh
    ./rustup-init.sh -y
    rm rustup-init.sh
else
    echo "❌ 无法下载Rust，请检查网络连接或手动安装"
    echo "可以尝试："
    echo "  1. 安装wget或curl"
    echo "  2. 从其他机器复制Rust二进制文件"
    exit 1
fi

# 3. 加载Rust环境
echo "3. 配置Rust环境..."
source ~/.cargo/env
export PATH="$HOME/.cargo/bin:$PATH"

# 添加到shell配置文件
echo 'source ~/.cargo/env' >> ~/.bashrc
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc

# 4. 验证安装
echo "4. 验证Rust安装..."
if command -v rustc &> /dev/null; then
    echo "✅ Rust安装成功"
    rustc --version
    cargo --version
else
    echo "❌ Rust安装失败，请检查错误信息"
    exit 1
fi

# 5. 构建rustbpe
echo "5. 构建rustbpe..."
if [ ! -d "rustbpe" ]; then
    echo "❌ 找不到rustbpe目录，请确保在nanochat-npu根目录"
    exit 1
fi

cd rustbpe

# 清理并重建
rm -rf target build
pip install --upgrade maturin

echo "开始构建..."
cargo build --release

echo "安装到Python..."
maturin develop --release

cd ..

# 6. 验证
echo "6. 验证安装..."
python -c "
import rustbpe
print('✅ rustbpe导入成功')
tok = rustbpe.Tokenizer()
print('✅ Tokenizer可用')
"

echo "🎉 安装完成！"
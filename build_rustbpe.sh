#!/bin/bash

echo "=== 构建 rustbpe tokenizer ==="

# 检查rust是否安装
if ! command -v rustc &> /dev/null; then
    echo "Rust未安装，正在安装..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# 进入rustbpe目录
cd rustbpe

# 构建rustbpe
echo "正在构建rustbpe..."
cargo build --release

# 安装到Python环境
echo "安装rustbpe到Python环境..."
pip install maturin
maturin develop --release

echo "=== rustbpe构建完成 ==="
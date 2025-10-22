#!/bin/bash

echo "=== 构建 rustbpe tokenizer ==="

# 检查rust是否安装
if ! command -v rustc &> /dev/null; then
    echo "Rust未安装，正在安装..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    # 重新加载环境变量
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 确保cargo可用
if ! command -v cargo &> /dev/null; then
    echo "加载cargo环境..."
    source ~/.cargo/env
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 检查是否在nanochat-npu目录中
if [ ! -d "rustbpe" ]; then
    echo "错误: 找不到rustbpe目录，请确保在nanochat-npu根目录执行此脚本"
    exit 1
fi

# 进入rustbpe目录
cd rustbpe

# 安装maturin (在构建前)
echo "安装maturin..."
pip install maturin

# 构建rustbpe
echo "正在构建rustbpe..."
if ! cargo build --release; then
    echo "错误: cargo构建失败"
    exit 1
fi

# 安装到Python环境
echo "安装rustbpe到Python环境..."
if ! maturin develop --release; then
    echo "错误: maturin安装失败"
    exit 1
fi

# 验证安装
cd ..
echo "验证rustbpe安装..."
python -c "
try:
    import rustbpe
    print('rustbpe导入成功')
    print(f'可用属性: {[attr for attr in dir(rustbpe) if not attr.startswith(\"_\")]}')
    
    # 测试Tokenizer类
    if hasattr(rustbpe, 'Tokenizer'):
        print('✅ Tokenizer类可用')
    else:
        print('❌ Tokenizer类不可用')
        
except ImportError as e:
    print(f'❌ rustbpe导入失败: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "=== rustbpe构建和验证完成 ==="
else
    echo "❌ rustbpe验证失败"
    exit 1
fi
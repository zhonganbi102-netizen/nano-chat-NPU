#!/bin/bash

echo "=== 修复Rust环境并构建rustbpe ==="

# 确保在rustbpe目录
if [ ! -f "Cargo.toml" ]; then
    echo "❌ 请在rustbpe目录中运行此脚本"
    echo "当前目录: $(pwd)"
    echo "使用方法: cd rustbpe && ../fix_cargo_env.sh"
    exit 1
fi

echo "✅ 当前在rustbpe目录中"

# 多种方式加载Rust环境
echo "加载Rust环境变量..."

# 方法1: 直接设置PATH
export PATH="$HOME/.cargo/bin:$PATH"

# 方法2: 加载cargo环境文件
if [ -f "$HOME/.cargo/env" ]; then
    echo "加载 ~/.cargo/env"
    source "$HOME/.cargo/env"
fi

# 方法3: 查找cargo二进制文件
CARGO_PATHS=(
    "$HOME/.cargo/bin/cargo"
    "/usr/local/bin/cargo"
    "/usr/bin/cargo"
)

for cargo_path in "${CARGO_PATHS[@]}"; do
    if [ -x "$cargo_path" ]; then
        echo "找到cargo: $cargo_path"
        export PATH="$(dirname $cargo_path):$PATH"
        break
    fi
done

# 验证cargo是否可用
echo "验证cargo..."
if command -v cargo &> /dev/null; then
    echo "✅ cargo可用"
    cargo --version
else
    echo "❌ cargo仍然不可用，尝试手动查找..."
    
    # 手动查找cargo
    find /root -name "cargo" -type f 2>/dev/null | head -5
    find /home -name "cargo" -type f 2>/dev/null | head -5
    
    echo "请检查Rust安装是否完整"
    exit 1
fi

# 验证rustc
if command -v rustc &> /dev/null; then
    echo "✅ rustc可用"
    rustc --version
else
    echo "❌ rustc不可用"
    exit 1
fi

# 清理之前的构建
echo "清理之前的构建..."
if [ -d "target" ]; then
    rm -rf target
    echo "删除target目录"
fi

if [ -d "build" ]; then
    rm -rf build
    echo "删除build目录"
fi

# 更新maturin
echo "确保maturin最新版本..."
pip install --upgrade maturin setuptools-rust

# 开始构建
echo "开始cargo构建..."
echo "命令: cargo build --release"

if cargo build --release; then
    echo "✅ cargo构建成功"
else
    echo "❌ cargo构建失败"
    echo "检查错误信息..."
    
    # 显示详细错误
    echo "尝试详细构建..."
    cargo build --release --verbose
    exit 1
fi

# 使用maturin安装
echo "使用maturin安装到Python..."
if maturin develop --release; then
    echo "✅ maturin安装成功"
else
    echo "❌ maturin安装失败"
    exit 1
fi

# 返回上级目录并验证
cd ..
echo "验证rustbpe安装..."
python -c "
try:
    import rustbpe
    print('✅ rustbpe导入成功')
    
    # 检查可用属性
    attrs = [attr for attr in dir(rustbpe) if not attr.startswith('_')]
    print(f'可用属性: {attrs}')
    
    # 测试Tokenizer
    if hasattr(rustbpe, 'Tokenizer'):
        tok = rustbpe.Tokenizer()
        print('✅ Tokenizer类可用')
        
        # 简单编码测试
        result = tok.encode('Hello world')
        print(f'✅ 编码测试成功: {result}')
    else:
        print('❌ Tokenizer类不可用')
        
except Exception as e:
    print(f'❌ 验证失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 rustbpe构建和安装完成！"
    echo "现在可以运行训练:"
    echo "  ./speedrun_npu.sh"
else
    echo ""
    echo "❌ 验证失败，请检查错误信息"
    exit 1
fi
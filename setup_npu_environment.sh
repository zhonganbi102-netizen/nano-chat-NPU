#!/bin/bash

echo "=== NPU服务器环境安装脚本 ==="

# 检测系统类型
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    echo "检测到操作系统: $OS"
else
    echo "无法检测操作系统类型"
    exit 1
fi

# 函数：安装基础工具
install_basic_tools() {
    echo "安装基础工具..."
    
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        apt-get update
        apt-get install -y curl wget gcc g++ make build-essential pkg-config libssl-dev
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        yum update -y
        yum groupinstall -y "Development Tools"
        yum install -y curl wget gcc gcc-c++ make openssl-devel
    elif command -v dnf &> /dev/null; then
        # Fedora
        dnf update -y
        dnf groupinstall -y "Development Tools"
        dnf install -y curl wget gcc gcc-c++ make openssl-devel
    else
        echo "❌ 不支持的包管理器，请手动安装curl、gcc、make等工具"
        exit 1
    fi
}

# 函数：安装Rust
install_rust() {
    echo "安装Rust..."
    
    # 创建临时目录
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # 下载Rust安装脚本
    if command -v wget &> /dev/null; then
        wget https://sh.rustup.rs -O rustup-init.sh
    else
        echo "❌ 无法下载Rust安装脚本"
        exit 1
    fi
    
    # 安装Rust
    chmod +x rustup-init.sh
    ./rustup-init.sh -y --default-toolchain stable
    
    # 清理临时文件
    cd /
    rm -rf "$TEMP_DIR"
    
    # 加载环境变量
    source ~/.cargo/env
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # 验证安装
    if command -v rustc &> /dev/null; then
        echo "✅ Rust安装成功"
        rustc --version
        cargo --version
    else
        echo "❌ Rust安装失败"
        exit 1
    fi
}

# 函数：构建rustbpe
build_rustbpe() {
    echo "构建rustbpe tokenizer..."
    
    # 确保在正确目录
    if [ ! -d "rustbpe" ]; then
        echo "❌ 找不到rustbpe目录"
        exit 1
    fi
    
    # 加载Rust环境
    source ~/.cargo/env
    export PATH="$HOME/.cargo/bin:$PATH"
    
    cd rustbpe
    
    # 清理旧构建
    if [ -d "target" ]; then
        rm -rf target
    fi
    
    # 安装maturin
    pip install --upgrade maturin setuptools-rust
    
    # 构建
    echo "开始cargo构建..."
    if ! cargo build --release; then
        echo "❌ cargo构建失败"
        exit 1
    fi
    
    # 安装到Python
    echo "安装到Python环境..."
    if ! maturin develop --release; then
        echo "❌ maturin安装失败"
        exit 1
    fi
    
    cd ..
    
    # 验证
    echo "验证rustbpe安装..."
    python -c "
try:
    import rustbpe
    print('✅ rustbpe导入成功')
    
    # 测试Tokenizer类
    tok = rustbpe.Tokenizer()
    print('✅ Tokenizer类可用')
    
    # 简单功能测试
    result = tok.encode('Hello')
    print(f'✅ 编码测试成功: {result}')
    
except Exception as e:
    print(f'❌ 验证失败: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo "🎉 rustbpe构建和安装完成！"
    else
        echo "❌ rustbpe验证失败"
        exit 1
    fi
}

# 主执行流程
main() {
    echo "开始环境配置..."
    
    # 检查是否为root用户
    if [ "$EUID" -ne 0 ]; then
        echo "请以root权限运行此脚本: sudo $0"
        exit 1
    fi
    
    # 1. 安装基础工具
    if ! command -v curl &> /dev/null; then
        install_basic_tools
    else
        echo "✅ 基础工具已安装"
    fi
    
    # 2. 安装Rust
    if ! command -v rustc &> /dev/null; then
        install_rust
    else
        echo "✅ Rust已安装"
        rustc --version
    fi
    
    # 3. 构建rustbpe
    build_rustbpe
    
    echo ""
    echo "🎉 全部安装完成！现在可以运行训练："
    echo "  ./speedrun_npu.sh"
}

# 运行主函数
main "$@"
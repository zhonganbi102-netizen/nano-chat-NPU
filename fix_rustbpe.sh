#!/bin/bash

echo "=== RustBPE 故障排除脚本 ==="

# 函数：检查Python环境
check_python_env() {
    echo "1. 检查Python环境..."
    python --version
    pip --version
    
    echo "当前Python路径:"
    which python
    
    echo "已安装的相关包:"
    pip list | grep -E "(maturin|rustbpe|torch)"
}

# 函数：检查Rust环境  
check_rust_env() {
    echo -e "\n2. 检查Rust环境..."
    if command -v rustc &> /dev/null; then
        rustc --version
        cargo --version
        echo "Rust路径: $(which rustc)"
    else
        echo "❌ Rust未安装"
        return 1
    fi
}

# 函数：清理之前的构建
clean_build() {
    echo -e "\n3. 清理之前的构建..."
    cd rustbpe
    if [ -d "target" ]; then
        echo "删除target目录..."
        rm -rf target
    fi
    
    if [ -d "build" ]; then
        echo "删除build目录..."
        rm -rf build
    fi
    
    # 卸载已安装的rustbpe
    echo "卸载旧的rustbpe..."
    pip uninstall rustbpe -y 2>/dev/null || true
    cd ..
}

# 函数：重新构建rustbpe
rebuild_rustbpe() {
    echo -e "\n4. 重新构建rustbpe..."
    
    # 确保环境变量正确
    if [ -f ~/.cargo/env ]; then
        source ~/.cargo/env
    fi
    export PATH="$HOME/.cargo/bin:$PATH"
    
    cd rustbpe
    
    # 安装/更新maturin
    echo "安装maturin..."
    pip install --upgrade maturin
    
    # 构建
    echo "开始构建..."
    if cargo build --release --verbose; then
        echo "✅ cargo构建成功"
    else
        echo "❌ cargo构建失败"
        return 1
    fi
    
    # 安装
    echo "安装到Python环境..."
    if maturin develop --release --verbose; then
        echo "✅ maturin安装成功"
    else
        echo "❌ maturin安装失败"
        return 1
    fi
    
    cd ..
}

# 函数：验证安装
verify_installation() {
    echo -e "\n5. 验证安装..."
    
    python -c "
import sys
print(f'Python版本: {sys.version}')
print(f'Python路径: {sys.executable}')

try:
    import rustbpe
    print('✅ rustbpe导入成功')
    
    # 检查所有可用属性
    attrs = [attr for attr in dir(rustbpe) if not attr.startswith('_')]
    print(f'可用属性: {attrs}')
    
    # 检查Tokenizer类
    if hasattr(rustbpe, 'Tokenizer'):
        print('✅ Tokenizer类可用')
        
        # 尝试创建tokenizer实例
        try:
            tok = rustbpe.Tokenizer()
            print('✅ Tokenizer实例创建成功')
        except Exception as e:
            print(f'❌ Tokenizer实例创建失败: {e}')
    else:
        print('❌ Tokenizer类不可用')
        print('可能的解决方案:')
        print('  1. 检查Cargo.toml中的pyo3配置')
        print('  2. 重新构建rustbpe')
        print('  3. 检查Python绑定代码')
        
except ImportError as e:
    print(f'❌ rustbpe导入失败: {e}')
    print('可能的解决方案:')
    print('  1. 重新运行 ./build_rustbpe.sh')
    print('  2. 检查Python环境是否正确')
    print('  3. 确保maturin develop成功执行')
"
}

# 函数：提供解决方案建议
suggest_solutions() {
    echo -e "\n6. 常见问题解决方案:"
    echo "问题1: 'module rustbpe has no attribute Tokenizer'"
    echo "  解决: 检查rustbpe/src/lib.rs中是否正确导出了Tokenizer类"
    echo ""
    echo "问题2: ImportError: No module named 'rustbpe'"  
    echo "  解决: 重新运行 maturin develop --release"
    echo ""
    echo "问题3: cargo构建失败"
    echo "  解决: 检查Rust版本，更新到最新稳定版"
    echo ""
    echo "问题4: maturin develop失败"
    echo "  解决: pip install --upgrade maturin pyo3"
}

# 主执行流程
main() {
    check_python_env
    
    if ! check_rust_env; then
        echo "请先安装Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi
    
    clean_build
    
    if rebuild_rustbpe; then
        verify_installation
    else
        echo -e "\n❌ 构建失败，查看上面的错误信息"
        suggest_solutions
        exit 1
    fi
    
    suggest_solutions
}

# 运行主函数
main "$@"
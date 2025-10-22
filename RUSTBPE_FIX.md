# RustBPE Tokenizer 问题快速修复指南

## 问题描述
运行训练时出现错误：
```
AttributeError: module 'rustbpe' has no attribute 'Tokenizer'
```

## 快速解决方案

### 方案1：使用自动修复脚本（推荐）
```bash
cd /mnt/linxid615/bza/nanochat-npu
git pull  # 获取最新的修复脚本
chmod +x fix_rustbpe.sh
./fix_rustbpe.sh
```

### 方案2：手动步骤修复
```bash
# 1. 确保Rust环境正确
source ~/.cargo/env
export PATH="$HOME/.cargo/bin:$PATH"

# 2. 清理旧的构建
cd rustbpe
rm -rf target build
pip uninstall rustbpe -y

# 3. 重新构建
cargo clean
cargo build --release --verbose

# 4. 重新安装到Python
pip install --upgrade maturin
maturin develop --release --verbose

# 5. 验证安装
cd ..
python -c "
import rustbpe
print('✅ rustbpe导入成功')
tok = rustbpe.Tokenizer()
print('✅ Tokenizer类可用')
"
```

## 常见问题和解决方案

### 问题1: Rust未安装或环境变量错误
```bash
# 安装Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# 验证安装
rustc --version
cargo --version
```

### 问题2: maturin版本过旧
```bash
pip install --upgrade maturin pyo3
```

### 问题3: cargo构建失败
```bash
# 更新Rust到最新版本
rustup update

# 清理缓存
cargo clean
rm -rf ~/.cargo/registry/cache
```

### 问题4: Python环境冲突
```bash
# 确认当前Python环境
which python
python --version

# 如果使用conda，确保在正确环境中
conda activate your_env_name
```

## 验证修复成功

运行以下命令验证rustbpe已正确安装：
```bash
python -c "
import rustbpe
print('Module imported successfully')
print('Available attributes:', [attr for attr in dir(rustbpe) if not attr.startswith('_')])

try:
    tok = rustbpe.Tokenizer()
    print('✅ Tokenizer class works!')
    
    # 测试基本功能
    result = tok.encode('Hello world')
    print(f'Encode test: {result}')
    
except Exception as e:
    print(f'❌ Error: {e}')
"
```

## 修复后继续训练

一旦rustbpe修复成功，重新运行训练：
```bash
./speedrun_npu.sh
```

## 如果仍然有问题

1. **检查完整错误日志**：
   ```bash
   ./fix_rustbpe.sh 2>&1 | tee rustbpe_debug.log
   ```

2. **验证依赖环境**：
   ```bash
   pip list | grep -E "(maturin|pyo3|setuptools-rust)"
   ```

3. **清理重新开始**：
   ```bash
   pip uninstall rustbpe maturin -y
   rm -rf rustbpe/target rustbpe/build
   pip install maturin
   ./build_rustbpe.sh
   ```

## 备选方案

如果rustbpe持续有问题，可以暂时使用纯Python tokenizer：

修改 `nanochat/tokenizer.py` 中的导入：
```python
try:
    import rustbpe
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("警告: rustbpe不可用，使用纯Python实现")
```

这样可以先进行其他训练步骤的测试。
#!/bin/bash

/**
 * @File: quick_start_platform.sh
 * @Author: 刘世宇
 * @Email: liusy@zhihuiyunxing.com
 * @Date: 2025-10-23
 * @Description: 华为平台快速启动脚本 - 自动处理rustbpe编译问题
 * @Company: 智慧云行（成都）科技有限公司
 * @Version: 1.0.0
 */

set -e

echo "🔥 华为NPU平台 - nanochat 完整FineWeb训练启动脚本 🔥"
echo ""

# 1. 检查并安装 Rust（如果需要）
echo "1️⃣ 检查 Rust 环境..."
if ! command -v cargo &> /dev/null; then
    echo "⚠️  未检测到 Rust，正在安装..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    echo "✅ Rust 安装完成"
else
    echo "✅ Rust 已安装: $(rustc --version)"
fi

# 2. 安装项目（编译 rustbpe）
echo ""
echo "2️⃣ 编译安装 nanochat（包含 rustbpe 扩展）..."
pip install maturin
pip install -e .
echo "✅ nanochat 安装完成"

# 3. 检查数据集
echo ""
echo "3️⃣ 检查 FineWeb 数据集..."
data_files=$(ls base_data/shard_*.parquet 2>/dev/null | wc -l || echo "0")
if [ "$data_files" -lt 100 ]; then
    echo "⚠️  FineWeb数据文件不足($data_files个)"
    echo "正在下载数据集（这可能需要10-20分钟）..."
    chmod +x download_fineweb_data.sh
    bash download_fineweb_data.sh
else
    echo "✅ FineWeb数据文件: $data_files 个"
fi

# 4. 启动完整训练
echo ""
echo "4️⃣ 启动完整 FineWeb 4NPU 训练..."
chmod +x full_fineweb_4npu_train.sh emergency_npu_cleanup.sh
bash full_fineweb_4npu_train.sh

echo ""
echo "🎉 训练启动成功！"


#!/bin/bash

echo "=== 安装nanochat-npu所需的Python依赖 ==="

# 基础依赖
echo "1. 安装基础依赖..."
pip install pyarrow pandas

# 机器学习依赖
echo "2. 安装机器学习依赖..."
pip install numpy

# Web服务依赖
echo "3. 安装Web服务依赖..."
pip install fastapi uvicorn jinja2

# 其他可能需要的依赖
echo "4. 安装其他依赖..."
pip install requests tqdm

# wandb用于实验跟踪（可选）
echo "5. 安装wandb（可选，用于实验跟踪）..."
pip install wandb

# 安装项目本身
echo "6. 安装nanochat-npu..."
pip install -e .

echo "=== 依赖安装完成 ==="
echo "现在可以运行: bash speedrun_npu.sh"
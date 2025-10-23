# FineWeb数据集NPU训练指南

## 🚀 概述

本指南提供在华为昇腾NPU上使用FineWeb教育数据集进行完整nanochat模型训练的解决方案。

## 📋 训练流程

### 步骤1: 数据下载

```bash
# 下载200个FineWeb数据文件 (~20-40GB)
./download_fineweb_data.sh
```

**特点:**
- 自动检查磁盘空间
- 并行下载 (3个并发)
- 断点续传支持
- 进度显示
- 重试机制

### 步骤2: 快速测试 (推荐)

```bash
# 快速验证环境和数据
./quick_fineweb_test.sh
```

**配置:**
- 模型深度: 8层
- 训练步数: 100步
- 验证环境和4NPU设置

### 步骤3: 完整训练

```bash
# 大规模base model训练
./train_with_fineweb.sh
```

**配置:**
- 模型深度: 12层
- 总批次大小: 262,144 tokens
- 训练步数: 5,000步
- 4NPU分布式训练

## 📊 预期性能

### 数据规模
- **文件数量**: 200个parquet文件
- **数据大小**: ~20-40GB
- **token数量**: ~13亿tokens (估算)

### 训练性能
- **4NPU速度**: 70-100k tok/sec
- **预计时间**: 
  - 快速测试: ~5-10分钟
  - 完整训练: ~3-4小时

### 模型质量
- **参数量**: ~300M (depth=12)
- **序列长度**: 2048
- **词汇量**: 65,536

## 🛠️ 技术特性

### NPU优化
- ✅ **4NPU分布式**: torchrun + HCCL
- ✅ **优化器兼容**: AdamW替代Muon
- ✅ **内存优化**: 批次大小和内存分片优化
- ✅ **编译禁用**: 避免NPU编译问题

### 数据处理
- ✅ **自动tokenizer训练**: 基于FineWeb数据
- ✅ **分布式数据加载**: 高效并行处理
- ✅ **动态批处理**: 优化内存使用

## 📁 文件结构

```
nanochat-npu/
├── download_fineweb_data.sh    # 数据下载脚本
├── quick_fineweb_test.sh       # 快速测试脚本  
├── train_with_fineweb.sh       # 完整训练脚本
├── base_data/                  # 数据目录
│   ├── shard_00005.parquet
│   ├── shard_00006.parquet
│   └── ...
└── ~/.cache/nanochat/          # 模型输出目录
    ├── tokenizer/              # tokenizer文件
    └── base_checkpoints/       # 模型检查点
        └── fineweb_base_d12/   # 完整训练模型
```

## 🔧 故障排除

### 数据下载问题
```bash
# 检查网络连接
curl -I https://hf-mirror.com

# 手动重试下载
export HF_ENDPOINT=https://hf-mirror.com
./download_fineweb_data.sh
```

### NPU环境问题
```bash
# 环境诊断
./fix_ascend_env.sh

# 清理NPU内存
./emergency_npu_cleanup.sh
```

### 训练错误
```bash
# 检查NPU状态
npu-smi info

# 验证torch_npu
python3 -c "import torch_npu; print(torch_npu.npu.is_available())"

# 先运行快速测试
./quick_fineweb_test.sh
```

## 📈 训练监控

### 关键指标
- **tok/sec**: 应该在70-100k范围
- **loss**: 应该稳定下降
- **memory**: NPU内存使用应稳定
- **mfu**: 模型FLOP利用率 ~1.3-1.4%

### 日志示例
```
step 01000/05000 (20.00%) | loss: 3.456 | lrm: 1.00 | dt: 180.5ms | tok/sec: 90,847 | mfu: 1.35 | total time: 3.2m
```

## 🎯 下一步

训练完成后可以：

1. **测试模型**:
   ```bash
   python -m scripts.chat_cli
   ```

2. **启动Web服务**:
   ```bash
   python -m scripts.chat_web
   ```

3. **继续训练流程**:
   - Mid-training
   - Chat SFT
   - Chat RL

## 💡 优化建议

### 性能优化
- 根据NPU内存调整`device_batch_size`
- 监控内存使用，调整`PYTORCH_NPU_ALLOC_CONF`
- 使用更多数据文件提升质量

### 质量优化  
- 增加训练步数到10,000+
- 调整学习率调度
- 使用更大模型深度 (16-20层)

---

🎉 **准备开始您的FineWeb NPU训练之旅！**

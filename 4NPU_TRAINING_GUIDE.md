# 4NPU分布式训练指南

## 📋 概述

本指南提供了在华为昇腾NPU上进行4卡分布式训练的完整解决方案。

## 🚀 快速开始

### 1. 环境检查

在开始训练前，请运行环境检查脚本：

```bash
# 设置环境变量
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 运行检查脚本
python check_4npu_setup.py
```

### 2. 选择训练脚本

根据您的需求选择合适的训练脚本：

#### 🟢 保守配置（推荐首次使用）
```bash
chmod +x npu_4gpu_conservative.sh
./npu_4gpu_conservative.sh
```

**特点：**
- 更小的batch size (device_batch_size=2)
- 更低的学习率 (降低30%)
- 更保守的内存设置
- 适合首次尝试或系统资源有限时使用

#### 🔵 标准配置
```bash
chmod +x npu_4gpu_train.sh
./npu_4gpu_train.sh
```

**特点：**
- 标准batch size (device_batch_size=4)  
- 标准学习率
- 更大的模型深度 (depth=8)
- 适合系统资源充足时使用

## 📊 配置对比

| 配置项 | 保守配置 | 标准配置 |
|-------|---------|---------|
| 模型深度 | 6 | 8 |
| 设备batch size | 2 | 4 |
| 总batch size | 32,768 | 65,536 |
| 学习率缩放 | 0.7x | 1.0x |
| 训练步数 | 500 | 1000 |
| 内存设置 | 32MB分片 | 默认 |

## 🔧 故障排除

### 常见问题

#### 1. HCCL初始化失败
```bash
# 检查环境变量
echo $MASTER_ADDR  # 应该是 127.0.0.1
echo $MASTER_PORT  # 应该是 29500 或其他可用端口
echo $WORLD_SIZE   # 应该是 4

# 检查端口占用
netstat -tuln | grep 29500

# 尝试不同端口
export MASTER_PORT=29501
```

#### 2. NPU内存不足
```bash
# 运行清理脚本
./emergency_npu_cleanup.sh

# 使用保守配置
./npu_4gpu_conservative.sh

# 或进一步降低batch size
# 修改脚本中的 --device_batch_size=1
```

#### 3. 进程组超时
```bash
# 增加超时时间
export TORCH_DISTRIBUTED_TIMEOUT=1800

# 检查网络连接
ping 127.0.0.1
```

### 环境重置

如果遇到严重问题，可以完全重置环境：

```bash
# 1. 清理所有Python进程
pkill -f python
pkill -f torchrun

# 2. 清理NPU内存
./emergency_npu_cleanup.sh

# 3. 重新设置环境变量
unset WORLD_SIZE MASTER_ADDR MASTER_PORT RANK LOCAL_RANK
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 4. 等待10秒后重新尝试
sleep 10
```

## 📈 性能监控

训练过程中可以监控以下指标：

```bash
# 监控NPU使用率
watch -n 1 'npu-smi info'

# 监控训练日志中的关键指标
# - tok/sec: 每秒处理的token数
# - mfu: 模型FLOPs利用率
# - loss: 训练损失
# - dt: 单步耗时
```

预期性能指标：
- **单NPU**: ~30-35k tok/sec
- **4NPU**: ~120-140k tok/sec (理想情况)
- **MFU**: 1.5-2.0%

## 🎯 最佳实践

1. **首次运行**: 始终使用保守配置测试
2. **监控内存**: 关注NPU内存使用情况
3. **渐进调优**: 成功后再逐步增加batch size
4. **备份检查点**: 定期保存训练进度
5. **日志记录**: 保存完整的训练日志

## 📝 日志分析

成功的分布式训练日志应该显示：

```
Using Ascend NPU, device count: 1  # 每个进程看到1个NPU
Distributed world size: 4          # 全局有4个进程
Total batch size 32,768 => gradient accumulation steps: 4  # 梯度累积
✅ NPU 4GPU分布式优化器补丁已应用
```

如果看到错误，请参考故障排除章节或查看完整日志。

## 🔗 相关文件

- `npu_4gpu_conservative.sh`: 保守4NPU训练脚本
- `npu_4gpu_train.sh`: 标准4NPU训练脚本  
- `check_4npu_setup.py`: 环境检查脚本
- `emergency_npu_cleanup.sh`: 环境清理脚本
- `npu_simple_train.sh`: 单NPU训练脚本（后备方案）

---

💡 **提示**: 如果4NPU训练遇到问题，可以回退到单NPU训练进行调试。

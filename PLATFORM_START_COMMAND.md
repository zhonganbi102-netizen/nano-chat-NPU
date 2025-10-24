/**
 * @File: PLATFORM_START_COMMAND.md
 * @Author: 刘世宇
 * @Email: liusy@zhihuiyunxing.com
 * @Date: 2025-10-23
 * @Description: 华为平台启动命令指南 - 用于在华为NPU平台上启动完整FineWeb训练
 * @Company: 智慧云行（成都）科技有限公司
 * @Version: 1.0.0
 */

# 华为NPU平台启动命令

## 📋 方案一：最简启动命令（推荐）

```bash
pip install datasets fastapi files-to-prompt numpy==1.26.4 psutil regex tiktoken tokenizers uvicorn wandb && cd /mnt/linxid615/bza/nanochat-npu && chmod +x full_fineweb_4npu_train.sh emergency_npu_cleanup.sh && bash full_fineweb_4npu_train.sh
```

## 📋 方案二：分步启动命令（便于调试）

### 步骤1：安装Python依赖
```bash
pip install datasets>=4.0.0 fastapi>=0.117.1 files-to-prompt>=0.6 numpy==1.26.4 psutil>=7.1.0 regex tiktoken>=0.11.0 tokenizers>=0.22.0 uvicorn>=0.36.0 wandb>=0.21.3
```

### 步骤2：进入项目目录并设置权限
```bash
cd /mnt/linxid615/bza/nanochat-npu && chmod +x full_fineweb_4npu_train.sh emergency_npu_cleanup.sh download_fineweb_data.sh
```

### 步骤3：下载数据集（如果还没下载）
```bash
bash download_fineweb_data.sh
```

### 步骤4：启动完整训练
```bash
bash full_fineweb_4npu_train.sh
```

## 📋 方案三：一键完整命令（包含数据下载）

```bash
pip install datasets fastapi files-to-prompt numpy==1.26.4 psutil regex tiktoken tokenizers uvicorn wandb && cd /mnt/linxid615/bza/nanochat-npu && chmod +x *.sh && bash download_fineweb_data.sh && bash full_fineweb_4npu_train.sh
```

## ⚠️ 注意事项

### 1. torch 和 torch_npu
华为平台的镜像**通常已包含** `torch` 和 `torch_npu`，**无需重复安装**。
如果平台要求或确实缺失，请咨询平台管理员获取正确的安装源。

### 2. 项目目录路径
- 当前使用的路径：`/mnt/linxid615/bza/nanochat-npu`
- 这是你的实际项目路径

### 3. 数据集说明
- `download_fineweb_data.sh` 会下载约 200-300 个数据分片（~30-45GB）
- 如果已经下载过数据，可以跳过步骤3

### 4. 训练时间估算
- 完整FineWeb数据集训练预计需要 **2-3小时**
- 使用4个NPU（910B3）
- 训练4000步，批次大小131072

## 🎯 推荐使用的启动命令

### 如果数据集已下载：
```bash
pip install datasets fastapi files-to-prompt numpy==1.26.4 psutil regex tiktoken tokenizers uvicorn wandb && cd /mnt/linxid615/bza/nanochat-npu && chmod +x full_fineweb_4npu_train.sh emergency_npu_cleanup.sh && bash full_fineweb_4npu_train.sh
```

### 如果需要先下载数据集：
```bash
pip install datasets fastapi files-to-prompt numpy==1.26.4 psutil regex tiktoken tokenizers uvicorn wandb && cd /mnt/linxid615/bza/nanochat-npu && chmod +x *.sh && bash download_fineweb_data.sh && bash full_fineweb_4npu_train.sh
```

## 📊 训练完成后

训练完成后，模型会保存在：
```
~/.cache/nanochat/base_checkpoints/full_fineweb_dataset_d12/
```

## 🔄 后续步骤（完成base-train后）

完成base-train后，你需要依次进行：
1. **mid-train**：中期训练
2. **chat-sft**：监督微调
3. **chat-rl**：强化学习

这些步骤的数据集需要分别下载，具体命令待base-train完成后再提供。

## 💡 小贴士

1. **wandb登录**：训练开始时会提示选择wandb选项，选择`3`（不可视化）可以离线运行
2. **查看日志**：训练过程中会实时输出loss和性能指标
3. **中断恢复**：如果训练中断，可以重新运行脚本继续训练（会从checkpoint恢复）


# NanoChat on Huawei Ascend NPU

这是NanoChat在华为昇腾NPU上的运行指南。

## 环境要求

### 硬件要求
- 华为昇腾NPU卡 (Ascend 910/310P等)
- 至少32GB系统内存
- 推荐多卡配置以提升训练速度

### 软件要求
- Ubuntu 18.04/20.04/22.04
- Python 3.8-3.11
- 华为昇腾驱动和固件 (CANN 7.0+)
- PyTorch >= 2.1.0
- torch_npu

## 安装步骤

### 1. 安装华为昇腾驱动和CANN

请参考华为官方文档安装：
- [昇腾驱动安装指南](https://www.hiascend.com/document)
- [CANN软件包安装](https://www.hiascend.com/software/cann)

```bash
# 检查NPU设备
npu-smi info
```

### 2. 安装Python依赖

```bash
# 克隆项目
git clone <nanochat-repo>
cd nanochat

# 创建虚拟环境
python -m venv venv_npu
source venv_npu/bin/activate

# 安装基础依赖
pip install -e .

# 安装NPU支持
pip install torch-npu -i https://pypi.org/simple/
```

### 3. 环境配置

```bash
# 配置环境
bash setup_ascend.sh

# 或手动设置环境变量
export ASCEND_HOME=/usr/local/Ascend
export PATH=$ASCEND_HOME/ascend-toolkit/latest/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/driver/lib64:$ASCEND_HOME/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$ASCEND_HOME/ascend-toolkit/latest/python/site-packages:$PYTHONPATH
```

### 4. 环境验证

```bash
# 运行环境检查
python check_npu.py
```

## 运行训练

### 快速开始

```bash
# source环境变量
source setup_ascend.sh

# 运行完整训练管道
bash speedrun_npu.sh
```

### 分步训练

1. **训练tokenizer**
   ```bash
   python -m scripts.tok_train
   ```

2. **Base模型预训练**
   ```bash
   # 单卡训练
   python -m scripts.base_train --depth=12 --device_batch_size=16
   
   # 多卡训练 (推荐)
   torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
     --depth=12 --device_batch_size=16 --total_batch_size=262144
   ```

3. **中间训练 (Midtraining)**
   ```bash
   torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
     --device_batch_size=16 --total_batch_size=262144
   ```

4. **监督微调 (SFT)**
   ```bash
   torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
     --device_batch_size=8 --target_examples_per_step=32
   ```

5. **强化学习 (RL, 可选)**
   ```bash
   torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- \
     --device_batch_size=4 --examples_per_step=16
   ```

## 性能优化

### NPU特定优化

```bash
# 设置NPU设备可见性
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 日志级别调整
export ASCEND_GLOBAL_LOG_LEVEL=3

# 内存优化
export ASCEND_LAUNCH_BLOCKING=1
```

### 训练参数调优

针对NPU的推荐参数：

```bash
# 对于Ascend 910 (32GB显存)
--device_batch_size=16    # 根据显存调整
--total_batch_size=262144 # 保持不变
--max_seq_len=2048        # 根据内存调整

# 对于多卡训练
--nproc_per_node=8        # NPU卡数
```

## 常见问题

### 1. NPU不可用
```
错误: NPU不可用
解决: 
- 检查驱动安装: npu-smi info
- 检查CANN版本匹配
- 验证torch_npu安装
```

### 2. 内存不足
```
错误: CUDA out of memory
解决:
- 减小batch_size
- 增加gradient_accumulation_steps
- 使用更小的模型深度
```

### 3. 分布式训练失败
```
错误: HCCL initialization failed
解决:
- 检查多卡环境配置
- 验证网络连通性
- 确认HCCL库安装
```

### 4. 性能问题
```
问题: 训练速度慢
优化:
- 使用bf16混合精度
- 启用torch.compile
- 调整数据加载器线程数
```

## 监控和调试

### 性能监控
```bash
# NPU使用情况
npu-smi info

# 实时监控
watch -n 1 npu-smi info
```

### 日志调试
```bash
# 启用详细日志
export ASCEND_GLOBAL_LOG_LEVEL=0  # ERROR
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 查看日志
tail -f $ASCEND_HOME/log/plog/plog-*.log
```

## 与CUDA版本差异

主要差异点：
1. 设备类型: `cuda` → `npu`
2. 内存管理: `torch.cuda.*` → `torch_npu.npu.*`
3. 分布式后端: `nccl` → `hccl`
4. 环境变量: `CUDA_VISIBLE_DEVICES` → `ASCEND_RT_VISIBLE_DEVICES`

## 支持和反馈

如遇到问题，请提供：
1. NPU型号和驱动版本
2. CANN版本信息
3. torch_npu版本
4. 完整的错误日志

## 参考链接

- [华为昇腾官方文档](https://www.hiascend.com/document)
- [torch_npu GitHub](https://github.com/Ascend/pytorch)
- [CANN开发者社区](https://www.hiascend.com/forum)

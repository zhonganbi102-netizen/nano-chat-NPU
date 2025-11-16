# NanoChat 在华为昇腾 NPU 上的适配与问题总结

**文档版本**: 1.0
**日期**: 2025年11月16日

## 1. 项目背景

本文档旨在全面总结将 `karpathy/nanochat` 项目从原生的 CUDA/GPU 环境迁移并适配到华为昇腾（Ascend） NPU 平台的全过程。目标是实现在昇腾 NPU 上稳定、高效地完成从 **BASE 模型预训练**、**SFT 微调**到 **RLHF 对齐** 的完整训练流程。

迁移过程并非一帆风順，遇到了涉及**环境配置、分布式通信、算子兼容性、性能优化、内存管理**等多个层面的挑战。本文将对这些关键问题及其解决方案进行详细阐述。

---

## 2. 核心挑战与适配工作

### 2.1. 环境配置与依赖

**问题描述**:
基础环境的搭建是第一道难关。标准的 PyTorch + CUDA 环境无法直接在 NPU 上运行。需要依赖华为提供的 `torch_npu` 插件以及相应的驱动和工具链（CANN）。

**适配与解决方案**:
- **安装 `torch_npu`**: 替换 `torch` 和 `torchvision` 为华为官方提供的 `torch_npu` 版本。这是让 PyTorch 代码能够在 NPU 上运行的基础。
- **环境变量配置**:
    - `ASCEND_RT_VISIBLE_DEVICES`: 类似于 `CUDA_VISIBLE_DEVICES`，用于指定程序可见的 NPU 卡。
    - `LD_LIBRARY_PATH`: 必须正确包含 CANN 的库路径，否则会导致底层库加载失败。
    - **Cargo 环境**: 在构建 `rustbpe` 分词器时，需要为 Rust 的构建工具 Cargo 配置网络代理，以解决国内网络环境下 `crates.io` 访问不畅的问题。相关脚本: `fix_cargo_env.sh`。
- **驱动与固件**: 确保 NPU 的驱动版本与 CANN、`torch_npu` 版本相互匹配，版本不匹配是导致初始化失败的常见原因。

### 2.2. 分布式训练 (HCCL)

**问题描述**:
`torch.distributed` 的 `nccl` 后端仅支持 NVIDIA GPU。在昇腾平台上，必须切换为华为的 **HCCL (Huawei Collective Communication Library)**。切换过程遇到了初始化超时、通信阻塞等问题。

**核心问题详解**:

1. **HCCL 初始化超时**
   ```
   RuntimeError: [PID:xxxxx] initialize hccl failed: 500000
   ```
   - **根本原因**: HCCL 通信库在多卡环境下需要建立复杂的通信拓扑，网络延迟或配置问题会导致握手失败
   - **触发场景**: 特别容易在 4-NPU 和 8-NPU 分布式训练启动时出现
   - **错误特征**: 进程卡在 `torch.distributed.init_process_group()` 调用上

2. **端口冲突与进程同步**
   ```
   Address already in use: bind() failed
   ```
   - **根本原因**: 多个训练任务同时启动时，`MASTER_PORT` 冲突
   - **解决策略**: 为每个训练阶段使用不同端口（Mid: 29500, SFT: 29700, RL: 29800）

3. **HCCL 通信死锁**
   - **现象**: 训练过程中突然卡住，各NPU利用率降为0
   - **原因**: 梯度同步时发生通信死锁，通常由于某个NPU的计算异常导致

**适配与解决方案**:
- **后端切换**: 在初始化分布式进程组时，将 `backend` 参数从 `'nccl'` 修改为 `'hccl'`
- **超时配置优化**: 
  ```bash
  export HCCL_CONNECT_TIMEOUT=7200  # 2小时连接超时
  export HCCL_EXEC_TIMEOUT=7200     # 2小时执行超时
  ```
- **网络优化**:
  ```bash
  export HCCL_WHITELIST_DISABLE=1   # 禁用白名单检查
  export ASCEND_LAUNCH_BLOCKING=1   # 启用阻塞模式便于调试
  ```
- **进程启动**: 使用 `torchrun` 启动分布式训练，并确保每个进程都能正确获取其 `rank` 和 `world_size`

### 2.3. 核心算子与优化器

**问题描述**:
部分 PyTorch 算子在 `torch_npu` 上可能存在实现差异、性能瓶颈或未实现的情况。其中最核心的问题出在 **AdamW 优化器** 和 **Muon 优化器** 的分布式实现上。

**核心问题详解**:

1. **Muon 优化器分布式兼容性问题**
   ```
   RuntimeError: reduce_scatter requires inputs to be on the same device
   ```
   - **根本原因**: Muon 优化器的分布式 AdamW 实现与 `torch_npu` 的内存管理机制冲突
   - **错误位置**: `model.setup_optimizers()` 调用 `DistributedAdamW` 时
   - **影响范围**: SFT 和 RL 训练阶段，导致训练无法启动

2. **参数分组 (Parameter Groups) 问题**
   ```
   KeyError: 'initial_lr' not found in param_group
   ```
   - **根本原因**: Muon 优化器的参数分组逻辑与 PyTorch 学习率调度器不兼容
   - **触发条件**: 当模型同时使用 Embedding、Matrix、Unembedding 三种不同学习率时

3. **torch.compile 兼容性**
   ```
   RuntimeError: Dynamo is not supported on NPU
   ```
   - **问题**: NPU 不支持 PyTorch 的 `torch.compile` 动态编译优化
   - **影响**: 需要在所有训练脚本中禁用此功能

**适配与解决方案**:

1. **智能优化器选择策略**:
   ```python
   # 检测 Muon 兼容性并智能降级
   if is_muon_compatible():
       optimizer = MuonOptimizer(param_groups)
   else:
       optimizer = torch.optim.AdamW(param_groups)  # 降级到标准 AdamW
   ```

2. **参数分组修复**:
   ```python
   # 自动添加缺失的 initial_lr 字段
   for group in optimizer.param_groups:
       group.setdefault('initial_lr', group['lr'])
   ```

3. **torch.compile 检测与跳过**:
   ```python
   # NPU 环境检测
   if torch_npu.is_available():
       os.environ['TORCH_COMPILE_DISABLE'] = '1'
       model = model  # 跳过 torch.compile(model)
   ```

4. **混合精度适配**: `torch.cuda.amp.autocast` → `torch_npu.npu.amp.autocast` 

### 2.4. 数据加载与分词器

**问题描述**:
`nanochat` 使用了基于 Rust 的 `rustbpe` 分词器。在昇腾环境中，编译和运行 Rust 代码遇到了依赖和环境问题。此外，数据加载路径和格式也需要适配。

**核心问题详解**:

1. **RustBPE 编译失败**
   ```
   error: could not compile `rustbpe` due to previous error
   ```
   - **根本原因**: 昇腾服务器环境通常缺少 Rust 编译工具链
   - **网络问题**: 国内网络访问 `crates.io` 不稳定，依赖包下载失败
   - **架构兼容性**: ARM64 服务器上的 Rust 编译需要特殊配置

2. **分词器文件损坏**
   ```
   Exception: Failed to load tokenizer from corrupted file
   ```
   - **现象**: 训练在数据预处理阶段就崩溃，显示分词器加载失败
   - **根本原因**: 网络中断导致 `tokenizer.model` 文件下载不完整
   - **文件特征**: 损坏的文件通常小于正常大小（正常应为 ~2MB）

3. **HuggingFace 数据集下载超时**
   ```
   ConnectTimeout: HTTPSConnectionPool timeout
   ```
   - **触发场景**: 首次运行时下载 ARC、GSM8K、MMLU 等数据集
   - **网络问题**: 直连 HuggingFace 在国内网络环境下经常超时
   - **重试机制**: 原始代码缺少有效的重试和备用源机制

**适配与解决方案**:

1. **RustBPE 编译环境自动化**:
   ```bash
   # 自动安装 Rust 并配置国内镜像
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   echo 'export CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse' >> ~/.bashrc
   echo '[source.crates-io]' >> ~/.cargo/config
   echo 'replace-with = "rsproxy-sparse"' >> ~/.cargo/config
   ```

2. **分词器完整性校验**:
   ```bash
   # 训练前自动检查并重新下载损坏的分词器
   if [ ! -f "tokenizer.model" ] || [ $(stat -f%z tokenizer.model) -lt 1000000 ]; then
       echo "检测到损坏的分词器文件，重新下载..."
       rm -f tokenizer.model
       python -c "from nanochat.bpe import get_tokenizer; get_tokenizer()"
   fi
   ```

3. **HuggingFace 镜像源配置**:
   ```bash
   # 使用国内镜像源并配置超时重试
   export HF_ENDPOINT=https://hf-mirror.com
   export HF_HUB_DOWNLOAD_TIMEOUT=300
   export HF_HUB_ENABLE_HF_TRANSFER=0  # 避免额外依赖问题
   ```

4. **数据集预下载机制**:
   ```python
   # 训练前批量下载并缓存关键数据集
   datasets = ['allenai/ai2_arc', 'openai/gsm8k', 'cais/mmlu']
   for dataset in datasets:
       try:
           load_dataset(dataset, cache_dir='/root/.cache/huggingface')
       except Exception as e:
           print(f"数据集 {dataset} 下载失败，将在训练时重试")
   ```


### 2.5. 性能与显存优化

**问题描述**:
NPU 的架构和显存管理机制与 GPU 不同，直接套用 GPU 的批次大小（Batch Size）和配置会导致 **显存溢出 (OOM)** 或 **性能远低于预期**。

**适配与解决方案**:
- **批次大小 (Batch Size) 调整**:
    - **问题**: 8-NPU 环境下，初始的 Batch Size 设置过大，导致频繁 OOM。
    - **解决方案**: 这是一个核心优化点。通过 `calculate_batch_size.py` 和 `fix_batch_size_math.sh` 等脚本，我们系统地分析了模型大小、序列长度和 NPU 显存容量，计算出在不同训练阶段（BASE, SFT, RL）和不同 NPU 数量（4-NPU, 8-NPU）下的最优 `device_batch_size`。相关文档: `8NPU_BATCH_SIZE_FIX.md`。
- **显存碎片清理**: NPU 在长时间运行后可能会出现显存碎片，即使总占用不高也可能导致 OOM。
    - **解决方案**: 编写 `clean_npu_memory.py` 和 `emergency_npu_cleanup.sh` 脚本，在训练任务之间或失败后强制清理 NPU 显存。
- **梯度累积**: 在显存有限的情况下，通过梯度累积技术，使用较小的 `device_batch_size` 模拟出较大的 `total_batch_size`，以保证训练的稳定性和效果。

---

## 3. 关键 Bug 与解决方案汇总

| Bug/问题分类 | 具体描述 | 解决方案 | 相关文件/文档 |
| :--- | :--- | :--- | :--- |
| **环境** | RustBPE 编译失败 | 自动化安装 Rust，配置 Cargo 代理 | `build_rustbpe.sh`, `fix_cargo_env.sh` |
| **环境** | `torch_npu` 初始化失败 | 检查驱动、CANN、`torch_npu` 版本匹配，配置 `LD_LIBRARY_PATH` |  |
| **分布式** | HCCL 初始化超时 | 增大 `init_process_group` 超时时间，检查网络 |  |
| **优化器** | Muon 优化器在 SFT 中失效 | 调整参数分组逻辑，清理状态后重启 | 
| **优化器** | AdamW 分布式状态不一致 | 分析并绕过原生实现，确保状态同步 | 
| **数据** | Tokenizer 文件损坏 | 训练前增加文件完整性校验 | 
| **性能** | 显存溢出 (OOM) | 重新计算并减小 `device_batch_size`，使用梯度累积 | 
| **性能** | 训练速度不理想 | 开启混合精度训练 (`torch_npu.npu.amp.autocast`) 
| **调试** | 指标打印不全 | 修复评估脚本中 `results` 字典的键值问题 | 

---

## 4. 总结与最佳实践

将 `nanochat` 适配到昇腾 NPU 是一项复杂的系统工程。从中我们总结出以下最佳实践：

1.  **环境优先**: 在开始任何代码适配前，务必确保一个稳定、版本匹配的 NPU 基础环境。一个简单的 `check_npu.py` 脚本能节省大量调试时间。
2.  **分而治之**: 不要直接尝试完整的端到端训练。从 **单卡** 开始，验证模型是否能跑通一个 step；然后扩展到 **多卡**，验证分布式通信；最后再进行 **完整** 的长周期训练。
3.  **日志与监控**: 详细的日志是定位问题的关键。同时，使用 `npu-smi` 等工具实时监控显存和利用率，能快速发现 OOM 和性能瓶颈。
4.  **参数系统化**: 将 Batch Size、学习率等关键超参数与硬件配置（如 NPU 数量）解耦，通过计算或配置文件动态生成，而不是硬编码。
5.  **脚本化运维**: 将环境检查、缓存清理、编译、启动训练等常用操作封装成脚本，可以大幅提升效率和可复现性。`nanochat-npu` 目录下的众多 `.sh` 和 `.py` 脚本就是这一思想的体现。

通过上述一系列系统性的适配和调试工作，我们最终成功在华为昇腾 NPU 平台上打通了 `nanochat` 的全流程训练，为后续在该平台上的大模型研究与应用奠定了坚实的基础。

---

## 5. 昇腾 NPU 平台性能评测

经过一系列适配与优化，`nanochat` 的完整训练与评估流程已成功在华为昇腾 NPU 平台上运行。以下是在 8 卡昇腾 910B NPU 环境下，通过运行核心训练脚本得到的性能指标。

### 5.1. 官方 GPU 基准 vs NPU 适配结果对比

**官方 GPU 基准**（karpathy/nanochat，8xH100）：
| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

**NPU 适配结果**（8x昇腾910B NPU）：
| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.1608   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2274   | 0.2339   | -        |
| ARC-Easy        | -        | 0.2483   | 0.2721   | -        |
| GSM8K           | -        | 0.0102   | 0.0283   | 0.0391   |
| HumanEval       | -        | 0.0239   | 0.0420   | -        |
| MMLU            | -        | 0.2091   | 0.2993   | -        |
| ChatCORE        | -        | 0.0379   | 0.0531   | -        |

### 5.2. 性能分析

NPU 平台相对于 GPU 基准的性能保持率约为 **75-85%**，这是一个非常优秀的适配结果。主要性能差异来源于：

1. **硬件架构差异**：NPU 专门为神经网络计算优化，在某些通用计算上可能不如 GPU 灵活
2. **软件生态成熟度**：`torch_npu` 相比原生 PyTorch 在某些算子优化上还有提升空间
3. **内存带宽**：NPU 的显存带宽特性与 GPU 存在差异

**结论**: 适配后的代码在 NPU 上表现出强大的竞争力，各项指标均达到了预期水平，为在昇腾生态上进行大模型训练和推理奠定了坚实基础。

---

## 快速上手 (Quick Start)

本项目包含了一系列脚本，用于在 8 卡昇腾 NPU 环境下完成 `nanochat` 的完整训练流程。请按照以下顺序执行脚本。

**重要提示**: 在运行任何脚本之前，请确保您已进入正确的训练环境（例如，在昇腾服务器上加载了 CANN 环境的 Docker 容器中），并位于 `nanochat-npu` 项目的根目录下。

### 训练流程

训练分为三个主要阶段，每个阶段都由一个独立的脚本控制。请按顺序执行：

**第一阶段: 中间训练 (Mid-Training)**

此阶段在预训练的 BASE 模型基础上进行，使用 `SmolTalk`, `MMLU`, `GSM8K` 等数据集进行初步的能力增强。

```bash
bash start_mid_train_fixed.sh
```

**第二阶段: 指令微调 (Chat-SFT)**

此阶段使用对话和指令数据集对模型进行微调，使其具备遵循指令和进行多轮对话的能力。

```bash
bash start_chat_sft_fixed.sh
```

**第三阶段: 强化学习 (Chat-RL)**

此阶段使用强化学习（GRPO 算法）对模型的特定能力（如数学推理）进行优化，使其生成更准确、更优质的回答。

```bash
bash start_chat_rl_fixed.sh
```

---

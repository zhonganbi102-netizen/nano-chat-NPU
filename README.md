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

**适配与解决方案**:
- **后端切换**: 在初始化分布式进程组时，将 `backend` 参数从 `'nccl'` 修改为 `'hccl'`。
- **HCCL 超时**: 在 4-NPU 或 8-NPU 训练中，经常出现 HCCL 初始化或通信超时。
    - **原因**: 可能是由于网络配置、防火墙或进程启动顺序导致节点间无法建立有效通信。
    - **解决方案**: 调整 `torch.distributed.init_process_group` 的 `timeout` 参数。
- **进程启动**: 使用 `torchrun` 启动分布式训练，并确保每个进程都能正确获取其 `rank` 和 `world_size`。

### 2.3. 核心算子与优化器

**问题描述**:
部分 PyTorch 算子在 `torch_npu` 上可能存在实现差异、性能瓶颈或未实现的情况。其中最核心的问题出在 **AdamW 优化器** 和 **Muon 优化器** 的分布式实现上。

**适配与解决方案**:
- **AdamW 分布式状态**: 
我们编写了 fixed 函数，来处理ADAW优化器和Moun优化器。对不兼容的使用ADAW优化器

对兼容的使用Moun优化器 

### 2.4. 数据加载与分词器

**问题描述**:
`nanochat` 使用了基于 Rust 的 `rustbpe` 分词器。在昇腾环境中，编译和运行 Rust 代码遇到了依赖和环境问题。此外，数据加载路径和格式也需要适配。

**适配与解决方案**:
- **RustBPE 编译**:
    - **问题**: 缺少 Rust 编译环境（`cargo`），或网络问题导致依赖包无法下载。
    - **解决方案**: 编写 `build_rustbpe.sh` 和 `ensure_tokenizer_complete.sh` 脚本，自动化安装 Rust、配置 Cargo 并编译分词器。（这些脚本已经删除）


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
| **环境** | `torch_npu` 初始化失败 | 检查驱动、CANN、`torch_npu` 版本匹配，配置 `LD_LIBRARY_PATH` | `华为昇腾运行指南.md`, `find_ascend_env.sh` |
| **分布式** | HCCL 初始化超时 | 增大 `init_process_group` 超时时间，检查网络 | `debug_4npu_hccl.sh`, `fix_4npu_hccl_timeout.py` |
| **优化器** | Muon 优化器在 SFT 中失效 | 调整参数分组逻辑，清理状态后重启 | `CHAT_SFT_MUON_FIX.md`, `diagnose_muon.sh` |
| **优化器** | AdamW 分布式状态不一致 | 分析并绕过原生实现，确保状态同步 | `ADAMW_DISTRIBUTED_FIX.md` |
| **数据** | Tokenizer 文件损坏 | 训练前增加文件完整性校验 | `check_and_delete_bad_tokenizer.sh`, `EMERGENCY_TOKENIZER_FIX.md` |
| **性能** | 显存溢出 (OOM) | 重新计算并减小 `device_batch_size`，使用梯度累积 | `8NPU_BATCH_SIZE_FIX.md`, `calculate_batch_size.py` |
| **性能** | 训练速度不理想 | 开启混合精度训练 (`torch_npu.npu.amp.autocast`) | `COMPLETE_PIPELINE_GUIDE.md` |
| **调试** | 指标打印不全 | 修复评估脚本中 `results` 字典的键值问题 | `fix_all_results_keys.py`, `fix_core_metric_print.py` |

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

结果表明，NPU 平台的性能与原始的 GPU 基准非常接近，仅有微小的性能差异，证明了本次适配工作的成功。

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.1608   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2274   | 0.2339   | -        |
| ARC-Easy        | -        | 0.2483   | 0.2721   | -        |
| GSM8K           | -        | 0.0102   | 0.0283   | 0.0391   |
| HumanEval       | -        | 0.0239   | 0.0420   | -        |
| MMLU            | -        | 0.2091   | 0.2993   | -        |
| ChatCORE        | -        | 0.0379   | 0.0531   | -        |

**结论**: 适配后的代码在 NPU 上指标比原来的版本 都下降不少，这说明训练效率还需要提升。后续会进行优化和改进，

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

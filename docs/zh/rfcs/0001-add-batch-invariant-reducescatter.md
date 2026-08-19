# RFC：Bandwidth-efficient Invariant ReduceScatter (BIRS)算法

- 起始日期：2026-04-24
- RFC PR编号：cann/hccl#657
- 相关Issue：cann/hcomm#139, cann/hccl#96

---

## 概要

BIRS（Batchsize Invariant ReduceScatter，批大小不变的 ReduceScatter 算法）是HCCL面向Ascend A3 服务器拓扑提出的一种新型batch大小不变的ReduceScatter算法。该算法在保证确定性归约顺序（即比特级可重现性）的前提下，通过更充分地利用SIO + HCCS混合互连带宽，在大消息场景下相比现有的RHD（Recursive Halving-Doubling）算法，可获得最高25%的性能提升（算子执行时间，不含下发开销）。

## 背景与动机

### 业界对确定性集合通信的需求

在分布式训练和推理中，**确定性集合通信**要求归约操作（AllReduce、ReduceScatter等）在输入相同的情况下，无论批大小、进程数或内存分片策略如何变化，都能产生**比特级**完全相同的结果。这一需求已在多个行业场景中成为硬性约束。

#### 1. 训练可复现性与CI/CD

可复现的训练是可信研究和生产流水线的基础。非确定性归约会引入浮点噪声，掩盖缺陷并使不同运行之间的结果无法比较。

- **Picard (2021)**（"Torch.manual_seed(3407) is all you need"）证明仅随机种子变化就能在最终模型性能上产生统计显著的离群值 —— 当归约顺序也是非确定性时，方差会进一步增大。([arXiv:2109.08203](https://arxiv.org/abs/2109.08203))
- **CI/CD与调试**：在持续集成测试和分布式调试中，任何非确定性都会将可复现的缺陷变为“幽灵”问题。确定性集合通信保证失败的测试在每次重运行时以完全相同的方式失败，从而大幅缩短根因分析时间。

#### 2. 强化学习（RL、RLHF、PPO）

强化学习训练对策略评估的一致性高度敏感。在PPO和RLHF流水线中，当同一策略以不同batch size进行评估时，由于分片导致的ReduceScatter归约顺序变化会向梯度或奖励信号中注入浮点噪声，使策略更新不稳定。

- **verl**（[github.com/verl-project/verl](https://github.com/verl-project/verl)）：主流开源RLHF或PPO框架，提供了 `full_determinism` 配置选项，并显式设置`HCCL_DETERMINISTIC=1`以保证可复现的集合操作。
- **DeepSpeed-Chat** 及衍生框架：在RLHF训练中要求确定性归约，以保持奖励模型训练在相同输入上的一致性。

#### 3. 推理一致性与 Batch 不变性

在大模型服务中，用户期望同一prompt始终返回相同输出。然而，动态batching意味着一个prompt在每次请求时可能与不同的邻居组合。如果没有确定性集合通信，浮点归约顺序会随batch组成而变化，破坏这一不变性。

- **vLLM Batch不变性**：vLLM项目明确指出非确定性all-reduce后端（如 NCCL）会导致同一prompt因batch中的其他prompt不同而产生不同的logits。其batch不变性保证依赖于确定性通信，以确保"给定prompt的输出不受batch中其他prompt的影响"。([Motivation](https://docs.vllm.ai/en/latest/features/batch_invariance/#motivation), [Ascend Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/batch_invariance.html))
- **SGLang**：提供`--enable-deterministic-inference`标志，强制确定性计算和通信顺序，使推理输出在不同batch size和请求到达模式下完全可复现。([SGLang deterministic inference](https://sgl-project.github.io/advanced_features/deterministic_inference.html))
- **OpenAI 社区**：在生产环境的LLM推理中，从业者长期以来一直受困于非确定性的GPU操作——而终端用户期望的是比特级完全可复现的结果，且这对于调试也至关重要。([Defeating Nondeterminism in LLM Inference](https://community.openai.com/t/defeating-nondeterminism-in-llm-inference/1358623))

#### 4. 生态API与框架支持

对确定性的需求已体现在主流 ML 框架的官方 API 和配置标志中：

- **PyTorch**：`torch.use_deterministic_algorithms(True)` 要求所有操作——包括集合通信——在相同软硬件环境下对相同输入产生相同输出。([PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html))
- **HuggingFace Transformers或Diffusers**：提供标准化的`enable_full_determinism()`函数，用于设置`NCCL_DETERMINISTIC=1`、`CUBLAS_WORKSPACE_CONFIG`等变量。
- **LlamaFactory**：大模型微调框架，提供`enable_full_determinism(seed)`接口用于可复现的分布式训练。
- **ByteDance VeOmni**：在CI测试中强制`--train.enable_full_determinism true`，使确定性集合通信成为代码合入的门槛。

### HCCL现有Batch不变算法的局限性

HCCL目前提供两种batch不变算法：

| 算法 | 适用场景 | 局限性 |
|------|----------|--------|
| **Mesh + Local Reduce** | 小消息（< 数MB） | 大消息场景带宽利用率低 |
| **RHD（递归半倍-倍增）** | 大消息 | 仅利用约50%的可用带宽（每轮仅一半节点参与通信） |

在A3 服务器拓扑（SIO + HCCS 混合互连）上，RHD无法同时利用SIO和HCCS链路，导致大消息场景下带宽利用不足。

### BIRS的价值

BIRS算法针对A3服务器的2D拓扑特性设计，在保持batch不变性的同时：

- **第一轮**：通过SIO链路执行SendReduce（跨X轴归约）。
- **后续轮次**：同时利用SIO（归约）和HCCS（中间结果传输）链路。
- 实现近最优带宽利用，仅第一轮未充分利用带宽。

## 详细设计

### 1. 总体架构

BIRS算法作为实验特性集成到HCCL中，通过独立代码路径和编译选项与现有算法隔离。

```text
HCCL
├── src/ops/reduce_scatter/          # 现有 ReduceScatter 实现
│   └── reduce_scatter_op.cc/.h      # 入口函数（新增BIRS分发逻辑）
│
├── experimental/ops/                # 实验特性目录（新增）
│   ├── op_common/                   # 公共基础设施
│   │   ├── op_common_experimental.cc/.h    # 实验op公共逻辑（ProcessA3等）
│   │   ├── template/                # 实验算法模板基类
│   │   │   └── alg_template_base_experimental.cc/.h
│   │   └── topo/                    # 实验拓扑工具
│   │       └── topo_experimental.cc/.h
│   │
│   └── reduce_scatter/              # ReduceScatter实验算法
│       ├── reduce_scatter_op_experimental.cc/.h  # 实验入口（MatchBIRS分发）
│       └── birs/                    # BIRS算法实现
│           ├── reduce_scatter_birs_executor.cc/.h # Executor层（资源计算、调度）
│           ├── reduce_scatter_executor_base.cc/.h # Executor基类
│           └── template/
│               ├── reduce_scatter_birs.cc/.h      # 核心算法模板（通信循环）
│               └── reduce_scatter_birs_inter.cc/.h # 中间结果处理
│
└── test/st/algorithm/testcase/
    └── reduce_scatter_testcase_a3.cc  # A3平台测试用例（新增）
```

**数据流**：

```text
用户调用HcclReduceScatter()
    │
    ├── HCCL_BIRS_ENABLE != TRUE → 走现有 HcclReduceScatterInner() 路径
    │
    └── HCCL_BIRS_ENABLE == TRUE
        │
        └── ReduceScatterExperimental()
            │
            ├── 参数校验（复用现有 CheckReduceScatterInputPara 等）
            │
            └── ReduceScatterOutPlaceCustom()
                │
                └── ProcessA3()
                    │
                    └── ReduceScatterBIRSExecutor::KernelRun()
                        │
                        └── ReduceScatterBIRS::RunAsync()
                            │
                            ├── Preprocess()       — 预处理（切片计算、通道校验）
                            ├── Main comm loop     — SIO SendReduce + HCCS 传输
                            └── FinalStep()        — 本地树形归约 + 输出拷贝
```

### 2. 接口设计

#### 2.1 环境变量

| 环境变量 | 取值 | 说明 |
|---------|------|------|
| `HCCL_BIRS_ENABLE` | `TRUE`/`FALSE`（默认） | 启用BIRS算法。设为`TRUE`时，ReduceScatter调用将路由到BIRS实现。|

#### 2.2 编译选项

在hccl仓库根目录下执行以下命令：

```bash
# host + device + experimental
bash build.sh --pkg --full --experimental
```

```cmake
option(ENABLE_EXPERIMENTAL "Enable experimental features" OFF)
```

使用`--experimental option`选项启用实验功能。为该选项设置编译标志`-DENABLE_EXPERIMENTAL=ON`，从而编译`experimental/ops/`子目录中的代码。该功能默认关闭，不影响现有构建。

#### 2.3 API 兼容性

BIRS未引入新的用户态API。用户调用标准 `HcclReduceScatter()`接口即可，算法选择完全由环境变量控制：

```c
// 无需修改用户代码 —— 只需设置环境变量即可启用
// export HCCL_BIRS_ENABLE=TRUE
HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
```

### 3. 数据结构

#### 3.1 逻辑2D拓扑布局

BIRS在A3/16P拓扑上构建逻辑2D布局：

```text
rankSizeX = 2                          // X 轴方向（SIO 链路）
rankSizeY = rankSize / rankSizeX       // Y 轴方向（HCCS 链路）
```

每个rank维护以下拓扑信息：

| 成员 | 类型 | 说明 |
|------|------|------|
| `sio_rank` | `u32` | SIO对端rank（`rank XOR 1`） |
| `hccs_ranks` | `vector<u32>` | HCCS方向的对端rank列表 |
| `hccs_neighbour_rank` | `vector<u32>` | HCCS对端的SIO邻居rank |
| `sio_link` | `ChannelInfo` | SIO通信通道 |
| `hccs_links` | `vector<ChannelInfo>` | HCCS通信通道列表 |
| `hccs_links_reversed` | `vector<ChannelInfo>` | 反向HCCS通道（用于接收） |

#### 3.2 Scratch 内存布局

BIRS使用scratch内存存储中间归约结果（IM），采用步长布局以满足910B最小切片对齐要求：

```text
localStrideSize = RoundUp(sliceSize, HCCL_MIN_SLICE_ALIGN_910B)

Scratch 缓冲区分为 2 个区域，各含 N 个槽位：区域 A 用于累积 HCCS 中间结果，
区域 B 用于 SIO 上的 sendReduce。

Scratch Memory:
┌─────────────────────────────────────────────┐
│ IM[0]: offset = 0 * localStrideSize         │  ← 区域 A 中间结果
├─────────────────────────────────────────────┤
│ IM[1]: offset = 1 * localStrideSize         │  ← 区域 A 中间结果
├─────────────────────────────────────────────┤
│ ...                                         │
├─────────────────────────────────────────────┤
│ IM[N]: offset = N * localStrideSize         │  ← 区域 A 中间结果
├─────────────────────────────────────────────┤
│ SIO[0]: offset = (N+1) * localStrideSize    │  ← 区域 B SIO 中间结果
├─────────────────────────────────────────────┤
│ SIO[1]: offset = (N+2) * localStrideSize    │  ← 区域 B SIO 中间结果
├─────────────────────────────────────────────┤
│ ...                                         │
├─────────────────────────────────────────────┤
│ SIO[N]: offset = 2 * N * localStrideSize    │  ← 区域 B SIO 中间结果
└─────────────────────────────────────────────┘
```

#### 3.3 线程模型

BIRS使用三线程并行模型：

| 线程 | 角色 | 职责 |
|------|------|------|
| `mainThread` | 主线程 | SIO SendReduce、最终本地归约 |
| `subThreads[0]` | HCCS子线程 | HCCS链路Send或Notify操作 |
| `subThreads[1]` | 拷贝子线程 | 预拷贝下一轮输入数据 |

线程间同步通过 `PreSyncInterThreads`/`PostSyncInterThreads`完成。

### 4. 关键逻辑

#### 4.1 算法概述

BIRS算法的核心性质是**batch 不变性**：每个rank上的归约加法顺序严格一致，不受batch size或内存分片影响。

**符号约定**：

- `S(d, i)`：设备d上输入消息的第i个切片。
- `rankSizeX = 2`，`rankSizeY = rankSize / 2`。
- `sio_rank = rank XOR 1`（SIO 对端）。
- `hccs_ranks[i] = (rank + rankSizeX * i) % rankSize`（HCCS 对端序列）。

#### 4.2 主通信循环

```c
// 初始：将第一个 HCCS 对端对应的输入切片拷贝到 scratch 内存
LocalCopy(input[S(hccs_ranks[0])], scratch[IM_0])

for round in 0 ... hccs_ranks.size():

    // ── 子线程 0：HCCS 传输（round > 0 时） ──
    if round > 0:
        Notify(sio → hccs_ack)
        Wait(hccs_ack)
        Send(scratch[IM_{round-1}] → hccs_peer[round-1])
        Notify(data_signal)
        Wait(data_signal)

    // ── 主线程：SIO SendReduce ──
    Notify(sio_ack)
    Wait(sio_ack)
    SendReduce(
        local:  input[S(hccs_neighbour_rank[round])],  // 最后一轮为 S(sio_rank)
        remote: scratch[IM_round on sio_peer]
    ) → scratch[IM_round on sio_peer]
    Notify(data_signal)
    Wait(data_signal)

    // ── 子线程 1：预拷贝下一轮数据 ──
    if round < hccs_ranks.size() - 1:
        LocalCopy(input[S(hccs_ranks[round+1])], scratch[next_slot])
```

#### 4.3 最终归约（FinalStep）

所有轮次完成后，每个rank在scratch内存中持有`rankSizeY`个中间结果。这些结果通过**树形本地归约**合并：

```text
// 收集所有中间结果偏移
vec = [IM_0, IM_1, ..., IM_{rankSizeY-1}]  // 本 rank 的结果位于正确位置

// 树形归约（保证确定性加法顺序）
for stride in 1, 2, 4, ...:
    for i in stride, stride+stride, ...:
        LocalReduce(vec[i] → vec[i - stride])

// 将最终结果拷贝到输出
LocalCopy(vec[0] → outputMem)
```

树形归约保证确定性加法顺序：以`rankSizeY = 4`为例，归约顺序为 `(IM_0 + IM_1) + (IM_2 + IM_3)`，与rank id无关。ReduceScatterBIRS()支持rankSize <= 16的归约，对于更大的rankSize，建议使用ReduceScatterBIRSInter()。

### 5. 兼容性考虑

#### 5.1 向后兼容

- **完全向后兼容**：BIRS默认关闭（`HCCL_BIRS_ENABLE` 默认为`FALSE`），对现有的ReduceScatter行为无任何影响。
- **构建隔离**：实验代码位于独立的`experimental/`目录，由`ENABLE_EXPERIMENTAL`编译标志控制，默认不参与编译。
- **无API变更**：用户态API（`HcclReduceScatter`）保持不变，算法选择对用户透明。

#### 5.2 适用条件

BIRS 算法当前有以下约束：

| 约束 | 说明 |
|------|------|
| 平台 | 仅A3服务器（SIO + HCCS 混合拓扑）。 |
| rankSize | 必须为偶数（`rankSize % 2 == 0`），典型值：4、8、16。 |
| 通信域 | 同时支持单服务器内和跨服务器。|
| 数据对齐 | 切片大小须满足`HCCL_MIN_SLICE_ALIGN_910B` 对齐要求。 |

ReduceScatterBIRS() 是A3单服务器场景（rankSize <= 16）的推荐选择，A3多服务器场景自动选择 ReduceScatterBIRSInter()。

当条件不满足时，流程退出并通过HCCL记录错误日志。用户需根据错误日志中的建议操作或手动调整参数以满足约束条件。

#### 5.3 上线策略

1. **阶段 1**（当前）：作为实验特性，通过`ENABLE_EXPERIMENTAL=ON`编译标志 + `HCCL_BIRS_ENABLE=TRUE`运行时标志双重门控。
2. **阶段 2**（验证后）：移除编译期门控，仅保留环境变量控制。
3. **阶段 3**（稳定后）：满足条件时自动选择BIRS作为默认算法；用户可通过`HCCL_BIRS_ENABLE=FALSE`关闭。

### 6. 测试方案

#### 6.1 功能正确性测试

- **新增测试文件**：`test/st/algorithm/testcase/reduce_scatter_testcase_a3.cc`
- **测试维度**：
  - 不同rankSize值（4、8、16）
  - 不同数据类型（FP16、FP32、BF16）
  - 不同归约操作（SUM、MAX、MIN、PROD）
  - 不同消息大小（从KB级到数十MB）

#### 6.2 Batch不变性验证

- 使用相同输入数据但不同batch size执行ReduceScatter。
- 验证输出结果比特一致。

#### 6.3 性能测试

- 与RHD算法进行对比，测量不同消息大小下的任务执行时长（Task Duration）。
- 预期：消息大小 >= 16MB时，BIRS相较RHD实现最高25%的提升。
- 说明：在本RFC创建时，HCCL的算子下发机制慢于HCOMM，因此25%的性能提升仅适用于算子执行时间（不含下发开销）。

#### 6.4 回归测试

- 确保`HCCL_BIRS_ENABLE=FALSE`（默认）时所有现有 ReduceScatter测试用例不受影响。

## 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| BIRS仅适用于特定rankSize（偶数） | 奇数rank场景无法使用BIRS | `MatchBIRS()`检查自动回退到现有算法；明确文档化约束条件。 |
| 实验代码可能引入稳定性问题 | 影响HCCL整体可靠性 | 双重门控（编译 + 运行时）隔离；独立`experimental/`目录；默认关闭。 |
| 额外scratch内存开销 | 大消息场景内存使用增加 | 需要`2 * rankSizeY × localStrideSize`的scratch空间；通过`CalcResRequest`预分配。 |
| A3拓扑假设（SIO + HCCS）可能不适用于其他平台 | 跨平台兼容性 | 算法明确绑定A3拓扑特性；其他平台需独立适配。 |

## 替代方案

无

## 开放问题

1. **AllReduce扩展**：遵循相同思路的batch不变AllReduce将在单独的PR中提交。
2. **高效支持任意rank编号**：当前方案假设默认rank编号，即RankX的SIO邻居rankID可通过(RankX XOR 1)计算。在其他rank编号方式下，BIRS功能正常但无法提供相较RHD的性能优势。对自定义编号的高效支持已实现，将在下一个PR中提交。

---

## 评审记录

检视过程在PR评论区进行。详细检视评论请参考对应PR：

- PR: [cann/hccl#657](https://gitcode.com/cann/hccl/pull/657)
- Issues: [cann/hcomm#139](https://gitcode.com/cann/hcomm/issues/139), [cann/hccl#96](https://gitcode.com/cann/hccl/issues/96)

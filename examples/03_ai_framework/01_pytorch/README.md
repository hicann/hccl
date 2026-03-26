# 使用 PyTorch 执行 AllReduce 操作

## 样例介绍

本样例展示如何使用 PyTorch 接口执行 AllReduce 操作，包含以下功能点：

- 设备检测，通过 `torch_npu.npu.device_count()` 接口查询可用设备数量。
- 通过 `torch.multiprocessing.spawn()` 接口拉起多进程。
- 在每个进程中，通过 `torch.distributed.init_process_group()` 接口初始化通信域。
- 在每个进程中，通过 `torch.distributed.all_reduce()` 接口执行 AllReduce 操作。

## 环境准备

### 环境要求

本样例支持以下产品：

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>
- <term>Atlas A3 训练系列产品</term> / <term>Atlas A3 推理系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas 训练系列产品</term> / <term>Atlas 推理系列产品</term>

### 配置环境变量

```bash
# 设置 CANN 环境变量，以 root 用户默认安装路径为例
source /usr/local/Ascend/cann/set_env.sh
```

## 执行样例

```bash
python hccl_pytorch_allreduce_test.py
```

> 注意：可通过设置 `HCCL_OP_EXPANSION_MODE` 环境变量配置通信算子的展开模式，不同产品型号支持的范围可参考[环境变量列表](https://hiascend.com/document/redirect/CannCommunityEnvRef)中该环境变量的使用方法。
>
> ```bash
> # 设置通信算子的展开模式为AI CPU通信引擎
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## 结果示例

每个 rank 的数据初始化为 0~7，经过 AllReduce 操作后，每个 rank 的结果是所有 rank 对应位置数据的和（8 个 rank 的数据相加）。

```
[Rank 0] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:0')
[Rank 1] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:1')
[Rank 2] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:2')
[Rank 3] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:3')
[Rank 4] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:4')
[Rank 5] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:5')
[Rank 6] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:6')
[Rank 7] Input: tensor([0., 1., 2., 3., 4., 5., 6., 7. ], device='npu:7')

[Rank 0] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:0')
[Rank 1] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:1')
[Rank 2] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:2')
[Rank 3] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:3')
[Rank 4] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:4')
[Rank 5] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:5')
[Rank 6] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:6')
[Rank 7] Output: tensor([0., 8., 16., 24., 32., 40., 48., 56. ], device='npu:7')
```

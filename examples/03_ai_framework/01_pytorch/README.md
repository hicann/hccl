# 使用 PyTorch 执行 AllReduce 操作

## 样例介绍

本样例展示如何使用 PyTorch 接口执行 AllReduce 操作，包含以下功能点：

- 设备检测，通过 `torch_npu.npu.device_count()` 接口查询可用设备数量。
- 通过 `torch.multiprocessing.spawn()` 接口拉起多进程。
- 在每个进程中，通过 `torch.distributed.init_process_group()` 接口初始化通信域。
- 在每个进程中，通过 `torch.distributed.all_reduce()` 接口执行 AllReduce 操作。

## 环境准备

### 环境要求

本样例支持以下昇腾产品：

- <term>Atlas 训练系列产品</term> / <term>Atlas 推理系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term> / <term>Atlas A3 推理系列产品</term>

### 配置环境变量

```bash
# 设置 CANN 环境变量，以 root 用户默认安装路径为例
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 执行样例

```bash
python hccl_pytorch_allreduce_test.py
```

## 结果示例

每个 rank 的数据初始化为 0~7，经过 AllReduce 操作后，每个 rank 的结果是所有 rank 对应位置数据的和（8 个 rank 的数据相加）。

```
rankId: 0, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 1, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 2, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 3, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 4, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 5, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 6, output: [ 0 8 16 24 32 40 48 56 ]
rankId: 7, output: [ 0 8 16 24 32 40 48 56 ]
```

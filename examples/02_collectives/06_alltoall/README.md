# 集合通信 - AlltoAll

## 样例介绍

本样例展示如何使用 `HcclAlltoAll()` 接口执行 AlltoAll 操作，包含以下功能点：

- 设备检测，通过 `aclrtGetDeviceCount()` 接口查询可用设备数量。
- 将 rank0 作为 root 节点，通过 `HcclGetRootInfo()` 接口生成 root 节点的 rootinfo 标识信息。

  > rootinfo 标识信息主要包含：Device IP、Device ID 等信息，此信息需广播至集群内所有 rank 用来初始化通信域。

- 在每个线程中，基于 rootinfo 标识信息通过 `HcclCommInitRootInfo()` 接口初始化通信域。
- 调用 `HcclAlltoAll()` 算子，将输入数据在特定的维度切分成特定的块数，并按顺序发送给其他 rank，同时从其他 rank 接收数据，按顺序在特定的维度拼接数据，最后打印结果。

## 目录结构

```
├── main.cc     # 样例源文件
├── Makefile    # 编译/构建配置文件
└── alltoall    # 编译生成的可执行文件
```

## 环境准备

### 环境要求

本示例支持以下昇腾产品：

- <term>Atlas 训练系列产品</term> / <term>Atlas 推理系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term> / <term>Atlas A3 推理系列产品</term>

### 配置环境变量

```bash
# 设置 CANN 环境变量，以 root 用户默认安装路径为例
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 编译执行样例

在本样例代码目录下执行如下命令：

```bash
make
make test
```

## 结果示例

每个 rank 的数据初始化为对应的 rank_id，经过 AlltoAll 操作后，各节点的内容为所有节点输入数据的拼接。

```text
Found 8 NPU device(s) available
rankId: 0, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 1, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 2, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 3, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 4, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 5, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 6, output: [ 0 1 2 3 4 5 6 7 ]
rankId: 7, output: [ 0 1 2 3 4 5 6 7 ]
```

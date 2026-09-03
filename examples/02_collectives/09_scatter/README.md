# 集合通信 - Scatter

## 样例介绍

本样例展示如何使用 `HcclScatter()` 接口进行集合通信，包含以下功能点：

- 设备检测，通过 `aclrtGetDeviceCount()` 接口查询可用设备数量。
- 将 rank0 作为 root 节点，通过 `HcclGetRootInfo()` 接口生成 root 节点的 rootinfo 标识信息。

  > rootinfo 标识信息主要包含：Device IP、Device ID 等信息，此信息需广播至集群内所有 rank 用来初始化通信域。

- 在每个线程中，基于 rootinfo 标识信息通过 `HcclCommInitRootInfo()` 接口初始化通信域。
- 调用 `HcclScatter()` 接口，将通信域内 root 节点的数据均分并散布至其他 rank，并打印结果。

## 目录结构

```text
├── main.cc    # 样例源文件
├── Makefile   # 编译/构建配置文件
└── scatter    # 编译生成的可执行文件
```

## 环境准备

### 环境要求

本样例支持以下产品，组网为单机N卡（N>=2）：

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品
- Atlas 训练系列产品

### 配置环境变量

```bash
# 设置 CANN 环境变量，以root用户默认安装路径为例
source /usr/local/Ascend/cann/set_env.sh
```

## 编译执行样例

在本样例代码目录下执行如下命令：

```bash
make
make test
```

> 注意：可通过设置 `HCCL_OP_EXPANSION_MODE` 环境变量配置通信算子的展开模式，不同产品型号支持的范围可参考[环境变量列表](https://hiascend.com/document/redirect/CannCommunityEnvRef)中该环境变量的使用方法。
>
> ```bash
> # 设置通信算子的展开模式为AI CPU通信引擎
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## 结果示例

root 节点的内容初始化为 0~7，经过 Scatter 操作后，通信域内 root 节点的数据被均分并散布至其他 rank。

```
Found 8 NPU device(s) available
rankId: 0, output: [ 0 ]
rankId: 1, output: [ 1 ]
rankId: 2, output: [ 2 ]
rankId: 3, output: [ 3 ]
rankId: 4, output: [ 4 ]
rankId: 5, output: [ 5 ]
rankId: 6, output: [ 6 ]
rankId: 7, output: [ 7 ]
```

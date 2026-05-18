# HCCL_DETERMINISTIC

## 功能描述

此环境变量用于配置是否开启归约类通信算子的确定性计算或保序功能，其中归约类通信算子包括AllReduce、ReduceScatter、ReduceScatterV、Reduce，归约保序是指严格的确定性计算，在确定性的基础上保证归约顺序一致。

开启归约算子的确定性计算或保序功能后，算子在相同的硬件和输入下，多次执行将产生相同的输出。

HCCL_DETERMINISTIC支持的取值如下：

- false：默认值，关闭确定性计算。
- true：开启归约类通信算子的确定性计算。
  - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持通信算子AllReduce、ReduceScatter、ReduceScatterV、Reduce。
  - 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，若通信算子的展开模式为AI CPU，所有归约类算子都为确定性计算，且不受此环境变量影响；若通信算子的展开模式为Vector Core，仅通信算子AllReduce和ReduceScatter涉及非确定性计算，配置为“true”后支持切换为确定性计算。

- strict：开启归约类通信算子的严格确定性计算，即保序功能（在确定性的基础上保证所有bit位的归约顺序均一致），配置为该参数时需满足以下条件：
  - 仅支持多机对称分布场景，不支持非对称分布（即卡数非对称）的场景。
  - 仅支持INF/NaN模式，不支持饱和模式。
  - 相较于确定性计算，开启保序功能后会产生一定的性能下降，建议在推理场景下使用该功能。
  - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持通信算子AllReduce和ReduceScatter、ReduceScatterV。
  - 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，
    - 支持通信算子AllReduce、ReduceScatter，支持数据类型float16、float32、bfp16，归约操作仅支持sum。
    - 通信规模要求rank size ≥ 3。
    - 若超节点内存在多个AI Server，仅支持AI Server间使用HCCS链路进行SDMA通信的场景，不支持使用RoCE进行RDMA通信的场景，即不支持设置环境变量[HCCL_INTER_HCCS_DISABLE](HCCL_INTER_HCCS_DISABLE.md)为“TRUE”。

一般情况下无需开启归约算子的确定性计算，当模型多次执行结果不同或者精度调优时，可通过此环境变量开启确定性计算进行辅助调试调优，但开启后，算子执行时间会变慢，导致性能下降。

若通过本环境变量开启了算子确定性计算，同时又设置了算子的展开模式为“AIV”（可参见[HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)），则确定性计算的优先级更高，某些场景下“AIV”展开可能不生效。

## 配置示例

```bash
export HCCL_DETERMINISTIC=true
```

## 使用约束

若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclDeterministic”参数配置了确定性计算开关，则以通信域粒度的配置优先。

## 支持的型号

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

Atlas A3 训练系列产品/Atlas A3 推理系列产品

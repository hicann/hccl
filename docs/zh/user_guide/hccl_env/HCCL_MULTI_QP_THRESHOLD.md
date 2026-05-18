# HCCL_MULTI_QP_THRESHOLD

## 功能描述

rank间RDMA通信使用多QP通信的场景下，开发者可通过本环境变量设置每个QP分担数据量的最小阈值。

该环境变量需要配置为整数，取值范围：\[1,8192\]，默认值：512，单位：KB。

- 如果“\(rank间单次通信数据量 / HCCL_RDMA_QPS_PER_CONNECTION的值\) < HCCL_MULTI_QP_THRESHOLD的值”，则HCCL执行时会自动减少QP个数，使得每个QP上分担的数据量大于等于HCCL_MULTI_QP_THRESHOLD的值，例如：

    rank间单次通信数据量为1MB，HCCL_RDMA_QPS_PER_CONNECTION配置为4，HCCL_MULTI_QP_THRESHOLD配置为512，此时每个QP最少要求分担512KB的数据量，则HCCL执行时，会减少QP个数为2，仅使用2个QP进行rank间的数据传输。

- 当rank间数据量小于HCCL_MULTI_QP_THRESHOLD时使用单QP传输。
- 当每个QP分担的数据量大于512KB时，使用HCCL Test工具进行RDMA流量测试时（仅测试跨机流量，不使用HCCS链路），多QP场景的下发调度开销相对于单QP场景性能劣化小于3%。

> [!NOTE]说明
> 可通过环境变量[HCCL_RDMA_QPS_PER_CONNECTION](HCCL_RDMA_QPS_PER_CONNECTION.md)或[HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md)开启多QP通信。

## 配置示例

```bash
export HCCL_MULTI_QP_THRESHOLD=512
```

## 使用约束

该环境变量仅支持单算子调用方式，不支持静态图模式。

## 支持的型号

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

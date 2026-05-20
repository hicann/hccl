# HCCL_HOST_SOCKET_PORT_RANGE

## 功能描述

当通信域的创建方式为“基于root节点信息创建”时，开发者可通过此环境变量配置HCCL在Host侧使用的通信端口。

该环境变量支持配置为具体的端口、端口范围或者字符串“auto”。

- 若指定具体的端口号或端口范围，规划的端口数量建议不小于单个NPU上的HCCL进程数，端口号取值范围为\[1,65535\]，且需要确保指定的端口未被其他进程占用。需要注意，\[1,1023\]为系统保留端口，应避免使用这些端口。

    具体的端口号与端口范围可以组合使用，中间使用英文“,”分隔，但逗号之间的端口号/端口范围不能存在范围重叠，配置方式可参见[配置示例](#配置示例)。

- 若指定为字符串“auto”，代表HCCL使用的Host通信端口由操作系统动态分配。

## 配置示例

```bash
# 方式一：配置为端口范围。
export HCCL_HOST_SOCKET_PORT_RANGE="60000-60050"
# 方式二：具体的端口号与端口范围配合使用，使用英文“,”分隔。
export HCCL_HOST_SOCKET_PORT_RANGE="60000,60050-60100,60150-60160"
# 方式三：指定具体的端口号，使用英文“,”分隔。
export HCCL_HOST_SOCKET_PORT_RANGE="56000,56005,56007,56008,56100,56105,56107,56108"
# 方式四：操作系统动态分配端口号
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
```

## 使用约束

- 若业务为单卡多进程场景（即多个业务进程同时共用一个NPU），建议配置此环境变量，否则业务可能会因为端口冲突运行失败。但需要注意，多进程会对资源开销、通信性能产生影响。
- 此环境变量优先级高于[HCCL_IF_BASE_PORT](HCCL_IF_BASE_PORT.md)，若配置了此环境变量，HCCL在Host侧使用的通信端口以此环境变量为准。
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，若网络中存在MC²通算融合算子（计算和通信融合的算子，例如AllGatherMatmul、MatmulReduceScatter、AlltoAllAllGatherBatchMatMul等），不支持配置此环境变量。

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

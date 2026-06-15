# HCCL简介

集合通信库HCCL（Huawei Collective Communication Library）是基于昇腾硬件的高性能集合通信库，为计算集群提供高性能、高可靠的通信方案。

## 核心功能

- 提供单机、多机环境中的高性能集合通信和点对点通信。
- 支持AllReduce、Broadcast、AllGather、ReduceScatter、AlltoAll、Send、Receive等集合通信原语。
- 支持Ring、Mesh、Recursive Halving-Doubling（RHD）等通信算法。
- 支持HCCS、RoCE、PCIe、UB（Unified Bus）等高速通信链路。
- 支持单算子和图模式两种执行模式。
- 支持通信算子的自定义开发。

## 软件架构

HCCL是CANN的核心组件，为NPU集群提供高性能、高可靠性的通信方案。HCCL向上支持多种AI框架，向下实现多款昇腾AI处理器之间的高效互联，其架构如下图所示。

**图1**  集合通信库软件架构图  
![集合通信库软件架构图](figures/hccl_architecture.png)

HCCL包含HCCL集合通信库与HCOMM（Huawei Communication）通信基础库：

- **HCCL集合通信库**：包含内置通信算子和扩展通信算子，提供对外的通信算子接口。
  - 内置通信算子：HCCL提供的基础通信算子，包含集合通信算子和点对点通信算子。
  - 扩展通信算子：用户可以使用HCOMM通信基础库提供的接口自定义扩展通信算子。

- **HCOMM通信基础库**：采用分层解耦的设计思路，将通信能力划分为控制面和数据面两部分。

  - 控制面：提供拓扑信息查询与通信资源管理功能。
  - 数据面：提供本地操作、算子间同步、通信操作等数据搬运和计算功能。

    控制面提供通信资源，数据面提供操作资源的方法，提供的相关接口可以让通信算子开发人员聚焦于业务创新，而无需关注芯片底层复杂的实现细节。

## 支持的产品

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品
<!-- end id5 -->

<!-- npu="910b,310p" id8 -->
> [!NOTE]说明
> <!-- npu="910b" id6 -->
> - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。
> <!-- end id6 -->
> <!-- npu="310p" id7 -->
> - 针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。
<!-- end id7 -->
<!-- end id8 -->

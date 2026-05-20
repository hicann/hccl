# 相关概念

为了您有更好的阅读体验，使用本文档前请先了解HCCL相关概念。

## HCCL基本概念

HCCL的典型通信组网如下图所示。

![典型通信组网示例](figures/typical_network.png)

上图中涉及如下基本概念：

- **AI Server**：又称计算节点，通常是8卡或16卡的昇腾NPU设备组成的服务器形态的统称。
- **AI集群**：多个AI Server通过交换设备互联后用于分布式训练或推理的系统。

    若AI Server间通过灵衢总线交换设备进行连接，组成的组网称之为**超节点组网**。

- **通信成员**：通常称为rank，是参与通信的最小逻辑实体，每个rank都会分配一个唯一标识。
- **通信域**：一组通信成员的组合，描述通信范围。一个计算任务可以创建多个通信域，通信成员也可以加入多个通信域。
- **通信算子**：在通信域内完成通信任务的算子，集合通信指所有成员一起参与的通信操作，如Broadcast、AllReduce等。
- **通信算法**：针对不同网络拓扑、数据量、硬件资源等场景，通信算子通常会采用不同的通信算法实现。

## 术语缩略语

| 名称 | 说明 |
| --- | --- |
| NPU | Neural Network Processing Unit，神经网络处理单元。<br>采用“数据驱动并行计算”的架构，擅长处理海量的视频和图像类多媒体业务数据，专门用于处理人工智能应用中的大量计算任务。 |
| HCCL | Huawei Collective Communication Library，华为集合通信库。<br>提供单机多卡以及多机多卡间的数据并行、模型并行集合通信方案。 |
| HCOMM | Huawei Communication，华为通信基础库。 |
| HCCS | Huawei Cache Coherence System，华为缓存一致性系统。<br>用于CPU/NPU之间的高速互联。 |
| HCCP | Huawei Collective Communication adaptive Protocol，集合通信适配协议。<br>提供跨NPU设备通信能力，向上屏蔽具体通信协议差异。 |
| TOPO | 拓扑、拓扑结构。<br>一个局域网内或者多个局域网之间的设备连接所构成的网络配置或者布置。 |
| PCIe | Peripheral Component Interconnect Express，一种串行外设扩展总线标准，常用于计算机系统中的外设扩展。 |
| PCIe-SW | PCIe Switch，符合PCIe总线扩展的交换设备。 |
| QP | Queue Pair，队列对。<br>QP是远程直接内存访问技术的核心通信单元，由发送队列（Send Queue，SQ）和接收队列（Receive Queue，RQ）组成，用于管理数据传输任务。 |
| SDMA | System Direct Memory Access，系统直接内存访问技术，简称DMA，允许外围设备直接访问系统内存，而不需要CPU的干预。 |
| RDMA | Remote Direct Memory Access，远程直接内存访问技术，能够直接将数据从一台机器的内存传输到另一台机器，无需双方操作系统的介入，一般指可以跨网络的内存访问方式。 |
| RoCE | RDMA over Converged Ethernet，承载在融合以太网上的RDMA技术，即跨越以太网的RDMA通信方式。 |
| AIV | AI Core中的Vector Core。 |
| TS | Task Scheduler，任务调度器。 |
| CCU | Collective Communication Unit，集合通信加速单元。 |

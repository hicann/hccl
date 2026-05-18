# 算法简介

针对同一个通信算子，随着网络拓扑、通信数据量、硬件资源等的不同，往往会采用不同的通信算法，从而最大化集群通信性能。HCCL提供了Mesh、Ring、Recursive Halving-Doubling（RHD）、Pairwise、Pipeline等拓扑算法用于Server内、Server间和超节点间的集合通信。

## Server内通信算法

HCCL通信域Server内支持Mesh、Ring、Double-Ring和Star算法，具体使用的算法根据硬件拓扑自动选择，用户无须配置也不支持配置。

## Server间/超节点间通信算法

HCCL通信域Server间/超节点间支持如下算法的自适应选择，自适应算法会根据产品形态、数据量和Server个数进行选择，用户默认无需配置。

- Ring算法：基于环结构的通信算法，通信步数多（线性复杂度），时延相对较高，但通信关系简单，受网络拥塞影响较小。适合通信域内Server个数较少、通信数据量较小、网络存在明显拥塞、且Pipeline算法不适用的场景。
- RHD（Recursive Halving-Doubling）算法：递归二分和倍增算法，通信步数少（对数复杂度），时延相对较低，但在非2次幂节点规模下会引入额外的通信量。适合通信域内Server个数是2的整数次幂且Pipeline算法不适用的场景，或Server个数不是2的整数次幂但通信数据量较小的场景。
- NHR（Nonuniform Hierarchical Ring）算法：非均衡的层次环算法，通信步数少（对数复杂度），时延相对较低。适合通信域内Server个数较多且Pipeline算法不适用的场景。
- NB（Nonuniform Bruck）算法：非均匀的数据块通信算法，通信步数少（对数复杂度），时延相对较低。适合通信域内Server个数较多且Pipeline算法不适用的场景。
- Pipeline算法：流水线并行算法，可并发使用Server内与Server间的链路，或超节点内与超节点间的链路，适合通信数据量较大且通信域内每机包含多卡的场景。
- Pairwise算法：逐对通信算法，仅用于AlltoAll、AlltoAllV与AlltoAllVC算子，通信步数较多（线性复杂度），时延相对较高，但可以避免网络中出现一打多现象（指一个rank通过同一个端口给多个rank发送数据），适合通信数据量较大、需要规避网络一打多现象的场景。
- AHC（Asymmetric Hierarchical Concatenate）算法：层次化集合通信算法，仅用于ReduceScatter、AllGather、AllReduce算子，适用于通信域内NPU分布存在多个层次、同时支持多个层次间NPU对称或者非对称分布的场景，当通信域内层次间存在带宽收敛时相对收益会更好。

> [!NOTE]说明 
>
> - 开发者若想指定Server间或超节点间通信算法，可通过环境变量[HCCL_ALGO](../hccl_env/HCCL_ALGO.md)进行设置。需要注意，若通过环境变量HCCL\_ALGO指定了Server间或超节点间通信算法，通信算法的自适应选择功能不再生效，以用户指定的算法为准。
> - 每种算法支持的算子以及产品可参见环境变量[HCCL_ALGO](../hccl_env/HCCL_ALGO.md)。

## 耗时评估

HCCL采用α–β模型（Hockney）进行性能评估，算法耗时计算用到的变量定义如下：

- α：节点间的固定时延，单位为s，由通信硬件设备与底层软件栈决定。
- β：每byte数据传输耗时，单位为s/byte，由通信链路能力决定。
- n：节点间通信的数据大小，单位为byte，由通信算法决定。
- γ：每byte数据归约计算耗时，单位为s/byte，由计算硬件设备能力决定。
- p：通信域节点个数，影响通信步数，由通信算子所在通信域决定。

单步传输并归约计算n byte数据的耗时为： D = α + nβ + nγ。

集合通信算法通过结合网络拓扑，优化通信关系和通信步骤，减少通信次数以减少固定时延，减少实际通信数据量以减少传输耗时和计算耗时，从而达到优化集合通信性能的目的。

## 分级通信原理

HCCL通常按节点内/节点间分为两级拓扑、或按节点内/节点间/超节点间分为三级拓扑，分级执行集合通信，不同的层级间链路带宽不同。分级通信可以使通信任务编排与网络拓扑亲和，从而最大化利用链路能力。

以Atlas A2 训练系列产品/Atlas A2 推理系列产品的单算子模式、节点内/节点间两级拓扑为例，各集合通信算子的具体分级通信过程如下表所示：

| 集合通信算子 | 阶段一 | 阶段二 | 阶段三 |
| --- | --- | --- | --- |
| ReduceScatter | Server间ReduceScatter | Server内ReduceScatter | / |
| ReduceScatterV | Server间ReduceScatterV | Server内ReduceScatterV | / |
| AllGather | Server内AllGather | Server间AllGather | / |
| AllGatherV | Server内AllGatherV | Server间AllGatherV | / |
| AllReduce | Server内ReduceScatter | Server间AllReduce | Server内AllGather |
| Scatter | Server间Scatter | Server内Scatter | / |
| Broadcast | Server内Scatter | Server间Broadcast | Server内AllGather |
| Reduce | Server内ReduceScatter | Server间Reduce | Server内Gather<br>HCCL未提供Gather算子，Gather操作与AllGather操作的区别是仅将结果发送到root节点的输出buffer。 |
| AlltoAll | Server内AlltoAll | Server间AlltoAll | / |
| AlltoAllV | Server内AlltoAllV | Server间AlltoAllV | / |
| AlltoAllVC | Server内AlltoAllVC | Server间AlltoAllVC | / |

详细分级通信流程示例可参见[分级通信原理](hierarchical_comm_principle.md)。

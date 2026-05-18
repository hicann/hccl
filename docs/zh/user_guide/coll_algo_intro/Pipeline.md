# Pipeline

## 算法描述

为降低网络流量冲突，AI计算集群中常常采用分级网络架构，即Server内通过直接连接的电缆互联，Server间同号卡通过交换机互联。为适配这种网络拓扑结构，集合通信采用分级算法策略，即将全局通信操作分解为多个层级的局部操作，利用分阶段、分层递进的方法以提升通信效率。

以AllGather算子为例，Server间先执行一次同号卡间的AllGather操作，再在Server内执行一次AllGather操作，以完成整个集群的数据集合过程。然而，这种方式会造成一定程度上的链路带宽浪费：当Server间数据传输时，Server内的链路处于空闲状态，未能充分利用带宽资源。

为了解决这一问题，HCCL采用了细粒度的分级流水（pipeline）算法，通过挖掘通信算法本身的数据依赖，并且结合流水并行的方式，解决带宽利用率不足的问题。

以AllGather为例，Server间通信选择Ring算法，Server内通信选择FullMesh算法，如下图所示。

**图 1**  AllGather算子Pipeline算法示意图

![](figures/pipeline.png)

如上图所示，绿色数据块从Rank5被发送到Rank1上（此处仅描述部分Rank的行为，其它Rank对称处理），在接下来的步骤中，Rank1继续向Rank3发送绿色数据块（Ring算法标准步骤），同时Rank1也可以向同Server中的Rank0发送绿色数据块。继续执行Ring算法，每一步在进行Server间数据传输的同时，还会向Server内其它Rank传输上一步接收到的数据块，Ring算法的最后一个步骤结束后，仅需要在Server内再进行一次数据块的传输即可完成全部算法步骤（Rank初始数据块的Server内传输操作，可以隐藏在Ring算法的第一步中进行）。

在Rank0视角上，全部传输任务编排如下图所示，LocalCopy操作仅在输入输出内存不同的场景中执行，用于将数据块从输入内存移动到输出内存，在输入输出内存相同场景中，则无需执行该操作。

**图 2**  AllGather算子Pipeline算法时序示意图

![](figures/allgather_pipeline.png)

## 耗时计算

**表1**   Pipeline算法中各操作计算耗时

| 操作          | 耗时                                                         |
| ------------- | ------------------------------------------------------------ |
| ReduceScatter | $max(\frac{s}{p} * \beta_{inter} + \alpha_{inter} , \frac{s}{p} * \beta_{intra} + \alpha_{intra}) * (p_{inter} -1) + \frac{s}{p} * \beta_{intra} + \alpha_{intra}$ |
| AllGather     | $max(\frac{s}{p} * \beta_{inter} + \alpha_{inter} , \frac{s}{p} * \beta_{intra} + \alpha_{intra}) * (p_{inter} -1) + \frac{s}{p} * \beta_{intra} + \alpha_{intra}$ |
| AllReduce     | $2*(max(\frac{s}{p} * \beta_{inter} +  \alpha_{inter}, \frac{s}{p} * \beta_{intra}+\alpha_{intra} ) * (p_{inter}-1)+ \frac{s}{p} * \beta_{intra} + \alpha_{intra})$ |

其中p表示完成集合通信的总卡数，$p_{inter}$表示server数，s表示集合通信操作总数据量，$\beta_{inter}$表示server间链路每byte数据传输耗时，$\beta_{intra}$表示server内链路每byte数据传输耗时，$\alpha_{inter}$表示server间链路传输固定耗时，$\alpha_{intra}$表示server内链路传输固定耗时。

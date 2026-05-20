# NHR

## 算法描述

在规模较大、通信节点数量较多的组网中，由于Ring算法的通信步数与通信节点规模成正相关，所以容易出现较大延迟，因此Ring算法的切分数据块的方式对小数据包场景不友好，更适合大数据包传输场景。在集群规模不是2的整数次幂时，使用RHD算法会引入额外的通信步数和开销，产生N-1规模集群下的通信性能比N规模集群通信性能还差的现象。此外，由于RHD算法每个通信阶段的对象在变化，导致通信链路也发生变化，这在大流量场景可能会引起交换机的流量冲突，从而导致带宽下降。

NHR（Nonuniform Hierarchical Ring，非均衡的层次环）算法对N个节点构建N棵生成树，通过N个生成树构建出最优通信关系。树的深度（即通信步数）是$⌈log_2⁡N⌉$，并通过重排数据片编号聚合发送，保证通信算法的理论性能最优。

该算法的最大通信流量集中在物理位置相近的节点之间，能够有效利用物理距离带来的性能优势，减少流量冲突。同时无论集群规模是否是2的幂次，NHR算法都能充分利用链路资源。对于小数据包通信场景，NHR算法进一步优化，采用N个节点仅构建1棵树的方法，从而减少网络中数据包数量和芯片并发执行的任务数，提升通信效率。

rank size是2的整数次幂时（以rank size等于4为例），NHR算法的通信过程如下图所示，可以看到过程中每一步的收发数据份数均为1，因为地址连续的数据切片能够被连续发送。

**图 1**  rank size为4时NHR算法通信过程 
![](figures/nhr_algo_4rank_flow.png "rank-size为4时NHR算法通信过程")

rank size不是2的整数次幂时（以rank size等于5为例），NHR算法的通信过程如下图所示，大部分的数据切片可以连续收发，只有少部分卡间的数据切片是离散的。

**图 2**  rank size为5时NHR算法通信过程 
![](figures/nhr_algo_5rank_flow.png "rank-size为5时NHR算法通信过程")

NHR算法同样适用于“星型”或“胖树”拓扑互联，算法的时间复杂度是$⌈log_2⁡N⌉$。

## 耗时计算

NHR为非均衡的层次环算法，当集群规模为2的整数次幂或者非2的整数次幂时，算法时间复杂度均为$O(⌈log_2⁡N⌉)$。如果节点数为p，则需要的通信次数为$⌈log_2⁡p⌉$，对ReduceScatter算子，第一步交换$n/2$的数据，每次通信数据量减半，最后一步交换1份数据，AllGather算子的收发关系则完全相反。

**表1**   NHR算法中各操作计算耗时

| 操作          | 耗时                                                         |
| ------------- | ------------------------------------------------------------ |
| ReduceScatter | $\lceil log_2⁡p\rceil\alpha + \frac{p−1}{p}n\beta + \frac{p−1}{p}n\gamma$ |
| AllGather     | 耗时同ReduceScatter，无$\gamma$相关部分<br/>$\lceil log_2⁡p\rceil\alpha + \frac{p−1}{p}n\beta$ |
| AllReduce     | 实现为ReduceScatter + AllGather：<br>$2\lceil log_2⁡p\rceil\alpha + 2\frac{p−1}{p}n\beta + \frac{p−1}{p}n\gamma$ |
| Scatter       | $\lceil log_2⁡p\rceil\alpha + \frac{p−1}{p}n\beta$            |
| Broadcast     | 实现为Scatter + AllGather：<br>$2\lceil log_2⁡p\rceil\alpha + 2\frac{p−1}{p}n\beta$ |

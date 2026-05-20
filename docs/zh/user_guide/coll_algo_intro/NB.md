# NB

## 算法描述

集合通信中，Ring算法通信步数为$O(N-1)$，其中N表示参与集合通信的rank数量，随着网络规模的增加，通信开销也会显著增加。RHD算法虽然将通信步数减少到了$log_2⁡N$，但在rank数量不是2的幂时，需要进行数据合并操作，导致通信数据量增加。而NB算法（Nonuniform Bruck，非均匀的数据块通信算法）通过动态调整步长的多重环状结构，实现不同rank数量下均保持通信步数为$⌈log_2⁡N⌉$，同时避免了额外的通信数据量增长。

rank size是2的幂时，NB算法的通信过程如下图所示（以rank size等于4为例）。

**图 1**  rank size为4时NB算法通信过程
![](figures/nb_algo_4rank_flow.png "rank-size为4时NB算法通信过程")

rank size不是2的幂时，NB算法的通信过程如下图所示（以rank size等于5为例）。

**图 2**  rank size为5时NB算法通信过程  
![](figures/nb_algo_5rank_flow.png "rank-size为5时NB算法通信过程")

对于ReduceScatter和AllGather算子，通信步数均为$⌈log_2⁡N⌉$。

- 针对ReduceScatter算子，每一步通信中，每张卡向通信步长为$2^k(0 \leq k<⌈log2(N)⌉)$的目标卡发送数据，每步发送数据的份数为$⌊(N-1+2^k)/2^{k+1}⌋$。
- 对于AllGather算子，每一步的通信步长递减，而通信数据量递增。当卡数不是2的幂时，最后一步的通信数据量为$N-2^{⌊log2(N)⌋}$。

NB算法同样适用于“星型”和“胖树”拓扑，算法的时间复杂度为$⌈log_2⁡N⌉$。

## 耗时计算

**表 1**  NB算法中各操作耗时

| 操作          | 耗时                                                         |
| ------------- | ------------------------------------------------------------ |
| ReduceScatter | $\lceil log(p)\rceil\alpha + \frac{p−1}{p}n\beta + \frac{p−1}{p}n\gamma$ |
| AllGather     | $\lceil log(⁡p)\rceil\alpha + \frac{p−1}{p}n\beta$ |
| AllReduce     | 实现为ReduceScatter + AllGather ，耗时为：<br>$2\lceil log(⁡p)\rceil\alpha + 2\frac{p−1}{p}n\beta + \frac{p−1}{p}n\gamma$ |
| Scatter       | $\lceil log(⁡p)\rceil\alpha + \frac{p−1}{p}n\beta$            |
| Broadcast     | 实现为Scatter + AllGather，耗时为：<br>$2\lceil log(⁡p)\rceil\alpha + 2\frac{p−1}{p}n\beta$ |

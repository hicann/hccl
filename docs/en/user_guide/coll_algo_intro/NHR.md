# NHR

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:53:45.770Z pushedAt=2026-08-04T07:26:38.609Z -->

## Algorithm Description

In large-scale networks with many communication nodes, the Ring algorithm tends to incur significant latency because its number of communication steps grows linearly with the number of nodes. Moreover, its data-splitting approach is unfriendly to small-packet scenarios and better suited for large-packet transmission. When the cluster size is not a power of two, the RHD algorithm introduces extra steps and overhead, leading to a phenomenon where the communication performance of an N-1 cluster is worse than that of an N cluster. In addition, since the RHD algorithm changes the communication peer at each stage, the communication links vary as well. This can cause traffic collisions on switches in high-traffic scenarios, potentially reducing bandwidth.

The NHR (Nonuniform Hierarchical Ring) algorithm constructs N spanning trees for N nodes and establishes optimal communication relationships through these N spanning trees. The tree depth (i.e., the number of communication steps) is $⌈log_2⁡N⌉$, and data slice indices are rearranged for aggregated sending, ensuring optimal theoretical performance of the algorithm communication.

The maximum communication traffic of this algorithm is concentrated between physically adjacent nodes, which effectively leverages the performance benefits of physical proximity and reduces traffic collisions. Moreover, the NHR algorithm can fully utilize link resources regardless of whether the cluster size is a power of two. For small-packet communication scenarios, the algorithm is further optimized by constructing only one tree for the N nodes, which reduces the number of packets in the network and the number of concurrent chip tasks, thereby improving communication efficiency.

When the rank size is an integer power of 2 (taking rank size = 4 as an example), the communication process of the NHR algorithm is shown in the following figure. It can be seen that the number of data copies sent and received at each step is 1, because data slices with contiguous addresses can be sent continuously.

**Figure 1** NHR algorithm communication process when rank size is 4
![](figures/nhr_algo_4rank_flow.png "NHR algorithm communication process when rank size is 4")

When the rank size is not an integer power of 2 (taking rank size 5 as an example), the NHR algorithm communication process is shown in the following figure. Most data slices can be sent and received continuously, and only a few data slices between cards are discrete.

**Figure 2** NHR algorithm communication process when rank size is 5
![](figures/nhr_algo_5rank_flow.png "NHR algorithm communication process when rank size is 5")

The NHR algorithm is also applicable to star or fat-tree topology interconnects, and the time complexity of the algorithm is $⌈log_2⁡N⌉$.

## Time Consumption Calculation

NHR is a nonuniform hierarchical ring algorithm. Regardless of whether the cluster size is a power of 2 or not, the time complexity of the algorithm is $O(⌈log_2⁡N⌉)$. If the number of nodes is p, the required number of communication steps is $⌈log_2⁡p⌉$. For the ReduceScatter operator, the first step exchanges $n/2$ of data, the data volume per communication is halved each time, and the last step exchanges one unit of data. The send/receive relationship of the AllGather operator is the exact opposite.

**Table 1** Time consumption calculation for each operation in the NHR algorithm

| Operation     | Time Consumption                                                      |
| ------------- | ------------------------------------------------------------ |
| ReduceScatter | $\lceil log_2⁡p\rceil\alpha + \frac{p−1}{p}n\beta + \frac{p−1}{p}n\gamma$ |
| AllGather     | Same time consumption as ReduceScatter, without the $\gamma$-related term<br/>$\lceil log_2⁡p\rceil\alpha + \frac{p−1}{p}n\beta$ |
| AllReduce     | Implemented as ReduceScatter + AllGather:<br>$2\lceil log_2⁡p\rceil\alpha + 2\frac{p−1}{p}n\beta + \frac{p−1}{p}n\gamma$ |
| Scatter       | $\lceil log_2⁡p\rceil\alpha + \frac{p−1}{p}n\beta$            |
| Broadcast     | Implemented as Scatter + AllGather:<br>$2\lceil log_2⁡p\rceil\alpha + 2\frac{p−1}{p}n\beta$ |

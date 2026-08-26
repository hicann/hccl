# NB

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:54:10.237Z pushedAt=2026-08-04T03:55:45.861Z -->

## Algorithm Description

In collective communication, the Ring algorithm has a communication step count of $O(N-1)$, where N represents the number of ranks participating in the collective communication. As the network scale increases, the communication overhead also increases significantly. Although the RHD algorithm reduces the communication step count to $log_2⁡N$, when the number of ranks is not a power of 2, data merging operations are required, leading to an increase in communication data volume. In contrast, the NB algorithm (Nonuniform Bruck) uses a multi-ring structure with dynamically adjusted step sizes to maintain a communication step count of $⌈log_2⁡N⌉$ for any number of ranks, while avoiding additional growth in communication data volume.

When the rank size is a power of 2, the NB algorithm communication process is shown in the following figure (using a rank size of 4 as an example).

**Figure 1**  NB algorithm communication process when the rank size is 4
![](figures/nb_algo_4rank_flow.png "NB algorithm communication process when the rank size is 4")

When the rank size is not a power of 2, the NB algorithm communication process is shown in the following figure (using a rank size of 5 as an example).

**Figure 2**  NB algorithm communication process when rank size is 5  
![](figures/nb_algo_5rank_flow.png "NB algorithm communication process when rank size is 5")

For ReduceScatter and AllGather operators, the number of communication steps is $⌈log_2⁡N⌉$.

- For the ReduceScatter operator, in each communication step, each card sends data to the target card with a communication stride of $2^k(0 \leq k<⌈log2(N)⌉)$, and the number of data blocks sent per step is $⌊(N-1+2^k)/2^{k+1}⌋$.

- For the AllGather operator, the communication stride decreases while the communication data volume increases in each step. When the number of cards is not a power of two, the communication data volume in the last step is $N-2^{⌊log2(N)⌋}$.

The NB algorithm is also applicable to star and fat-tree topologies, with a time complexity of $⌈log_2⁡N⌉$.

## Time Calculation

**Table 1** Time consumption of each operation in the NB algorithm

| Operation     | Time                                                         |
| ------------- | ------------------------------------------------------------ |
| ReduceScatter | $\lceil log(p)\rceil\alpha + \frac{p−1}{p}n\beta + \frac{p−1}{p}n\gamma$ |
| AllGather     | $\lceil log(p)\rceil\alpha + \frac{p−1}{p}n\beta$ |
| AllReduce     | Implemented as ReduceScatter + AllGather, and the time consumption is:<br>$2\lceil log(p)\rceil\alpha + 2\frac{p−1}{p}n\beta + \frac{p−1}{p}n\gamma$ |
| Scatter       | $\lceil log(p)\rceil\alpha + \frac{p−1}{p}n\beta$            |
| Broadcast     | Implemented as Scatter + AllGather, and the time consumption is:<br>$2\lceil log(p)\rceil\alpha + 2\frac{p−1}{p}n\beta$ |

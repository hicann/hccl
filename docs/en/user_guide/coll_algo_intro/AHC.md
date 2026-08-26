# AHC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:52:56.200Z pushedAt=2026-08-03T09:16:48.856Z -->

## Algorithm Description

When a cluster network exhibits hierarchical characteristics with bandwidth convergence between layers, collective communication faces two major technical challenges. First, due to bandwidth convergence between different regions, the performance of traditional single-layer collective communication algorithms degrades. Second, the number of compute units varies across different regions (i.e., asymmetric card counts), rendering conventional hierarchical algorithms inapplicable. For example, in a cluster, the same communication domain may span two SuperPoDs, and the number of cards in the two SuperPoDs is inconsistent (e.g., one SuperPoD has 64 cards while the other has 128 cards). This scenario poses significant challenges to the performance of collective communication algorithms.

**Figure 1**  AHC implementing the AllReduce process based on logical same-number cards (5 ranks, two groups of 2+3)
![](figures/ahc_allreduce_5rank_flow.png "AHC implementing the AllReduce process based on logical same-number cards (5 ranks, two groups of 2+3)")

The core idea of this algorithm is to regroup the NPUs and their data within the communication domain based on topology. Within each group, high-speed network bandwidth is fully utilized; between groups, asymmetric concatenation is implemented based on logical same-number cards. The specific process is illustrated in the figure above and consists of the following three steps:

1. Group compute units based on physical topology. Adjacent NPUs are divided into a group. The number of cards in each group does not need to be identical, and bandwidth between groups may exhibit convergence compared to bandwidth within a group.

    1. Calculate the least common multiple (LCM) of the numbers of all groups. If there are G groups, the data is divided into LCM\*G slices. As shown in the figure above, with groups of 2 and 3, LCM=6 and G=2, the data is divided into 12 slices.

    2. Standard ReduceScatter is executed in parallel within each group.

2. Define "logical same-number" cards and implement inter-group AllReduce based on these cards.

    1. The data to be reduced in each group is split according to the data boundaries between NPU cards within the group, forming several uneven data blocks.

    2. Each data segment within a group has a counterpart of equal size in every other group. Based on this data correspondence, a corresponding relationship is also established between NPUs across groups. NPUs that share such a correspondence are referred to as “logical same-number” cards.

    3. Perform AllReduce across logical same-number cards.

3. Perform AllGather across NPUs within each group.

For specific intra-group and inter-group operations such as ReduceScatter, AllGather, and AllReduce, any known algorithm (e.g., NB, NHR, Ring) can be used for implementation. The current AHC algorithm internally selects the splicing algorithm type that delivers better performance based on specific scenarios and strategies.

## Time Consumption Calculation

When the NB algorithm is used for both intra-group and inter-group operations, the algorithm time consumption of the AllReduce operator is as follows:

**Table 1** Time consumption of the AHC algorithm

| Operation      | Time Consumption            |
| ------------- | --------------------------------------------------------------------- |
| ReduceScatter | $2(\lceil log(m+d)\rceil + \lceil log(G)\rceil)\alpha + 2(\frac{m+d−1}{m+d}+  \frac{(G-1)*C}{Gm}n\beta + (\frac{m+d-1}{m+d} + \frac{G-1}{Gm})n\gamma$ <br> Where m is the minimum number of groups, m+d is the maximum number of groups, G is the number of groups, and C is the convergence ratio of inter-group bandwidth to intra-group bandwidth.|

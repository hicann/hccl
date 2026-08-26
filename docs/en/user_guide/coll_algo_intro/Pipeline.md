# Pipeline

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:54:16.723Z pushedAt=2026-08-04T09:20:01.917Z -->

## Algorithm Description

To reduce network traffic contention, AI computing clusters often adopt a hierarchical network architecture, where intra-server connections use directly attached cables and inter-server connections between same-index cards use switches. To accommodate this network topology, collective communication employs a hierarchical algorithm strategy, which decomposes global communication operations into local operations at multiple levels and uses a phased, layer-by-layer progressive approach to improve communication efficiency.

Taking the AllGather operator as an example, the operation is first performed across servers among same-index cards, followed by an AllGather within each server. This completes the data gathering process for the entire cluster. However, this approach results in some waste of link bandwidth: when data is being transmitted between servers, the intra-server links remain idle and fail to fully utilize the available bandwidth.

To address this issue, HCCL adopts a fine-grained hierarchical pipeline algorithm. By leveraging the data dependencies inherent in the communication algorithm itself and incorporating pipeline parallelism, it resolves the problem of insufficient bandwidth utilization.

Taking AllGather as an example, the Ring algorithm is selected for inter-server communication and the FullMesh algorithm for intra-server communication, as shown in the following figure.

**Figure 1** Schematic diagram of the Pipeline algorithm for the AllGather operator

![](figures/pipeline.png)

As shown in the figure above, the green data block is sent from Rank5 to Rank1 (only the behavior of some Ranks is described here; other Ranks are processed symmetrically). In the next step, Rank1 continues to send the green data block to Rank3 (a standard step of the Ring algorithm), while Rank1 can also send the green data block to Rank0 in the same server. As the Ring algorithm proceeds, at each step, while inter-server data transmission is in progress, the data block received in the previous step is also transmitted to other Ranks within the server. After the last step of the Ring algorithm is completed, only one more intra-server data block transmission is needed to finish all algorithm steps (the intra-server transmission of the Rank's initial data block can be hidden within the first step of the Ring algorithm).

From the perspective of Rank0, the orchestration of all transmission tasks is shown in the figure below. The LocalCopy operation is executed only when the input and output memory are different, and is used to move the data block from the input memory to the output memory. When the input and output memory are the same, this operation does not need to be executed.

**Figure 2** Timing diagram of the Pipeline algorithm for the AllGather operator

![](figures/allgather_pipeline.png)

## Time Consumption Calculation

**Table 1** Time consumption of each operation in the Pipeline algorithm

| Operation     | Time                                                         |
| ------------- | ------------------------------------------------------------ |
| ReduceScatter | $max(\frac{s}{p} * \beta_{inter} + \alpha_{inter} , \frac{s}{p} * \beta_{intra} + \alpha_{intra}) * (p_{inter} -1) + \frac{s}{p} * \beta_{intra} + \alpha_{intra}$ |
| AllGather     | $max(\frac{s}{p} * \beta_{inter} + \alpha_{inter} , \frac{s}{p} * \beta_{intra} + \alpha_{intra}) * (p_{inter} -1) + \frac{s}{p} * \beta_{intra} + \alpha_{intra}$ |
| AllReduce     | $2*(max(\frac{s}{p} * \beta_{inter} +  \alpha_{inter}, \frac{s}{p} * \beta_{intra}+\alpha_{intra} ) * (p_{inter}-1)+ \frac{s}{p} * \beta_{intra} + \alpha_{intra})$ |

In the preceding table, p denotes the total number of cards participating in the collective communication, $p_{inter}$ denotes the number of servers, s denotes the total data volume of the collective communication operation, $\beta_{inter}$ denotes the per-Byte data transmission time over inter-server links, $\beta_{intra}$ denotes the per-byte data transmission time over intra-server links, $\alpha_{inter}$ denotes the fixed transmission overhead over inter-server links, and $\alpha_{intra}$ denotes the fixed transmission overhead over intra-server links.

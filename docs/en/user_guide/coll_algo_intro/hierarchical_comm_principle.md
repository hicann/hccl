# Hierarchical Communication Principles

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:53:06.010Z pushedAt=2026-08-04T01:59:54.821Z -->

The following uses the communication operators ReduceScatter, AllGather, and AllReduce as examples to introduce the hierarchical communication flow.

## ReduceScatter

As shown in the figure below, the ReduceScatter operator requires that rank i ultimately obtains the i-th reduction result. To ensure the continuity of data blocks exchanged between servers, the ReduceScatter operation is first performed across servers, followed by the ReduceScatter operation within each server.

**Figure 1**  ReduceScatter operator hierarchical communication flow
![](figures/reduce_scatter_hierarchical_flow.png "ReduceScatter operator hierarchical communication flow")

## AllGather

As shown in the following figure, the AllGather operator requires that the input data of the i-th rank appears at the i-th position of the result. To ensure the continuity of communication data blocks between servers, an AllGather operation is first performed within each server, followed by an AllGather operation between servers.

**Figure 2** AllGather operator hierarchical communication flow
![](figures/allgather_hierarchical_flow.png "AllGather operator hierarchical communication flow")

## AllReduce

As shown in the figure below, the AllReduce operator produces a complete reduction result. Although it is decomposed into two phases, ReduceScatter and AllGather, the semantics of these two operators do not need to be strictly followed. This allows the higher-bandwidth intra-server communication to handle larger data volumes. The recommended procedure is to first perform ReduceScatter within each server, then perform AllReduce across servers, and finally perform AllGather within each server.

**Figure 3**  AllReduce operator hierarchical communication flow
![](figures/allreduce_hierarchical_flow.png "AllReduce operator hierarchical communication flow")

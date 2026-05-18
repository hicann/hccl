# 典型算子行为分析

以Atlas 800T A2双机场景下，AllReduce算子的Profiling数据为例，介绍如何将通信算子的任务编排与Profiling中的task对应，下图为其中一个rank上完整的AllReduce算子执行流程，同时将AllReduce的各个算子执行步骤与Profiling进行对应。

![AllReduce算子执行流程](figures/allreduce_task.png)

1. 将通信数据从用户输入内存拷贝至HCCL Buffer内存中。

    ![usermen_to_hcclbuffer](figures/usermen_to_hcclbuffer.png)

2. 节点内实现ReduceScatter通信语义，包括notify前同步、ReduceInline内存拷贝、随路运算以及notify尾同步。

    ![reducescatter_task](figures/reducescatter_task.png)

3. 节点间实现AllReduce通信语义。由于节点间通过RoCE来实现notify同步及数据的通信，且notify record任务及数据通信任务均已RDMASend下发WQE的形式实现，因此在Profiling中会以RDMASend（notify record） + notify wait的组合对应着机间前同步和尾同步任务，同时会以RDMASend（数据通信）+ RDMASend（notify record） + notify wait的组合对应着机间的数据通信。

    ![节点间allreduce](figures/inter_allreduce.png)

    此外可以在RDMASend（数据通信）任务的详细信息中获取该任务的本端、对端、数据量及带宽信息等。

    ![rdma_send](figures/rdma_send.png)

4. 节点内实现AllGather通信语义，包括notify前同步、memcpy内存拷贝以及notify尾同步。

    ![节点内allgather](figures/inner_allgather.png)

5. 将通信数据从HCCL Buffer拷贝到用户输出内存中。

    ![hcclbuffer_to_usermem](figures/hcclbuffer_to_usermem.png)

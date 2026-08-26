# Typical Operator Behavior Analysis

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-10T09:34:19.547Z pushedAt=2026-08-10T12:14:25.224Z -->

Taking the profile data of the AllReduce operator on dual-server Atlas 800T A2 as an example, this section describes how to map the task orchestration of a communication operator to tasks in profiling. The following figure shows the complete AllReduce operator execution flow on one rank, and maps each execution step of the AllReduce operator to profiling.

![AllReduce operator execution flow](figures/allreduce_task.png)

1. Copy communication data from the user input memory to the HCCL Buffer memory.

    ![usermen_to_hcclbuffer](figures/usermen_to_hcclbuffer.png)

2. Implement intra-node ReduceScatter communication semantics, including pre-notify synchronization, ReduceInline memory copy, inline computation, and post-notify synchronization.

    ![reducescatter_task](figures/reducescatter_task.png)

3. Implement inter-node AllReduce communication semantics. Since inter-node notify synchronization and data communication are implemented through RoCE, and both notify record tasks and data communication tasks are implemented by delivering WQEs via RDMASend, the combination of RDMASend (notify record) and notify wait corresponds to the inter-node pre-synchronization and post-synchronization tasks in profiling. Meanwhile, the combination of RDMASend (data communication), RDMASend (notify record), and notify wait corresponds to the inter-node data communication.

    ![Inter-node AllReduce](figures/inter_allreduce.png)

    In addition, you can obtain information such as the local end, peer end, data volume, and bandwidth of the task from the detailed information of the RDMASend (data communication) task.

    ![rdma_send](figures/rdma_send.png)

4. Implement intra-node AllGather communication semantics, including pre-notify synchronization, memcpy, and post-notify synchronization.

    ![inner_allgather](figures/inner_allgather.png)

5. Copy communication data from the HCCL Buffer to the user output memory.

    ![hcclbuffer_to_usermem](figures/hcclbuffer_to_usermem.png)
    
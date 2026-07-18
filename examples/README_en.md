# HCCL Code Examples

This directory provides sample code for using the HCCL interface to implement collective communication in different scenarios.

## Point-to-Point Communication

- [HcclSend and HcclRecv (Basic Send and Receive)](./01_point_to_point/01_send_recv)
- [HcclBatchSendRecv (Ring Communication)](./01_point_to_point/02_batch_send_recv_ring)

## Collective Communication

- [AllReduce](./02_collectives/01_allreduce)
- [Broadcast](./02_collectives/02_broadcast)
- [AllGather](./02_collectives/03_allgather)
- [ReduceScatter](./02_collectives/04_reduce_scatter)
- [Reduce](./02_collectives/05_reduce)
- [AlltoAll](./02_collectives/06_alltoall)
- [AlltoAllV](./02_collectives/07_alltoallv)
- [AlltoAllVC](./02_collectives/08_alltoallvc)
- [Scatter](./02_collectives/09_scatter)

## AI Frameworks

- [PyTorch](./03_ai_framework/01_pytorch)
- [TensorFlow](./03_ai_framework/02_tensorflow)

## Custom Point-to-Point Communication Operator

- [Custom Send and Recv Operator (Based on AICPU Communication Engine)](./04_custom_ops_p2p)

## Custom Collective Communication Operator

- [Custom AllGather Operator](./05_custom_ops_allgather)

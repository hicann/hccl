# HCCL 代码示例

本目录提供了不同场景下使用 HCCL 接口实现集合通信功能的示例代码。

## 点对点通信

- [HcclSend/HcclRecv（基础收发功能）](./01_point_to_point/01_send_recv/)
- [HcclBatchSendRecv（实现 Ring 环状通信）](./01_point_to_point/02_batch_send_recv_ring/)

## 集合通信

- [AllReduce](./02_collectives/01_allreduce/)
- [Broadcast](./02_collectives/02_broadcast/)
- [AllGather](./02_collectives/03_allgather/)
- [ReduceScatter](./02_collectives/04_reduce_scatter/)
- [Reduce](./02_collectives/05_reduce/)
- [AlltoAll](./02_collectives/06_alltoall/)
- [AlltoAllV](./02_collectives/07_alltoallv/)
- [AlltoAllVC](./02_collectives/08_alltoallvc/)
- [Scatter](./02_collectives/09_scatter/)

## AI 框架

- [PyTorch](./03_ai_framework/01_pytorch/)
- [Tensorflow](./03_ai_framework/02_tensorflow/)

## 自定义点对点通信算子

- [自定义 Send/Recv 算子（基于 AICPU 通信引擎）](./04_custom_ops_p2p/)

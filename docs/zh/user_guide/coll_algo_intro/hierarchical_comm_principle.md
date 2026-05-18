# 分级通信原理

下面以通信算子ReduceScatter、AllGather、AllReduce为例介绍分级通信的流程。

## ReduceScatter

如下图所示，ReduceScatter算子要求第i个rank最终得到第i份归约结果，为了保证Server间通信数据块的连续性，首先在Server间执行ReduceScatter操作，再在Server内执行ReduceScatter操作。

**图 1**  ReduceScatter算子分级通信流程
![](figures/reduce_scatter_hierarchical_flow.png "ReduceScatter算子分级通信流程")

## AllGather

如下图所示，AllGather算子要求第i个rank的输入数据出现在结果的第i个位置上，为了保证Server间通信数据块的连续性，首先在Server内执行AllGather操作，再在Server间执行AllGather操作。

**图 2**  AllGather算子分级通信流程
![](figures/allgather_hierarchical_flow.png "AllGather算子分级通信流程")

## AllReduce

如下图所示，AllReduce算子的输出是完整的归约结果，因此虽然拆解为了ReduceScatter和AllGather两个阶段，但不需要严格遵循ReduceScatter和AllGather的语义，可以将较大数据量的通信过程放在带宽更高的Server内，即先在Server内执行ReduceScatter操作，然后在Server间执行AllReduce操作，最后在Server内执行AllGather操作。

**图 3**  AllReduce算子分级通信流程
![](figures/allreduce_hierarchical_flow.png "AllReduce算子分级通信流程")

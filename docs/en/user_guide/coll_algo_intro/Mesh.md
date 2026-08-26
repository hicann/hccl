# Mesh

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:53:26.679Z pushedAt=2026-08-04T03:42:15.387Z -->

## Algorithm Description

Mesh is a basic algorithm within the FullMesh interconnection topology. It provides full connectivity between NPUs, where any two NPUs can directly send and receive data.

![](figures/mesh.png)

The following figure shows the process of implementing the AllReduce operator using the Mesh algorithm. Each NPU concurrently uses multiple HCCS links to read data from or write data to peer NPUs, thereby utilizing the bidirectional bandwidth of the duplex interconnection links simultaneously.

![](figures/mesh_algo_principle.png)

The time complexity of the Mesh algorithm is O(1).

## Cost Calculation

**Table 1** Time consumption of each operation in the Mesh algorithm

| Operation          | Time Cost                             |
| ------------- | -------------------------------- |
| Scatter       | $\alpha+\frac{1}{p}n\beta$       |
| Gather        | $\alpha+\frac{1}{p}n\beta$       |
| Broadcast     |  Implemented as Scatter + AllGather. The time consumption is:<br> $2\alpha + \frac{2}{p}n\beta$     |
| Reduce       |  Implemented as ReduceScatter + Gather. The time consumption is:<br> $2\alpha + \frac{2}{p}n\beta + \frac{p-1}{p}n\gamma$     |
| ReduceScatter | $\alpha+\frac{1}{p}n\beta+\frac{p-1}{p}n\gamma$|
| AllGather        | $\alpha+\frac{1}{p}n\beta$       |
| AllReduce     | Implemented as ReduceScatter + AllGather. The time consumption is:<br> $2\alpha+\frac{2}{p}n\beta + \frac{p-1}{p}n\gamma$                                   |

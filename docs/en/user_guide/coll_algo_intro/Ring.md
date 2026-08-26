# Ring

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:54:40.821Z pushedAt=2026-08-04T10:24:35.878Z -->

## Algorithm Description

In the Ring algorithm, all NPUs are connected in a ring topology. Each card has a left-hand neighbor and a right-hand neighbor—one for receiving data and the other for sending data. The algorithm performs gradient accumulation in a cyclic manner and then synchronizes parameters in a second loop.

![](figures/ring.png)

The Ring algorithm is suitable for star or fat-tree topologies. Its key characteristic is that all NPU devices are connected in series through their single-port duplex links to form a ring.

The following figure illustrates the process of implementing the AllReduce operator using the Ring algorithm. In each step, the corresponding data block is sent to the downstream neighbor. After completing one full loop around the ring, the ReduceScatter phase is finished, and after another full loop, the AllGather phase is completed.

![](figures/ring_algo_principle.png)

The time complexity of the Ring algorithm is O\(n-1\), where n is the number of NPU devices on the Ring.

## Time Consumption Calculation

The overall approach is as follows: all participating nodes form a ring, and each node communicates only with its left and right neighbors. If the number of nodes is p, the number of communication rounds required is p-1, with each exchange transferring $\frac{1}{p}$ of the data.

**Table 1** Time consumption calculation for each operation in the Ring algorithm

| Operation          | Time Consumption                                                         |
| ------------- | ------------------------------------------------------------ |
| Scatter       | $(p-1)(\alpha+\frac np\beta)=(p-1)\alpha+\frac {p-1}p n\beta$  |
| Gather        | $(p-1)(\alpha+\frac np\beta)=(p-1)\alpha+\frac {p-1}p n\beta$     |
| Broadcast     | $(p-1)(\alpha+n\beta)=(p-1)\alpha+ (p-1)n\beta$    |
| Reduce     | $(p-1)(\alpha+n\beta + n\gamma)=(p-1)\alpha+ (p-1)n\beta +(p-1)n\gamma$                                        |
|  ReduceScatter |  $(p-1)(\alpha+\frac{n}{p}\beta+\frac{n}{p}\gamma)=(p-1)\alpha+\frac{p-1}{p}n\beta+\frac{p-1}{p}n\gamma$  |
|  AllGather    | $(p-1)(\alpha+\frac{n}{p}\beta)=(p-1)\alpha+\frac{p-1}{p}n\beta$  |
| AllReduce     | Implemented as ReduceScatter + AllGather: <br> $2(p-1)\alpha+2\frac{p-1}{p}n\beta+\frac{p-1}{p}n\gamma$ |

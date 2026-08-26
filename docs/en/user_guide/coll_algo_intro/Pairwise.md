# Pairwise

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:53:59.486Z pushedAt=2026-08-04T07:42:04.335Z -->

## Algorithm Description

Typically, each node has only one RDMA NIC. If the Mesh algorithm is used over RDMA links to perform AllToAll, the "many-to-many" problem arises, where each node receives data from and sends data to multiple nodes simultaneously. Multiple data streams contend for resources on the same link, which may lead to overall performance degradation.

The Pairwise algorithm is a step-by-step execution version of the Mesh algorithm. Through proper planning, it decomposes the communication into multiple steps, where each step only receives data from one node and sends data to one node. For example, for a node with rankid i, the first step receives data from node \(i-1\) and sends data to node \(i+1\); the second step receives data from node \(i-2\) and sends data to node \(i+2\); and so on.

![](figures/pairwise.png)

## Time Consumption Calculation

Define $n_{ij}$ as the amount of data that node $i$ needs to send to node $j$.

For step $k$, node $i$ sends data of size $n_{i,i+k}$ to node $i+k$, and the time cost of step $k$ is:
$\alpha + \beta \cdot \underset{i}{\max}(n_{i,i+k})$

Hence, the total time to complete the entire PairWise exchange is:

$(p-1)\alpha + \beta \cdot \underset{k}{\Sigma} \underset{i}{\max}(n_{i,i+k})$

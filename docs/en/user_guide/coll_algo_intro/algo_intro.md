# Algorithm Overview

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:53:28.557Z pushedAt=2026-08-03T09:45:57.386Z -->

For the same collective communication operator, different communication algorithms are often adopted depending on network topology, communication data volume, hardware resources, and other factors, so as to maximize cluster communication performance. HCCL provides topology algorithms such as Mesh, Ring, Recursive Halving-Doubling (RHD), Pairwise, and Pipeline for intra-server, inter-server, and inter-SuperPoD collective communication.

## Intra-Server Communication Algorithms

Within the HCCL communication domain, Mesh, Ring, Double-Ring, and Star algorithms are supported for intra-server communication. The algorithm to be used is automatically selected based on the hardware topology; no configuration is required, nor is it supported.

## Inter-Server/Inter-Supernode Communication Algorithms

HCCL supports adaptive selection of the following algorithms for inter-server/inter-SuperPoD communication in the communication domain. The adaptive algorithm selects the appropriate one based on the product form, data volume, and number of servers. By default, no user configuration is required.

- Ring algorithm: A ring-based communication algorithm with a large number of communication steps (linear complexity) and relatively high latency. However, its communication relationships are simple and it is less affected by network congestion. It is suitable for scenarios where the number of servers in the communication domain is small, the communication data volume is small, the network has significant congestion, and the Pipeline algorithm is not applicable.

- RHD (Recursive Halving-Doubling) algorithm: A recursive halving and doubling algorithm with a small number of communication steps (logarithmic complexity) and relatively low latency. However, it introduces additional communication overhead at non-power-of-two node scales. It is suitable for scenarios where the number of servers in the communication domain is a power of two and the Pipeline algorithm is not applicable, or scenarios where the number of servers is not a power of two but the communication data volume is small.

- NHR (Nonuniform Hierarchical Ring) algorithm: A nonuniform hierarchical ring algorithm with a small number of communication steps (logarithmic complexity) and relatively low latency. It is suitable for scenarios where the number of servers in the communication domain is large and the Pipeline algorithm is not applicable.

- NB (Nonuniform Bruck) algorithm: A nonuniform data block communication algorithm with few communication steps (logarithmic complexity) and relatively low latency. It is suitable for scenarios where there are many servers in the communication domain and the Pipeline algorithm is not applicable.

- Pipeline algorithm: A pipeline parallel algorithm that can concurrently use intra-server and inter-server links, or intra-SuperPoD and inter-SuperPoD links. It is suitable for scenarios with large communication data volumes and multiple cards per machine in the communication domain.

- Pairwise algorithm: A pairwise communication algorithm used only for AlltoAll, AlltoAllV, and AlltoAllVC operators. It has many communication steps (linear complexity) and relatively high latency, but can avoid the one-to-many phenomenon in the network (where one rank sends data to multiple ranks through the same port). It is suitable for scenarios with large communication data volumes where the one-to-many network phenomenon needs to be avoided.

- AHC (Asymmetric Hierarchical Concatenate) algorithm: A hierarchical collective communication algorithm used only for ReduceScatter, AllGather, and AllReduce operators. It is suitable for scenarios where NPU distribution in the communication domain has multiple levels and supports both symmetric and asymmetric NPU distribution across levels. The relative benefit is greater when bandwidth convergence exists between levels in the communication domain.

> [!NOTE]Note
>
> - To specify an inter-server or inter-SuperPoD communication algorithm, developers can set the environment variable [HCCL_ALGO](../hccl_env/HCCL_ALGO.md). Note that if an inter-server or inter-SuperPoD communication algorithm is specified through the HCCL_ALGO environment variable, the adaptive algorithm selection function will no longer take effect, and the user-specified algorithm will be used instead.
> - For the operators and products supported by each algorithm, see the environment variable [HCCL_ALGO](../hccl_env/HCCL_ALGO.md).

## Special Notes

  - Grouped Full Mesh algorithm: A grouped fully connected communication algorithm used only for the AlltoAll, AlltoAllV, and AlltoAllVC operators on Atlas A3 training products/Atlas A3 inference products. In large-scale clusters, communication is completed in multiple groups with a certain degree of concurrency. Within a SuperPoD, the concurrency is high and latency is low; between SuperPoDs, the concurrency is low and latency is relatively high (to avoid the incast phenomenon in the network). This algorithm cannot be configured through the [HCCL_ALGO](../hccl_env/HCCL_ALGO.md) environment variable.

  - NHR-HCF (NHR Highest Common Factor) algorithm: A greatest common divisor algorithm applicable only to Atlas A3 training products/Atlas A3 inference products. It takes effect by default in scenarios where the number of servers differs between SuperPoDs but the number of cards within each server is the same. This algorithm cannot be configured through the [HCCL_ALGO](../hccl_env/HCCL_ALGO.md) environment variable. The algorithm splits the communication domain into multiple symmetrically distributed logical SuperPoDs by computing the greatest common divisor of the inter-SuperPoD server counts, and selects a communication algorithm based on the new logical topology. The relative benefit is greater in scenarios where the greatest common divisor of the inter-SuperPoD server counts is greater than 1.

## Time Consumption Evaluation

HCCL uses the α–β model (Hockney) for performance evaluation. The variables used in algorithm time consumption calculation are defined as follows:

- α: Fixed latency between nodes, in seconds (s), determined by the communication hardware device and the underlying software stack.

- β: Data transmission time per byte, in seconds per byte (s/Byte), determined by the communication link capability.

- n: Size of communication data between nodes, in bytes (Byte), determined by the communication algorithm.

- γ: Reduction computation time per byte of data, in s/Byte, determined by the computing hardware device capability.

- p: Number of nodes in the communication domain, which affects the number of communication steps and is determined by the communication domain where the collective communication operator resides.

The time for a single-step transmission and reduction computation of n bytes of data is: D = α + nβ + nγ.

By leveraging network topology, collective communication algorithms optimize communication relationships and communication steps, reduce communication frequency to decrease fixed latency, and reduce the actual communication data volume to decrease transmission time and computation time, thereby achieving the goal of optimizing collective communication performance.

## Hierarchical Communication Principle

HCCL typically divides the topology into two levels (intra-node/inter-node) or three levels (intra-node/inter-node/inter-SuperPoD), and executes collective communication hierarchically, with different bandwidths across different levels of links. Hierarchical communication enables communication task orchestration to be aligned with the network topology, thereby maximizing link utilization.

Taking the single-operator mode and intra-node/inter-node two-level topology of Atlas A2 training products/Atlas A2 inference products as an example, the specific hierarchical communication process of each collective communication operator is shown in the following table:

| Collective Communication Operator | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- |
| ReduceScatter | Inter-server ReduceScatter | Intra-server ReduceScatter | / |
| ReduceScatterV | Inter-server ReduceScatterV | Intra-server ReduceScatterV | / |
| AllGather | Intra-server AllGather | Inter-server AllGather | / |
| AllGatherV | Intra-server AllGatherV | Inter-server AllGatherV | / |
| AllReduce | Intra-server ReduceScatter | Inter-server AllReduce | Intra-server AllGather |
| Scatter | Inter-server Scatter | Intra-server Scatter | / |
| Broadcast | Intra-server Scatter | Inter-server Broadcast | Intra-server AllGather |
| Reduce | Intra-server ReduceScatter | Inter-server Reduce | Intra-server Gather<br>HCCL does not provide the Gather operator. The difference between the Gather operation and the AllGather operation is that only the result is sent to the output buffer of the root node. |
| AlltoAll | Intra-server AlltoAll | Inter-server AlltoAll | / |
| AlltoAllV | Intra-server AlltoAllV | Inter-server AlltoAllV | / |
| AlltoAllVC | Intra-server AlltoAllVC | Inter-server AlltoAllVC | / |

For detailed hierarchical communication process examples, see [Hierarchical Communication Principle](hierarchical_comm_principle.md).

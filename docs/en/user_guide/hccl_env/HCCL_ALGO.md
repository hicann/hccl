# HCCL_ALGO

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:54:22.355Z pushedAt=2026-08-11T03:21:04.686Z -->

## Function

This environment variable is used to configure the inter-server communication algorithm and inter-SuperPoD communication algorithm for collective communication. It supports two configurations: globally configuring the algorithm type and configuring the algorithm type by operator.

> [!NOTE] Note
>
>- HCCL offers adaptive algorithm selection. By default, it selects an appropriate algorithm based on the product form, data volume, and number of servers. In most cases, you do not need to manually specify an algorithm. If you specify an inter-server or inter-SuperPoD communication algorithm using this environment variable, the adaptive algorithm selection feature will no longer take effect.
>- For certain communication operators, when a specific type of AI processor is used and the data volume is small, the communication algorithm is adaptively selected by HCCL and is not controlled by this environment variable.
>- The algorithms listed in this section are all communication algorithms supported by HCCL. For the inter-server communication algorithms and inter-SuperPoD communication algorithms supported by different products, see [Inter-Server Communication Algorithm Support List](inter_server_algo_support.md) and [Inter-SuperPoD Communication Algorithm Support List](inter_superpod_algo_support.md).

- **Globally configuring the algorithm type:**

  ```bash
  export HCCL_ALGO="level0:NA;level1:<algo>;level2:<algo>"
  ```

  - `level0` represents the intra-server communication algorithm. Currently, only `NA` is allowed.

  - `level1` represents the inter-server communication algorithm, which supports the following values:

    - ring: A ring-based communication algorithm with many communication steps (linear complexity) and relatively high latency, but simple communication relationships and less impact from network congestion. It is suitable for use cases where the number of servers within the communicator is small, the communication data volume is small, the network has significant congestion, and the pipeline algorithm is not applicable.

    - H-D_R: Recursive Halving-Doubling (RHD) algorithm with few communication steps (logarithmic complexity) and relatively low latency, but it introduces additional communication overhead when the number of nodes is not an integer power of 2. It is suitable for use cases where the number of servers within the communicator is an integer power of 2 and the pipeline algorithm is not applicable, or where the number of servers is not an integer power of 2 but the communication data volume is small.

    - NHR: Nonuniform Hierarchical Ring algorithm with few communication steps (logarithmic complexity) and relatively low latency. It is suitable for use cases where the number of servers within the communicator is large and the pipeline algorithm is not applicable.

      **In the current version, Ascend 950PR and Ascend 950DT support only the NHR algorithm.**

    - NHR_V1: Corresponds to the NHR algorithm of earlier versions. It has few communication steps (root complexity) and relatively low latency. It is suitable for use cases where the number of servers in the communicator is not an integer power of 2 and the pipeline algorithm is not applicable. The theoretical performance of the NHR_V1 algorithm is lower than that of the new NHR algorithm. This config will be phased out in the future. You are advised to use the NHR algorithm.

    - NB: Nonuniform Bruck algorithm. It has few communication steps (logarithmic complexity) and relatively low latency. It is suitable for use cases where the number of servers in the communicator is large and the pipeline algorithm is not applicable.

    - AHC: Asymmetric Hierarchical Concatenate algorithm. It is suitable for use cases where NPU distribution within the communicator has multiple levels, with symmetric or asymmetric NPU distribution across levels (i.e., asymmetric device count). The benefit is more tangible when bandwidth convergence exists between levels within the communicator.

      Note: When level1 (inter-server communication algorithm) is set to `AHC`, level2 (inter-SuperPoD communication algorithm) automatically adopts the AHC algorithm without additional configuration. Even if other algorithms are set for level2, these settings will not take effect.

    - pipeline: Pipeline parallel algorithm. It can concurrently use intra-server and inter-server links. It is suitable for use cases with large communication data volumes where each machine in the communicator contains multiple devices.

    - pairwise: A pairwise communication algorithm, used only for AlltoAll, AlltoAllV, and AlltoAllVC operators. It involves a large number of communication steps (linear complexity) and relatively high latency, and requires additional memory allocation with the memory size proportional to the data volume. However, it is suitable for use cases with large communication data volumes where one-to-many network patterns need to be avoided.

    When level1 is not set:

    - For Ascend 950PR/Ascend 950DT, the NHR algorithm is used by default.

    - For Atlas A3 training products/Atlas A3 inference products, the algorithm is automatically selected internally based on the product form, number of nodes, and data volume.

    - For Atlas A2 training products/Atlas A2 inference products, the algorithm is automatically selected internally based on the product form, number of nodes, and data volume.

    - For Atlas training products, when the number of servers within the communicator is not an integer power of 2, the ring algorithm is used by default; in other use cases, the H-D_R algorithm is used by default.

  - level2 represents the inter-SuperPoD communication algorithm, which supports the following values:

    - ring: A ring-structure-based communication algorithm with many communication steps (linear complexity) and relatively high latency, but simple communication relationships and less susceptibility to network congestion. Suitable for use cases where there are few SuperPoDs within the communicator and the number is not an integer power of 2.

    - H-D_R: Recursive Halving-Doubling (RHD) algorithm with few communication steps (logarithmic complexity) and relatively low latency, but introduces additional communication at non-integer-power-of-2 node scales. Suitable for use cases where the number of SuperPoDs within the communicator is an integer power of 2, or where the number of SuperPoDs is not an integer power of 2 but the communication data volume is small.

    - NHR: Nonuniform Hierarchical Ring algorithm with few communication steps (logarithmic complexity) and relatively low latency. Suitable for use cases where there are many SuperPoDs within the communicator.

    - NB: Nonuniform Bruck algorithm, featuring fewer communication steps (logarithmic complexity) and relatively low latency. Suitable for use cases with a large number of SuperPoDs within the communicator.

    - pipeline: Pipeline parallel algorithm that can concurrently use intra-SuperPoD and inter-SuperPoD links. Suitable for use cases with large communication data volume where each SuperPoD within the communicator contains multiple devices.

        For details about the communication operators, data types, and network operation modes supported by each inter-SuperPoD communication algorithm, see [Inter-SuperPoD Communication Algorithm Support List](inter_superpod_algo_support.md).

        When level2 is not set, the ring algorithm is used if the number of SuperPoDs within the communicator is less than 8 and is not an integer power of 2; otherwise, the H-D_R algorithm is used.

    The level2 settings currently apply only to:

    - Atlas A3 training products and Atlas A3 inference products.

    - When the communication operator expansion mode is AI_CPU. This mode can be set through the environment variable [HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md).

- **Configuring communication algorithms by operator type:**

    ```bash
    export HCCL_ALGO="<op0>=level0:NA;level1:<algo0>;level2:<algo1>/<op1>=level0:NA;level1:<algo3>;level2:<algo4>"
    ```

    Where:

  - <op\> specifies the communication operator type. The following settings are allowed:

    - allgather: corresponds to the communication operators AllGather and AllGatherV.

    - reducescatter: corresponds to the communication operators ReduceScatter and ReduceScatterV.

    - allreduce: corresponds to the communication operator AllReduce.

    - broadcast: corresponds to the communication operator Broadcast.

    - reduce: corresponds to the communication operator Reduce.

    - scatter: corresponds to the communication operator Scatter.

    - alltoall: Corresponds to the communication operators AlltoAll, AlltoAllV, and AlltoAllVC.

  - <algo\> specifies the communication algorithm used by the communication operator. The supported configurations are the same as the level1 value and level2 values in the global configuration method. Ensure that the specified communication algorithm is of a type supported by the communication operator. For the algorithms supported by each operator, see [Inter-Server Communication Algorithm Support List](inter_server_algo_support.md) and [Inter-SuperPoD Communication Algorithm Support List](inter_superpod_algo_support.md). Communication operators for which no algorithm is specified will automatically select a communication algorithm based on the product form, number of nodes, and data volume.

  - Use `/` to separate configurations of multiple operators.

## Configuration Example

- Globally configuring algorithm type

    ```bash
    export HCCL_ALGO="level0:NA;level1:NHR"
    ```

- Configuring algorithm type by operator

    ```bash
    # The AllReduce operator uses the Ring algorithm, the AllGather operator uses the RHD algorithm, and other operators automatically select a communication algorithm based on the product type, number of nodes, and data volume.
    export HCCL_ALGO="allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:H-D_R"
    ```

## Constraints

- In the current version, the intra-server communication algorithm can only be configured as `NA`.

- For Atlas A2 training products and Atlas A2 inference products, configuring the HCCL_ALGO environment variable is not recommended in order-preserving scenarios that require strict deterministic computation.

- If you call the HCCL C API to initialize a communicator with specific configurations and specify a communication algorithm through the `hcclAlgo` parameter of `HcclCommConfig`, the communicator-level configurations take precedence.

## Applicable Products

Ascend 950PR/Ascend 950DT

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products

<!-- npu="910" id1 -->

Atlas training products

<!-- end id1 -->
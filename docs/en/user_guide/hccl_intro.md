# Introduction to HCCL

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T04:00:29.786Z pushedAt=2026-08-13T10:21:59.998Z -->

HCCL (Huawei Collective Communication Library) is a high-performance collective communication library based on Ascend hardware, providing high-performance and high-reliability communication solutions for computing clusters.

## Core Features

- Provides high-performance collective communication and point-to-point communication in single-node and multi-node environments.

- Supports collective communication primitives such as AllReduce, Broadcast, AllGather, ReduceScatter, AlltoAll, Send, and Receive.

- Supports communication algorithms such as Ring, Mesh, and Recursive Halving-Doubling (RHD).

- Supports high-speed communication links such as HCCS, RoCE, PCIe, and UB (Unified Bus).

- Supports two execution modes: single-operator mode and graph mode.

- Supports custom development of communication operators.

## Software Architecture

HCCL is a core component of CANN, providing high-performance and high-reliability communication solutions for NPU clusters. HCCL supports multiple AI frameworks on the upper layer and enables efficient interconnection among various Ascend AI processors on the lower layer. Its architecture is shown in the following figure.

**Figure 1**  Collective communication library software architecture
![Collective communication library software architecture](figures/hccl_architecture.png)

HCCL consists of the HCCL collective communication library and the HCOMM (Huawei Communication) communication basics library:

- **HCCL collective communication library**: Includes built-in communication operators and extended communication operators, and provides external communication operator interfaces.

  - Built-in communication operators: Basic communication operators provided by HCCL, including collective communication operators and point-to-point communication operators.

  - Extended communication operators: Users can customize extended communication operators using the interfaces provided by the HCOMM communication basics library.

- **HCOMM communication basics library**: Adopts a layered decoupling design approach, dividing communication capabilities into a control plane and a data plane.

  - Control plane: provides topology information query and communication resource management functions.

  - Data plane: provides data movement and computation functions such as local operations, inter-operator synchronization, and communication operations.

The control plane provides communication resources, while the data plane provides operation resources. The interfaces provided by this approach allow communication operator developers to focus on service innovation without concerning themselves with the complex implementation details at the underlying chip level.

## Supported Products

<!-- npu="950" id1 -->

- Ascend 950PR/Ascend 950DT

<!-- end id1 -->
<!-- npu="A3" id2 -->

- Atlas A3 training products/Atlas A3 inference products

<!-- end id2 -->
<!-- npu="910b" id3 -->

- Atlas A2 training products/Atlas A2 inference products

<!-- end id3 -->
<!-- npu="910" id4 -->

- Atlas training products

<!-- end id4 -->
<!-- npu="310p" id5 -->

- Atlas inference products

<!-- end id5 -->

<!-- npu="910b,310p" id8 -->

> [!NOTE]Note
> <!-- npu="910b" id6 -->
> - For Atlas A2 training products/Atlas A2 inference products, only Atlas 800T A2 training server, Atlas 900 A2 PoD cluster base unit, and Atlas 200T A2 Box16 heterogeneous subrack are supported.
> <!-- end id6 -->
> <!-- npu="310p" id7 -->
> - For Atlas inference products, only Atlas 300I Duo inference card is supported.

<!-- end id7 -->
<!-- end id8 -->

# Related Concepts

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:59:45.169Z pushedAt=2026-08-17T02:51:48.847Z -->

Before using this document, it is recommended that readers familiarize themselves with HCCL-related concepts.

## HCCL Basic Concepts

A typical HCCL communication network is shown in the following figure.

![Typical Communication Network Example](figures/typical_network.png)

The preceding figure involves the following basic concepts:

- **AI server**: Also known as a compute node, it is a collective term for a server form factor typically consisting of 8-card or 16-card Ascend NPU devices.

- **AI cluster**: A system in which multiple AI servers are interconnected through switching devices for distributed training or inference.

    If AI servers are connected through Lingqu Bus switching devices, the resulting network is referred to as **super-node networking**.

- **Communication member**: Commonly referred to as a rank, which is the smallest logical entity participating in communication. Each rank is assigned a unique identifier.

- **Communicator**: A grouping of communication members that defines the communication scope. A computing task can create multiple communicators, and a communication member can join multiple communicators.

- **Communication operator**: An operator that performs communication tasks within a communicator. Collective communication refers to communication operations in which all members participate together, such as Broadcast and AllReduce.

- **Communication algorithm**: For different scenarios such as network topologies, data volumes, and hardware resources, communication operators typically use different communication algorithms for implementation.

## Glossary

| Name | Description |
| --- | --- |
| NPU | Neural Network Processing Unit.<br>Adopts a "data-driven parallel computing" architecture, excels at processing massive amounts of video and image multimedia service data, and is specifically designed to handle large-scale computing tasks in AI apps. |
| HCCL | Huawei Collective Communication Library.<br>Provides data parallelism and model parallelism collective communication solutions for single-node multi-card and multi-node multi-card scenarios. |
| HCOMM | Huawei Communication, the basic communication library from Huawei. |
| HCCS | Huawei Cache Coherence System.<br>Used for high-speed interconnection between CPUs and NPUs. |
| HCCP | Huawei Collective Communication Adaptive Protocol.<br>Provides cross-NPU device communication capabilities, shielding upper layers from specific communication protocol differences. |
| TOPO | Topology.<br>A network configuration or arrangement formed by device connections within a LAN or across multiple LANs. |
| PCIe | Peripheral Component Interconnect Express, a serial peripheral expansion bus standard commonly used for peripheral expansion in computer systems. |
| PCIe-SW | PCIe Switch, a switching device compliant with PCIe bus expansion. |
| QP | Queue Pair.<br>QP is the core communication unit of Remote Direct Memory Access (RDMA) technology, consisting of a Send Queue (SQ) and a Receive Queue (RQ), used to manage data transfer tasks. |
| SDMA | System Direct Memory Access, abbreviated as DMA, allows peripheral devices to directly access system memory without CPU intervention. |
| RDMA | Remote Direct Memory Access, a technology that enables direct data transfer from the memory of one machine to another without involving the operating systems of either machine, generally referring to a memory access method that can cross networks. |
| RoCE | RDMA over Converged Ethernet, an RDMA technology carried over converged Ethernet, i.e., an RDMA communication method over Ethernet. |
| AIV | Vector Core in the AI Core. |
| TS | Task Scheduler. |
| CCU | Collective Communication Unit, a collective communication acceleration unit. |

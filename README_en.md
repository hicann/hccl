# HCCL

## Latest News

- [2026/05] Ascend950 communication operators support static libraries. For details, see [Issue#90](https://gitcode.com/cann/hccl/issues/90) and [Issue#105](https://gitcode.com/cann/hccl/issues/105).
- [2026/04] Ascend950 communication operators support graph mode. For details, see [PR#613](https://gitcode.com/cann/hccl/pull/613), [PR#340](https://gitcode.com/cann/hccl/pull/340), [PR#296](https://gitcode.com/cann/hccl/pull/296), and [GE PR#1620](https://gitcode.com/cann/ge/pull/1620).
- [2026/03] Ascend950 communication operators support AIV and AICPU communication engines (single operator).
- [2025/11/30] The HCCL project is officially open-sourced.

## Overview

The Huawei Collective Communication Library (HCCL) is a high-performance collective communication library based on the Ascend AI processor. It provides high-performance and reliable communication solutions for computing clusters with the following core features:

- Provides high-performance collective communication and point-to-point communication in single-node and multi-node environments.
- Supports collective communication primitives such as AllReduce, Broadcast, AllGather, ReduceScatter, and AlltoAll.
- Supports communication algorithms such as Ring, Mesh, and Recursive Halving-Doubling (RHD).
- Supports high-speed communication links such as HCCS, RoCE, and PCIe.
- Supports both single-operator and graph execution modes.

HCCL is a core component of CANN. It supports multiple AI frameworks upward and enables communication capabilities among multiple Ascend AI processors downward. The following figure shows the software architecture:

<img src="./docs/en/build/figures/architecture.png" alt="hccl-architecture" style="width: 65%; height:65%;" />

HCCL consists of the HCCL collective communication library and the HCOMM (Huawei Communication) basic communication library:

- HCCL: includes built-in and extended communication operators, and provides external communication operator APIs.
- [HCOMM](https://gitcode.com/cann/hcomm): adopts a layered and decoupled design, dividing communication capabilities into a control plane​ and a data plane.

## Directory Structure

The key directories of this project are as follows:

```text
│── src                         # HCCL operator source code directory
|    ├── common                 # Common logic, including type definitions, logging modules, and so on
|    └── ops                    # HCCL operator implementation
|        ├── all_gather         # AllGather operator implementation
|        ├── all_gather_v       # AllGatherV operator implementation
|        ├── all_reduce         # AllReduce operator implementation
|        ├── all_to_all_v       # AlltoAll, AlltoAllV, and AlltoAllVC operator implementation
|        ├── barrier            # Barrier operator implementation
|        ├── batch_send_recv    # BatchSendRecv operator implementation
|        ├── broadcast          # Broadcast operator implementation
|        ├── interface_graph_mode # Graph mode interface implementation
|        ├── op_common          # Common operator components
|        │   ├── executor       # Executor
|        │   ├── selector       # Algorithm selector
|        │   ├── template       # Algorithm template
|        │   └── topo           # Communication domain topology information acquisition and conversion
|        ├── recv               # Recv operator implementation
|        ├── reduce             # Reduce operator implementation
|        ├── reduce_scatter     # ReduceScatter operator implementation
|        ├── reduce_scatter_v   # ReduceScatterV operator implementation
|        ├── scatter            # Scatter operator implementation
|        └── send               # Send operator implementation
├── include                     # HCCL external header files
├── test                        # Test code directory
|   ├── ut                      # Unit test code directory
|   └── st                      # System test code directory
├── docs                        # Documentation directory
├── examples                    # Sample code directory
└── build.sh                    # Compilation and build script
```

## Version Matching

The source code of this project is released along with the CANN software version. For the mapping between CANN software versions and project tags, refer to the corresponding version descriptions in the [Release Repository](https://gitcode.com/cann/release-management).

To ensure smooth source code customization, select a compatible CANN version that matches the GitCode tag source code. Using the master branch may cause version mismatch risks.

## Quick Start

To quickly build and experience this project, refer to the following guides:

- [Source Code Build](./docs/en/build/build.md): Learn how to compile, install, and perform basic tests for this project.
- [Sample Execution](./examples/README_en.md): Follow detailed sample code and step-by-step instructions for a quick trial.

## Learning Tutorials

HCCL provides user guides, communication operator development guides, technical articles, and training videos. For details, see [HCCL references](./docs/README_en.md).

## Related Information

- [Contribution Guide](CONTRIBUTING_en.md)
- [Security Statement](SECURITY_en.md)
- [License](LICENSE)

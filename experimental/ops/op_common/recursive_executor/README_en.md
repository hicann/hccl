# Recursive Executor

English | [简体中文](./README.md)

> An experimental recursive executor that replaces hard-coded dispatch chains with an algorithm description tree + a generic interpreter, providing a unified, composable, and recursive execution framework for HCCL collective communication operators.

This module is the reference implementation of the HCCL architecture refactoring RFC ([issue #2](https://gitcode.com/luyang20/hccl/issues/2)). It resides under `experimental/` and is not included in production builds. Compatibility is not guaranteed.

---

## Table of Contents

- [Motivation](#motivation)
- [Design](#design)
- [Usage](#usage)
- [Status](#status)
- [Limitations](#limitations)

---

## Motivation

### Problems with the Current Architecture

The existing HCCL operator implementation has the following structural issues:

1. **Hard-coded dispatch logic**: The multi-level communication flow for each operator (e.g., AllGather's server→super-pod→cross-super-pod) is hard-coded as sequential code inside the operator. It cannot be reused, and adding a new operator requires rewriting the entire chain.
2. **Algorithm-execution coupling**: Algorithm selection, data partitioning, and transfer execution are mixed within the same function, making them difficult to test or replace independently.
3. **High cost of new operators**: Adding a new operator requires understanding the full-chain code and copying large amounts of boilerplate logic, creating a heavy maintenance burden.
4. **Fragmented multi-level topology support**: Communication planners for different levels (Mesh/NHR/OCS) are scattered across multiple places, lacking unified orchestration.
5. **Difficult pipeline overlap**: Overlap between data transfer and computation requires hand-written complex state machines at the operator level, which is hard to generalize.
6. **Poor testability**: Dispatch logic is embedded inside operators, making it impossible to unit-test individual communication levels.

### Driving Factor: Four-Level Topology

Ascend large-scale training clusters have a naturally four-level network topology:

| Level  | Topology Type      | Description                          |
| ------ | ------------------ | ------------------------------------ |
| layer0 | server Mesh        | Intra-server Mesh interconnect        |
| layer1 | cross-server NHR   | Inter-server Non-Uniform Hierarchical Ring |
| layer2 | cross-super-pod NHR | Inter-super-pod NHR                 |
| layer3 | cross-super-pod OCS | OCS dynamic links                    |

The current architecture writes independent dispatch code for each topology level, preventing composition. The recursive executor uses a **unified algorithm tree** to describe multi-level topology flows, encapsulating each level's communication as a Template leaf node and automatically combining them through recursive orchestration.

### Goals

- **Unified dispatch**: Express arbitrary multi-level communication flows with a single algorithm description tree
- **Composable**: Algorithm nodes can nest sub-algorithms, enabling hierarchical recursion
- **Extensible**: New operators only need to assemble an algorithm tree + implement Templates, without modifying the executor
- **Testable**: Each Template/CommPlanner can be tested independently
- **Pipeline-friendly**: Execution strategies (SEQUENCE/PARALLEL/OMNIPIPE) are part of the algorithm description, handled uniformly by the executor

---

## Design

### Overall Architecture

```text
┌─────────────────────────────────────────────────────┐
│                  AlgSelector (Registry)              │
│  Operator → HcclAlgorithm (static algorithm description:  │
│             algorithm tree + parameter mapping)      │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│                  OpsExecutor (Generic Interpreter)   │
│  Recursively traverse algorithm tree → orchestrate  │
│  by strategy → invoke Template execution             │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              Template (Single-Level Execution Unit)  │
│  PreCopy → RunAlgorithm(CommPlanner) → SendAll → PostCopy │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              CommPlanner (Communication Plan Generator)│
│  Compute peers, data slices, transfer direction     │
│  → invoke HCOMM transfer                             │
└─────────────────────────────────────────────────────┘
```

Four-layer separation of concerns:

| Layer | Component               | Responsibility                                          |
| ----- | ----------------------- | ------------------------------------------------------- |
| L1    | AlgSelector / HcclAlgorithm  | Static algorithm description: execution strategy tree + data parameter mapping |
| L2    | OpsExecutor              | Generic recursive interpreter: traverses algorithm tree, orchestrates child nodes by strategy |
| L3    | Template                | Single-level execution framework: PreCopy → CommPlanner → SendAll → PostCopy |
| L4    | CommPlanner               | Communication plan: compute peers, data slices, transfer direction |

### Core Data Structures

#### AlgoExecDesc — Recursive Algorithm Tree Node

```cpp
// inc/algo_desc.h

enum HcclAlgExecPolicy {
    SEQUENCE,   // Children execute sequentially, prior output = next input
    PARALLEL,   // Children execute in parallel, data partitioned among them
    OMNIPIPE,   // Pipeline overlap: receive, compute, and send concurrently
};

// Leaf node: points to a Template execution description
struct TemplateExecDesc {
    TemplateDesc templateDesc;  // Template identifier + parameters
};

// Recursive node: itself is an algorithm tree
struct AlgoExecDesc {
    HcclAlgExecPolicy policy;
    std::vector<VariantType<TemplateExecDesc, std::shared_ptr<AlgoExecDesc>>> children;
    // ranksForInputData / ranksForOutputData connect data ownership across stages
};
```

`VariantType` is a variant of `TemplateExecDesc` (leaf) or `std::shared_ptr<AlgoExecDesc>` (subtree), supporting arbitrary-depth recursive nesting.

#### DataParams — Unified Data Parameters

```cpp
// inc/data_types.h

struct DataParams {
    void* inputData;       // Input data pointer
    void* outputData;      // Output data pointer
    void* cclBuffer;       // CCL intermediate buffer
    DataType dataType;     // Data type
    uint64_t count;        // Total element count
    // ... offsets, strides, rank lists, etc.
};
```

Unified memory model: `Input → CCL Buffer → ... → CCL Buffer → Output`

Each Template reads input from `DataParams`, writes output to `CCL Buffer`, and the next Template reads input from the same `CCL Buffer`, forming a pipeline.

### Execution Strategies

| Strategy       | Semantics               | Data Flow                                                          |
| -------------- | ----------------------- | ------------------------------------------------------------------ |
| **SEQUENCE**  | Children execute sequentially | Prior child's `ranksForOutputData` = next child's `ranksForInputData` |
| **PARALLEL**  | Children execute in parallel | Data is partitioned to each child for independent processing       |
| **OMNIPIPE**  | Pipeline overlap         | Receive, compute, and send concurrently, minimizing bubble overhead |

`ranksForInputData` / `ranksForOutputData` are data ownership contracts connecting stages, ensuring data consistency across SEQUENCE stages.

### Algorithm Assembly Example

Using AllGather with four-level topology as an example, assembled as a SEQUENCE tree:

```text
AlgoExecDesc(policy=SEQUENCE)
├── TemplateExecDesc(Mesh)       // layer0: intra-server Mesh AllGather
├── AlgoExecDesc(policy=SEQUENCE)
│   └── TemplateExecDesc(NHR)    // layer1: cross-server NHR AllGather
├── AlgoExecDesc(policy=SEQUENCE)
│   └── TemplateExecDesc(NHR)    // layer2: cross-super-pod NHR AllGather
└── TemplateExecDesc(Mesh)       // layer3: cross-super-pod Mesh finalization
```

### OpsExecutor Recursive Orchestration

```cpp
// executor/ops_executor.cc

void OpsExecutor::OrchestrateLoop(const AlgoExecDesc& hcclAlgorithm, const DataParams& params) {
    switch (hcclAlgorithm.policy) {
        case SEQUENCE:
            // Execute children sequentially, passing ranksForOutputData → ranksForInputData
            for (auto& child : hcclAlgorithm.children) {
                if (child.isTemplate()) {
                    ExecuteTemplate(child.asTemplate(), params);
                } else {
                    OrchestrateLoop(child.asHcclAlgorithm(), params);  // Recurse
                }
            }
            break;
        case PARALLEL:
            // Partition data and execute in parallel
            break;
        case OMNIPIPE:
            // Pipeline overlap orchestration
            break;
    }
}
```

### Template / CommPlanner Separation

**Template** is responsible for the execution framework, with a fixed flow:

```text
PreCopy → RunAlgorithm(CommPlanner) → SendAll → PostCopy
```

- **PreCopy**: Move input data to CCL Buffer
- **RunAlgorithm**: Invoke CommPlanner to execute actual communication
- **SendAll**: Distribute CCL Buffer data to each rank
- **PostCopy**: Move CCL Buffer data to output

**CommPlanner** is responsible for communication plan generation: computing communication peers, data slices, and transfer direction, then invoking HCOMM transfer interfaces.

This separation ensures that Template only concerns itself with the execution framework, while CommPlanner only concerns communication details. Both can be developed and tested independently.

### Four-Level Topology Matching

`TopoMatchFourLevel` (`topo/topo_match_four_level.h`) implements four-level symmetric topology matching:

- Detects the current rank's position in the four-level topology
- Selects the corresponding Template for each level (Mesh / NHR / OCS)
- The current implementation requires symmetric topology (consistent view across all ranks)

---

## Usage

### Build

This module is compiled as a `RecursiveExecutor` OBJECT library into the main HCCL build, controlled by the top-level CMakeLists:

```bash
# Building the main repo compiles this module
bash build.sh --pkg
```

CMake configuration (`CMakeLists.txt`) key points:

- Source file list `RE_CORE_SRC` contains 14 `.cc` files
- Linked as an OBJECT library into `libhccl.so`
- Defines compilation macro `ENABLE_EXPERIMENTAL`
- Include paths: `src/`, `inc/`, CANN installation directory
- Reads CANN version from `cann_version.h`

### Runtime Switch

This feature is gated by the runtime switch `HCCL_EXPERIMENTAL_RECURSIVE_EXECUTOR=true` (see [experimental/README_en.md](../../../README_en.md#5-runtime-switch)).

The switch function `IsRecursiveExecutorEnabled()` is defined in `executor/adaptor_executor.cc`. The switch is applied at registration time rather than call time: the `REGISTER_ALG` macro checks this switch during static initialization. When the switch is off, neither the algorithm nor the executor is registered, so the selector cannot select it and the execution path is never entered.

- The compile-time constant defaults to `false` (disabled). To enable at runtime, change `constexpr bool recursiveExecutorEnabled` in `IsRecursiveExecutorEnabled()` to `true` and rebuild.
- Runtime activation: set the environment variable `HCCL_EXPERIMENTAL_RECURSIVE_EXECUTOR=true`.

### Registering a New Algorithm

Register algorithms to the `AlgSelector` singleton via the `REGISTER_ALG` macro:

```cpp
// algorithm/all_gather.cc — existing registration example

static void RegisterAllGather() {
    auto& selector = AlgSelector::Instance();
    selector.Register("AllGather", BuildAllGatherHcclAlgorithm());
}

REGISTER_ALG("AllGather", RegisterAllGather);
```

`BuildAllGatherHcclAlgorithm()` assembles the algorithm tree (`AlgoExecDesc`), returning an `HcclAlgorithm` containing the algorithm tree and parameter mapping functions.

### Adding a New Template

1. Create a new Template class under `template/<engine>/`, inheriting from `BaseTemplate`
2. Implement `PreCopy`, `RunAlgorithm`, `SendAll`, `PostCopy` methods
3. Register in `template_factory.h`'s `GetTemplate()`

```cpp
// template_factory.h

std::shared_ptr<BaseTemplate> GetTemplate(const TemplateDesc& desc) {
    switch (desc.type) {
        case TemplateType::ALLGATHER_MESH:
            return std::make_shared<AllGatherMeshTemplate>(desc);
        case TemplateType::ALLGATHER_NHR:
            return std::make_shared<AllGatherNhrTemplate>(desc);
        // Register new Templates here
    }
}
```

### Adding a New CommPlanner

1. Create a new CommPlanner implementation file under `template/comm_planners/`
2. Implement `RunMeshXxx()` / `RunNhrXxx()` etc. functions to compute peers + data slices + transfer direction
3. Call from the corresponding Template's `RunAlgorithm()`

Existing CommPlanners:

| CommPlanner          | File                              | Description                                              |
| ------------------ | --------------------------------- | -------------------------------------------------------- |
| `RunMeshAllGather` | `comm_planners/mesh_comm_planner.cc`  | Mesh topology AllGather                                  |
| `RunNhrAllGather`  | `comm_planners/nhr_comm_planner.cc`   | NHR topology AllGather (halving algorithm + last-step direct-write optimization) |

### Adding a New Operator

1. Create a new `<op_name>.cc` under `algorithm/`
2. Assemble the algorithm tree `AlgoExecDesc`, setting execution strategy and data parameter mapping
3. Register with `REGISTER_ALG`
4. (Optional) Add new Template / CommPlanner

---

## Status

### Implementation Progress: Phase 1 / 3

The RFC is planned in three phases. Currently in Phase 1:

| Phase   | Goal                                                | Status     |
| -------- | --------------------------------------------------- | ---------- |
| Phase 1  | Core framework + AllGather + AICPU engine + four-level symmetric topology | 🔨 Skeleton phase |
| Phase 2  | Multi-operator coverage + PARALLEL strategy + multi-engine (AIV/CCU) | ⏳ Planned  |
| Phase 3  | OMNIPIPE pipeline + asymmetric topology + production integration | ⏳ Planned  |

### Implemented (Skeleton)

- **Core framework**: `AlgoExecDesc` recursive algorithm tree, `OpsExecutor` generic recursive interpreter, `DataParams` unified data model
- **Algorithm registration**: `AlgSelector` singleton + `REGISTER_ALG` macro (macro defined, no instantiation calls yet)
- **Template implementations**:
  - `AllGatherMeshTemplate` (`template/aicpu/allgather_mesh_template.cc`) — Mesh AllGather, supports DirectToOutput mode
  - `AllGatherNhrTemplate` (`template/aicpu/allgather_nhr_template.cc`) — NHR AllGather, parallel PostCopy DMA optimization
- **CommPlanner implementations**:
  - `RunMeshAllGather` (`template/comm_planners/mesh_comm_planner.cc`)
  - `RunNhrAllGather` (`template/comm_planners/nhr_comm_planner.cc`) — Recursive halving algorithm, `CanReadLastStepToOutput` last-step direct write
- **Topology matching**: `TopoMatchFourLevel` (`topo/topo_match_four_level.cc`) — four-level symmetric topology
- **Executor adapter**: `AdaptorExecutor` (`executor/adaptor_executor.cc`) — bridges HCCL framework, `REGISTER_ALG` macro integration
- **OmniPipe utilities**: `OmniPipeXYdata` data structure defined (`executor/omnipipe_utils.h`), but not yet integrated into the executor main flow

### Not Implemented / Planned

- **OMNIPIPE strategy**: Data structures defined, orchestration logic not implemented
- **PARALLEL strategy**: Interface defined, data partitioning logic not implemented
- **Multi-engine**: Only AICPU; AIV (AI Core Vector) / CCU not implemented
- **Multi-operator**: Only AllGather; AllReduce / Broadcast / ReduceScatter / AlltoAll etc. not implemented
- **Asymmetric topology**: Only symmetric four-level topology supported
- **OCS CommPlanner**: layer3 OCS communication primitive not implemented

---

## Limitations

1. **Experimental, not in production builds**: This module resides under `experimental/`, does not participate in production builds by default, and does not guarantee compatibility. APIs may change at any time.
2. **Single engine**: Only AICPU engine Templates are implemented; AIV / CCU engines are not supported.
3. **Single operator**: Only AllGather is registered; AllReduce / Broadcast / ReduceScatter / AlltoAll / Send / Recv etc. are not supported.
4. **Single execution strategy**: SEQUENCE strategy is implemented; PARALLEL and OMNIPIPE have only data structure definitions, orchestration logic is incomplete.
5. **Symmetric topology requirement**: `TopoMatchFourLevel` requires symmetric four-level topology (consistent view across all ranks); asymmetric topology is not supported.
6. **Limited test coverage**: The framework has structure, but UT / ST coverage is incomplete; tests should be supplemented before production use.

---

## References

- experimental/ conventions: [experimental/README_en.md](../../../README_en.md)
- HCCL architecture brief: [docs/zh/architecture/architecture-brief.md](../../../../docs/zh/architecture/architecture-brief.md)

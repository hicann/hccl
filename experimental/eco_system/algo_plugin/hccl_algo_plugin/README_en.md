# HCCL-ALGO-Plugin Implementation

This directory contains the implementation of the [`HCCL-ALGO-Plugin RFC`](../../../../docs/en/rfcs/0002-HCCL-ALGO-Plugin.md). Based on the `HCCL-ALGO-Plugin-master` repository, it adds a custom collective communication algorithm plugin capability: users can integrate custom algorithm implementations for specified operators as independent dynamic libraries without modifying or recompiling the HCCL main library source code, and can dynamically decide whether a custom algorithm matches according to runtime parameters such as data size and topology.

When the related environment variables are not set, HCCL behavior remains completely unchanged.

## Directory Structure

```
├── src/                                     New/modified files to be merged into the HCCL repository (organized by repository-relative path)
│   ├── CMakeLists.txt                       [Modified] Add algo_plugin paths to INCLUDE_LIST + add_subdirectory(algo_plugin)
│   ├── algo_plugin/                         [New] HcclAlgoPluginMgr, compiled together with libhccl.so
│   │   ├── hccl_algo_plugin_mgr.h/.cc       Plugin loading, parameter conversion (FillHcclAlgoPluginParam), singleton management
│   │   ├── inc/hccl_algo_plugin_common.h    Common structure definitions such as HcclAlgoPluginParam / HcclAlgoPluginAlgEntry
│   │   ├── inc/hccl_algo_plugin_broker_api.h HcclAlgoPlugin_t function-table definition
│   │   └── CMakeLists.txt                   [New] target_sources(hccl PRIVATE ...) + link dl
│   └── ops/op_common/
│       ├── op_common.cc                     [Modified] Insert Plugin-first matching and execution branches into Selector()/HcclExecOp()
│       └── inc/alg_param.h                  [Modified] Add pluginSelected flag to OpParam
│
├── plugin_broker/                           PluginBroker dynamic library, built as an independent project and fully decoupled from the HCCL main library
│   ├── include/hccl_algo_plugin_broker_internal.h
│   ├── src/plugin_broker.cc                 Directory scanning, algorithm-registry construction, SelectAlg/ExecuteAlg/QueryAlgs implementation
│   └── CMakeLists.txt                       Produces libhccl_algo_PluginBroker.so
│
├── sdk/
│   └── hccl_algo_plugin_sdk.h               Custom algorithm development SDK: REGISTER_HCCL_ALGO macro + automatic registry +
│                                             standard HcclAlgoPluginQueryEntries() implementation; algorithm developers only need to include this header
│
└── example/                                 Two examples that validate the plugin framework's registration/selection/dispatch path
    ├── AllReduce/                           Scenario 1: two algorithms (AllReduceAlgoSmall/Large) share one implementation .so
    │   ├── op_host/allreduce_custom_algos.cc
    │   ├── selector/allreduce_selector.cc
    │   └── CMakeLists.txt
    └── Broadcast/                           Scenario 2: one algorithm (BroadcastAlgoTree) has its own implementation .so
        ├── op_host/broadcast_custom_algo.cc
        ├── selector/broadcast_selector.cc
        └── CMakeLists.txt
```

## Architecture

```
HCCL main library (libhccl.so)          PluginBroker (independent .so)       Custom algorithms (independent .so)
┌───────────────────────┐               ┌──────────────────────┐             ┌─────────────────────────┐
│ Selector()            │               │                      │             │ libhccl_plugin_         │
│  └ HcclAlgoPluginMgr  │──SelectAlg→   │ PluginBrokerContext  │─dlopen→     │ {op}_selector.so        │
│      ::Instance()     │               │  ::SelectAlg()       │             │  REGISTER_HCCL_ALGO(...)│
│                       │               │                      │             │  Select()               │
│ HcclExecOp()          │               │                      │             └─────────────────────────┘
│  └ (if pluginSelected)│──ExecuteAlg→  │ PluginBrokerContext  │─dlopen→     ┌─────────────────────┐
│      call exec func   │               │  ::ExecuteAlg()      │             │ lib{Xxx}Impl.so      │
└───────────────────────┘               └──────────────────────┘             │  fnSymbol(...)      │
                                                                              └─────────────────────┘
```

- **HcclAlgoPluginMgr** (`src/algo_plugin/`): Compiled into `libhccl.so`. At the entry of `Selector()`, it lazily loads PluginBroker through `dlopen(HCCL_ALGO_PLUGIN_PATH)`, obtains the `HcclAlgoPlugin_t` function table, and then invokes the PluginBroker interfaces.
- **PluginBroker** (`plugin_broker/`): Independently built as `libhccl_algo_PluginBroker.so`. When loaded, its internal global static object automatically scans each operator subdirectory under `HCCL_PLUGIN_ALG_DIR`, `dlopen`s the corresponding `..._selector.so`, retrieves the registered algorithm entries (algorithm name / implementation .so path / execution-function symbol name), and builds the global algorithm registry.
- **Custom algorithm developers**: Include `sdk/hccl_algo_plugin_sdk.h`, register algorithms with the `REGISTER_HCCL_ALGO` macro, implement the `Select()` decision function, and compile `libhccl_plugin_{op}_selector.so`. The algorithm body is compiled separately into an implementation `.so` and only needs to export an execution function that matches the standard signature of the corresponding operator; it does not need to depend on HCCL internal structures.

## Build

```bash
source ~/Ascend/cann/set_env.sh

# 1) Build the HCCL main library and produce libhccl.so
bash build.sh --pkg -p ~/Ascend
bash ./build_out/cann-hccl_*.run --full

# 2) Build PluginBroker
cd plugin_broker && mkdir -p build && cd build
cmake .. && make -j
# Produces libhccl_algo_PluginBroker.so
cd ../..

# 3) Build the AllReduce example (two algorithms share one implementation .so)
cd example/AllReduce && mkdir -p build && cd build
cmake .. -DASCEND_HOME=$ASCEND_HOME_PATH && make -j
# Produces libhccl_plugin_allreduce_selector.so and libAllReduceCustomAlgosImpl.so
cd ../../..

# 4) Build the Broadcast example (one algorithm has its own implementation .so) to validate cross-operator routing
cd example/Broadcast && mkdir -p build && cd build
cmake .. -DASCEND_HOME=$ASCEND_HOME_PATH && make -j
cd ../../..
```

## Deployment and Enabling

```bash
mkdir -p ~/hccl_plugin_broker ~/hccl_plugin_algs/AllReduce ~/hccl_plugin_algs/Broadcast

cp plugin_broker/build/libhccl_algo_PluginBroker.so ~/hccl_plugin_broker/

cp example/AllReduce/build/libhccl_plugin_allreduce_selector.so \
   example/AllReduce/build/libAllReduceCustomAlgosImpl.so \
   ~/hccl_plugin_algs/AllReduce/

cp example/Broadcast/build/libhccl_plugin_broadcast_selector.so \
   example/Broadcast/build/libBroadcastCustomAlgoImpl.so \
   ~/hccl_plugin_algs/Broadcast/

export HCCL_ALGO_PLUGIN_PATH=~/hccl_plugin_broker/libhccl_algo_PluginBroker.so
export HCCL_PLUGIN_ALG_DIR=~/hccl_plugin_algs
```

### Environment Variables

| Environment Variable | Purpose | Read By |
| --- | --- | --- |
| `HCCL_ALGO_PLUGIN_PATH` | Specifies the absolute path to `libhccl_algo_PluginBroker.so` | `HcclAlgoPluginMgr` (HCCL side, `src/algo_plugin/hccl_algo_plugin_mgr.cc`) |
| `HCCL_PLUGIN_ALG_DIR` | Specifies the root directory of custom algorithms (one same-name subdirectory per operator) | `PluginBroker` (`plugin_broker/src/plugin_broker.cc`) |

If either variable is not set, or if PluginBroker loading/validation fails, the plugin framework as a whole does not take effect and HCCL behavior remains completely unchanged.

## Validation

Using AllReduce as an example, after a successful run, logs similar to the following should appear on stderr:

```
[HCCL-ALGO-PluginBroker][INFO] op [AllReduce] registered 2 custom algorithm(s) from .../libhccl_plugin_allreduce_selector.so
[AllReduceSelector][Select] totalBytes=..., hit=AllReduceAlgoSmall
```

## Description of the Two Example Algorithms

The two examples are intended only to verify whether the **plugin framework itself** has a working registration/selection/dispatch path. They are **not** production-ready AllReduce/Broadcast algorithm implementations. Their execution functions are stubs that only print logs and return success, without performing real data movement or reduction computation. Together, they cover the following three typical scenarios:

| Example | Validation Scenario | Description |
| --- | --- | --- |
| `AllReduce/` | Multiple algorithms map to the same implementation .so | `AllReduceAlgoSmall` and `AllReduceAlgoLarge` both point to `libAllReduceCustomAlgosImpl.so`; `Select()` chooses one according to data size, using 1 MB as the threshold |
| `Broadcast/` | One algorithm has its own implementation .so | `BroadcastAlgoTree` exclusively uses `libBroadcastCustomAlgoImpl.so`; `Select()` matches only when root is 0 |
| Deploying both together | Different operators are correctly routed to their own selector/.so without interfering with each other | PluginBroker must locate the `AllReduce/` or `Broadcast/` subdirectory according to `opName` |

## Known Limitations

- **`PluginBroker::ExecuteAlg()` currently supports dispatch for 9 operators**: Send / Recv / Broadcast / AllReduce / Reduce / AllGather / ReduceScatter / AllToAll (non-V equal-size scenario) / Scatter. For operators involving variable-length parameters or multiple items, such as `AllToAllV`, `AllToAllVC`, `AllGatherV`, `ReduceScatterV`, `BatchSendRecv`, and `Barrier`, `sdk/hccl_algo_plugin_sdk.h` does not yet define the corresponding standard execution-function signatures, so they are currently unsupported and uniformly return `HCCL_E_NOT_SUPPORT` when matched. To support them, the standard signatures for the corresponding operators must first be added to the SDK, and then the branches in `ExecuteAlg()` must be extended.
- **The example algorithms are pure stub implementations**: They do not perform real communication or reduction computation and are used only to validate the framework path. They cannot be used to evaluate performance or correctness.

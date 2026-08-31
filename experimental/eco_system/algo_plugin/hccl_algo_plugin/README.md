# HCCL-ALGO-Plugin 实现

　　本目录是 [`HCCL-ALGO-Plugin RFC`](../../../../docs/zh/rfcs/0002-HCCL-ALGO-Plugin.md) 设计文档的落地代码，在 `HCCL-ALGO-Plugin-master` 代码仓基础上新增了"自定义集合通信算法插件"能力：允许用户在不修改、不重编HCCL主库源码的前提下，以独立动态库的形式为指定算子接入自定义算法实现，并按运行时参数（数据量、拓扑等）动态决定是否命中。

　　未设置相关环境变量时，HCCL行为与原有完全一致。

## 目录结构

```
├── src/                                     需合入HCCL代码仓的新增/修改文件（按仓库相对路径组织）
│   ├── CMakeLists.txt                      【已修改】INCLUDE_LIST新增algo_plugin路径 + add_subdirectory(algo_plugin)
│   ├── algo_plugin/                        【新增】HcclAlgoPluginMgr，随libhccl.so一同编译
│   │   ├── hccl_algo_plugin_mgr.h/.cc      Plugin加载、参数转换（FillHcclAlgoPluginParam）、单例管理
│   │   ├── inc/hccl_algo_plugin_common.h   HcclAlgoPluginParam / HcclAlgoPluginAlgEntry 等公共结构体定义
│   │   ├── inc/hccl_algo_plugin_broker_api.h   HcclAlgoPlugin_t 函数表定义
│   │   └── CMakeLists.txt                  【新增】target_sources(hccl PRIVATE ...) + link dl
│   └── ops/op_common/
│       ├── op_common.cc                    【已修改】Selector()/HcclExecOp()中插入Plugin优先匹配与执行分支
│       └── inc/alg_param.h                 【已修改】OpParam新增pluginSelected标记字段
│
├── plugin_broker/                          PluginBroker动态库，独立工程，与HCCL主库完全解耦编译
│   ├── include/hccl_algo_plugin_broker_internal.h
│   ├── src/plugin_broker.cc                目录扫描、算法注册表构建、SelectAlg/ExecuteAlg/QueryAlgs实现
│   └── CMakeLists.txt                      产出 libhccl_algo_PluginBroker.so
│
├── sdk/
│   └── hccl_algo_plugin_sdk.h              自定义算法开发SDK：REGISTER_HCCL_ALGO宏 + 自动注册表 +
│                                            HcclAlgoPluginQueryEntries()标准实现，算法开发者只需引用此头文件
│
└── example/                                两个示例，验证插件框架本身的注册/选择/派发链路
    ├── AllReduce/                          验证场景一：两个算法（AllReduceAlgoSmall/Large）共用一个实现so
    │   ├── op_host/allreduce_custom_algos.cc
    │   ├── selector/allreduce_selector.cc
    │   └── CMakeLists.txt
    └── Broadcast/                          验证场景二：一个算法（BroadcastAlgoTree）独占一个实现so
        ├── op_host/broadcast_custom_algo.cc
        ├── selector/broadcast_selector.cc
        └── CMakeLists.txt
```

## 架构

```
HCCL主库(libhccl.so)                   PluginBroker(独立.so)          自定义算法(独立.so)
┌───────────────────────┐             ┌──────────────────────┐        ┌─────────────────────────┐
│ Selector()            │             │                      │        │ libhccl_plugin_         │
│  └ HcclAlgoPluginMgr  │──SelectAlg→ │ PluginBrokerContext  │─dlopen→│ {op}_selector.so        │
│      ::Instance()     │             │  ::SelectAlg()       │        │  REGISTER_HCCL_ALGO(...)│
│                       │             │                      │        │  Select()               │
│ HcclExecOp()          │             │                      │        └─────────────────────────┘
│  └ (若pluginSelected) │──ExecuteAlg→│ PluginBrokerContext  │─dlopen→┌─────────────────────┐
│      直接调用执行函数  │             │  ::ExecuteAlg()       │       │ lib{Xxx}Impl.so      │
└───────────────────────┘             └──────────────────────┘        │  fnSymbol(...)      │
                                                                      └─────────────────────┘
```

- **HcclAlgoPluginMgr**（`src/algo_plugin/`）：编译进 `libhccl.so`，在 `Selector()` 入口处通过`dlopen(HCCL_ALGO_PLUGIN_PATH)` 懒加载 PluginBroker，取到 `HcclAlgoPlugin_t` 函数表后即可调用。
- **PluginBroker**（`plugin_broker/`）：独立编译的 `libhccl_algo_PluginBroker.so`。加载时其内部全局静态对象自动扫描 `HCCL_PLUGIN_ALG_DIR` 下每个算子子目录，`dlopen` 对应的 `..._selector.so` 取出已注册的算法条目（算法名 / 实现so路径 / 执行函数符号名），构建全局算法注册表。
- **自定义算法开发者**：引用 `sdk/hccl_algo_plugin_sdk.h`，用 `REGISTER_HCCL_ALGO` 宏注册算法、实现`Select()` 决策函数，编译出 `libhccl_plugin_{op}_selector.so`；算法本体单独编译成实现so，只需按各算子标准签名导出执行函数即可，无需感知HCCL内部结构体。

## 构建

```bash
source ~/Ascend/cann/set_env.sh

# 1) 编译HCCL主库，产出 libhccl.so
bash build.sh --pkg -p ~/Ascend
bash ./build_out/cann-hccl_*.run --full

# 2) 编译 PluginBroker
cd plugin_broker && mkdir -p build && cd build
cmake .. && make -j
# 产出 libhccl_algo_PluginBroker.so
cd ../..

# 3) 编译 AllReduce 示例（两个算法共用一个实现so）
cd example/AllReduce && mkdir -p build && cd build
cmake .. -DASCEND_HOME=$ASCEND_HOME_PATH && make -j
# 产出 libhccl_plugin_allreduce_selector.so、libAllReduceCustomAlgosImpl.so
cd ../../..

# 4) 编译 Broadcast 示例（一个算法独占一个实现so），验证跨算子路由
cd example/Broadcast && mkdir -p build && cd build
cmake .. -DASCEND_HOME=$ASCEND_HOME_PATH && make -j
cd ../../..
```

## 部署与启用

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

### 环境变量

| 环境变量 | 作用 | 读取方 |
| --- | --- | --- |
| `HCCL_ALGO_PLUGIN_PATH` | 指定 `libhccl_algo_PluginBroker.so` 的绝对路径 | `HcclAlgoPluginMgr`（HCCL侧，`src/algo_plugin/hccl_algo_plugin_mgr.cc`） |
| `HCCL_PLUGIN_ALG_DIR` | 指定自定义算法根目录（每个算子一个同名子目录） | `PluginBroker`（`plugin_broker/src/plugin_broker.cc`） |

两者任一未设置，或PluginBroker加载/校验失败，插件框架整体不生效，HCCL行为与原有完全一致。

## 验证

以 AllReduce 为例，跑通后应能在stderr中看到如下类似日志：

```
[HCCL-ALGO-PluginBroker][INFO] op [AllReduce] registered 2 custom algorithm(s) from .../libhccl_plugin_allreduce_selector.so
[AllReduceSelector][Select] totalBytes=..., hit=AllReduceAlgoSmall
```

## 两个示例算法说明

两个示例的目的仅为验证**插件框架本身**的注册/选择/派发链路是否打通，**不是**可直接用于生产的真实AllReduce/Broadcast算法实现——其执行函数内部只做打桩（打印日志 + 返回成功），不做真实的数据搬运与规约计算。分别覆盖以下三种典型场景：

| 示例 | 验证场景 | 说明 |
| --- | --- | --- |
| `AllReduce/` | 多个算法对应同一个实现so | `AllReduceAlgoSmall`/`AllReduceAlgoLarge` 均指向 `libAllReduceCustomAlgosImpl.so`，`Select()`按数据量（1MB为界）二选一 |
| `Broadcast/` | 一个算法独占一个实现so | `BroadcastAlgoTree` 独占 `libBroadcastCustomAlgoImpl.so`，`Select()`仅当root为0时命中 |
| 二者合并部署 | 不同算子分别正确路由到各自的selector/so，互不干扰 | PluginBroker需要根据opName定位到`AllReduce/`还是`Broadcast/`子目录 |

## 已知限制

- **`PluginBroker::ExecuteAlg()` 目前支持9个算子的分发**：Send / Recv / Broadcast / AllReduce /Reduce / AllGather / ReduceScatter / AllToAll（非V等长场景）/ Scatter。`AllToAllV`、`AllToAllVC`、`AllGatherV`、`ReduceScatterV`、`BatchSendRecv`、`Barrier` 等涉及变长参数或多item的算子，`sdk/hccl_algo_plugin_sdk.h` 尚未定义对应的标准执行函数签名，暂不支持，命中时统一返回`HCCL_E_NOT_SUPPORT`。如需支持，需先在SDK中补充对应算子的标准签名，再在 `ExecuteAlg()` 中扩展分支。
- **示例算法为纯打桩实现**：不做真实的通信与规约计算，仅用于验证框架链路，不能用来评估性能或正确性。
# HCCL工程兼容性方案梳理

## 1. 为什么需要处理兼容性

HCCL（Huawei Collective Communication Library）作为CANN（Compute Architecture for Neural Networks）生态中的集合通信库，其运行环境存在两个维度的差异，必须在同一份源码中同时覆盖：

**CANN版本差异。** HCCL依赖的下层通信运行库hcomm以及CANN SDK头文件会随版本演进不断新增API和类型定义。例如8.5.0 不存在 `HcclThreadAcquireWithConfig`、`HcclRankGraphGetTopoInstsByLayer` 等接口，也不存在 `HcclCommStatus`、`ThreadConfig`、`EndpointAttr` 等类型；9.0.0 引入了CCU（Communication Control Unit）相关能力；9.1.0 又新增了 `HcclRankGraphGetEndpointNum` 等拓扑查询接口和fast-launch路径。HCCL需要用同一份代码编译出能在8.5.0、9.0.0、9.1.0 等多个版本上运行的二进制。

**运行环境差异（Host侧vs Device侧）。** HCCL的代码既运行在Host CPU上（负责通信域管理、资源分配、算子编排），也运行在Device AICPU上（负责实际的notify等待、数据搬运执行）。两侧能调用的底层API集合完全不同：Host侧通过 `libhcomm.so` 获取资源管理、拓扑查询、CCU等接口；Device侧通过 `libccl_kernel.so` 获取通信原语、device侧profiling、诊断等接口。两侧的兼容性探测必须独立进行，不能交叉引用。

此外，SDK头文件中的设备类型枚举命名也经历了变更（`DEV_TYPE_910_95` → `DEV_TYPE_950`），需要通过外部注入的编译宏来切换。

综上，HCCL的兼容性方案沿着"编译时vs运行时"和"Host侧vs Device侧"两条轴线展开，形成四象限的完整覆盖。

---

## 2. 编译时兼容性

### 2.1 需求

编译时兼容性解决的问题是：**当用不同版本的CANN SDK头文件编译HCCL源码时，代码必须能正确编译通过，并且编译产物的行为与该版本SDK匹配。** 具体包括：

- 旧版SDK头文件中缺失的类型定义，需要HCCL自行补桩。
- 仅在新版本中可用的API声明和算法实现，需要通过条件编译排除或纳入。
- SDK头文件中枚举值命名变更，需要通过宏切换。

>**注意**：我们所说的编译兼容指的是**社区正式发布的不同版本**之间的兼容，而不是要求公司内部daily版本之间的兼容。例如：目前CANN主线处在9.1.0版本，HCOMM今天上了一个新接口，HCCL仓上了一笔代码调用该新接口，上了之后CANN版本号没有变化，仍然是9.1.0，那么需保证HCCL仓当前的代码能用**低于**9.1.0的CANN SDK头文件编译通过，但不用保证当前代码能用今天以前的9.1.0版本CANN编译通过（也无法保证）。
>
>**社区发布版本公告**：https://www.hiascend.com/productbulletins?tab=CANN

### 2.2 核心机制：CANN_VERSION_NUM版本号

#### 版本号的构造与注入

版本号体系定义在 `src/common/hcomm_dlsym/dlsym_common.h:18-22`：

```c
#define CANN_VERSION_VAL(M, m, p) ((M) * 10000000 + (m) * 100000 + (p) * 1000)
#define CANN_VERSION_3(M, m, p)    (CANN_VERSION_VAL(M, m, p))
#define CANN_VERSION_4(M, m, p, b) (CANN_VERSION_VAL(M, m, p) - 200 + (b))
#define CANN_VERSION_PICK(_1, _2, _3, _4, NAME, ...) NAME
#define CANN_VERSION(...) CANN_VERSION_PICK(__VA_ARGS__, CANN_VERSION_4, CANN_VERSION_3)(__VA_ARGS__)
```

`CANN_VERSION(M, m, p)` 生成正式版本号的整数值，`CANN_VERSION(M, m, p, b)` 生成beta版本号。beta版本是两个正式版本之间发布的社区预览版，其版本号通过 `正式版本号 - 200 + b` 计算得出，使其在数值上落在前一个正式版本和当前正式版本之间。以9.1.0 系列为例：

| 版本 | 计算 | 数值 |
|------|------|------|
| 9.0.0 | 9×10000000 + 0×100000 + 0×1000 | 90000000 |
| 9.1.0-beta.1 | 91000000 - 200 + 1 | 90999801 |
| 9.1.0-beta.2 | 91000000 - 200 + 2 | 90999802 |
| 9.1.0 | 9×10000000 + 1×100000 + 0×1000 | 91000000 |

由此得到版本大小关系：**9.1.0 > 9.1.0-beta.2 > 9.1.0-beta.1 > 9.0.0**，代码中可直接用 `CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0, 1)` 这样的写法来精确匹配从某个beta版本开始的特性。

> **举例**：当前CANN主线版本是9.1.0，已经拉出的最新商分版本9.1.0-beta.2，如果我们的代码既要上主线又要上商分，那么包含我们这笔代码的最小版本就是9.1.0-beta.2，代码中判断版本号来确定是否包含该特性时就要用CANN_VERSION(9, 1, 0, 2)；如果我们的代码只上主线，不上商分，那么代码中判断版本号时就要用CANN_VERSION(9, 1, 0)。

`CANN_VERSION_NUM` 是一个由CMake在编译时注入的整数宏，代表当前编译所用CANN SDK的版本。

CMake的注入逻辑位于 `src/CMakeLists.txt:14-72`，根据编译模式不同有两种来源：

- **开源仓编译（BUILD_OPEN_PROJECT）**：从CANN包的 `include/version/cann_version.h` 文件中解析 `#define CANN_VERSION_NUM` 的值。
- **大工程编译**：从HCCL自身的 `version.cmake` 中的 `set_cann_package(hccl VERSION "X.Y.Z")` 语句解析版本号并计算。

解析出的值通过 `hccl_apply_cann_compat()` 函数（`src/CMakeLists.txt:67-72`）以 `target_compile_definitions` 的方式注入到每个编译目标的私有定义中：

```cmake
function(hccl_apply_cann_compat target_name)
    target_compile_definitions(${target_name} PRIVATE CANN_VERSION_NUM=${HCCL_CANN_VERSION_NUM})
endfunction()
```

此外，CMake还会根据版本号推导出一个布尔开关 `HCCL_CANN_COMPAT_850`（`src/CMakeLists.txt:59-62`）：

```cmake
set(HCCL_CANN_COMPAT_850 OFF)
if(HCCL_CANN_VERSION_NUM GREATER 0 AND HCCL_CANN_VERSION_NUM LESS 90000000)
    set(HCCL_CANN_COMPAT_850 ON)
endif()
```

当版本号小于9.0.0（`90000000`）时，该开关为ON，表示当前编译目标需要兼容8.5.0。

#### 条件编译的使用方式

**方式一：`#if CANN_VERSION_NUM >= CANN_VERSION(x, y, z)` 守卫代码块**

这是最常见的用法，用于条件性地包含头文件、注册算法或编译特定函数体。例如在 `src/ops/scatter/executor/ins_v2_scatter_sole_executor.cc:14-21`，CCU相关的模板头文件仅在9.0.0+ 编译时才引入：

```cpp
#ifndef AICPU_COMPILE
#include "aiv_temp_scatter_mesh_1D.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_scatter_mesh1d.h"
#include "ccu_temp_scatter_nhr1d_mem2mem.h"
#include "ccu_kernel_scatter_nhr1d_mem2mem.h"
#endif
#endif
```

在 `src/ops/all_gather/executor/ins_v2_all_gather_sole_executor.cc:270-284`，CCU算法变体仅在9.0.0+ 时注册：

```cpp
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherMesh1DMem2Mem, ...);
#endif
```

在 `src/ops/op_common/executor/channel/channel.cc:199` 和 `src/ops/op_common/op_common.cc:282`，9.1.0 新增的拓扑查询和CCU fast-launch路径用更细粒度的版本号守卫：

```cpp
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    // 9.1.0 新增的HcclRankGraphGetEndpointNum等接口逻辑
#endif
```

**方式二：`#if !defined(HCCL_CANN_COMPAT_850)` 排除旧版本代码**

这种写法更简洁，专门用于在8.5.0 上排除CCU相关代码。例如 `src/ops/op_common/executor/channel/channel.cc:360-364`：

```cpp
#if !defined(HCCL_CANN_COMPAT_850) && !defined(AICPU_COMPILE)
    channels.clear();
    // ... CCU相关的channel计算逻辑
#endif
```

在CMake层面，`HCCL_CANN_COMPAT_850` 还用于跳过整个源文件的编译。在 `src/ops/*/CMakeLists.txt` 中广泛使用：

```cmake
if(NOT HCCL_CANN_COMPAT_850)
    target_sources(${TARGET} PRIVATE ...)
endif()
```

**方式三：旧版SDK缺失类型的桩定义**

当旧版CANN头文件不包含某些类型定义时，HCCL在自己的兼容层头文件中补桩。这些桩定义用 `#if CANN_VERSION_NUM < CANN_VERSION(...)` 守卫，仅当编译时的SDK版本不够时才生效。例如：

`dlsym_common.h:30-41` 为8.5.0 补充 `HcclCommStatus` 和 `ThreadHandle`：

```c
#if CANN_VERSION_NUM < CANN_VERSION(9, 0, 0)
typedef enum {
    HCCL_COMM_STATUS_READY = 0,
    HCCL_COMM_STATUS_SUSPENDING = 1,
    HCCL_COMM_STATUS_INVALID = 254,
    HCCL_COMM_STATUS_RESERVED = 255
} HcclCommStatus;
typedef uint64_t ThreadHandle;
#endif
```

`hccl_res_dl.h:47-65` 为9.1.0 之前的版本补充 `ThreadType`、`ThreadConfig` 以及 `ThreadConfigInit`：

```c
#if CANN_VERSION_NUM < CANN_VERSION(9, 1, 0)
typedef enum {
    THREAD_TYPE_INVALID = -1,
    THREAD_TYPE_TS = 0
} ThreadType;
typedef struct {
    uint32_t notifyNumPerThread;
} ThreadConfig;

static inline HcommResult ThreadConfigInit(ThreadConfig *config, uint32_t num)
{
    for (uint32_t i = 0; i < num; i++) {
        config[i].notifyNumPerThread = 0;
    }
    return 0;
}
#endif
```

`hccl_rank_graph_dl.h:18-32` 为9.0.0_beta.2 之前的版本补充 `EndpointAttr` 等类型。

`hccl_host_comm_dl.h:18-37` 为9.0.0 之前的版本补充 `HcclOpExpansionMode` 和 `HcclConfigType`。

**方式四：旧版SDK缺失inline函数的桩定义**

方式三中的 `ThreadConfigInit` 不仅是缺失类型，还是一个 `static inline` 函数。这类函数无法使用运行时弱符号方案（`DEFINE_WEAK_FUNC`），只能通过编译时 `#if` 桩来处理。这涉及一个重要的判别准则：**什么样的函数能用弱符号方案，什么样的不能。**（弱符号方案详见第3章节“运行时兼容性”）

弱符号方案（`DEFINE_WEAK_FUNC` + `dlsym`）能工作的前提是函数具有**外部链接（external linkage）**，并在共享库中产生**动态符号**。整条链路如下：

```text
头文件声明 (extern)  →  .cc中DEFINE_WEAK_FUNC生成weak桩定义
                         →  编译进 .so产生弱动态符号
                         →  dlsym在运行库中探测符号是否存在
                         →  真实 .so的强符号覆盖弱桩
```

如果函数满足以下条件，就可以用弱符号方案：

1. 函数在SDK头文件中以 `extern` 声明（非 `static`、非 `inline`），实际定义在SDK的 `.so` 中。
2. 函数被编译为具有外部链接的符号，在运行库的导出符号表中可以查到。
3. 调用方通过函数指针或直接调用引用该符号，链接器能在加载时完成符号重定位。

HCCL中绝大部分底层API（如 `HcclThreadAcquireWithConfig`、`HcommWriteWithNotifyOnThread`、`HcclCommGetStatus` 等）都满足这些条件，因此都采用了 `DEFINE_WEAK_FUNC` 方案。

如果函数**不满足**上述条件——典型情况是SDK头文件中以 `static inline` 或 `inline` 定义的函数——则弱符号方案失效，原因如下：

- `static inline` 函数具有**内部链接**，每个包含该头文件的编译单元各自获得一份私有副本，不产生任何动态符号。
- 编译器通常将函数体直接内联展开到调用点；即使不内联，也不会导出到 `.so` 符号表中。
- `dlsym` 查不到该符号（返回 `nullptr`），`INIT_SUPPORT_FLAG` 永远为 `false`，探测毫无意义。
- 如果在 `.cc` 中同时用 `DEFINE_WEAK_FUNC` 声明 `extern weak` 版本，而新SDK头文件中已有 `static inline` 定义，两者在同一编译单元中相遇会导致编译冲突（同一函数名既有 `static inline` 定义又有 `extern` 声明，ill-formed）。

对于这类函数，唯一可行的兼容手段是编译时 `#if` 桩：

- 旧版SDK（头文件中没有该函数）：`#if CANN_VERSION_NUM < CANN_VERSION(...)` 块内提供 `static inline` 定义，使代码能编译通过，函数体在编译时直接内联到调用点。
- 新版SDK（头文件中已有该函数）：`#if` 块跳过，使用SDK原版 `inline` 定义。
- 选择权完全在预处理器，不经过链接器或动态加载器。

简言之：**弱符号 + dlsym是运行时机制，只适用于有动态导出符号的外部链接函数；`static inline` 函数在编译时内联展开，没有动态符号可供探测，只能用 `#if` 在编译时做选择。**

### 2.3 设备类型枚举兼容：MACRO_DEV_TYPE_NEW

SDK头文件中的 `DevType` 枚举经历了命名变更：旧版使用 `DEV_TYPE_910_95`，新版改为 `DEV_TYPE_950`。这个差异不由CANN版本号控制，而是由外部构建环境注入的 `MACRO_DEV_TYPE_NEW` 宏切换。该宏在HCCL源码中没有被 `#define`，完全由SDK/构建环境在编译时提供。

典型用法（`src/ops/all_reduce/all_reduce_op.cc:31-39`）：

```cpp
#ifdef MACRO_DEV_TYPE_NEW
if (deviceType != DevType::DEV_TYPE_950) {
#else
if (deviceType != DevType::DEV_TYPE_910_95) {
#endif
    return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
}
```

这种模式在所有算子入口文件（`all_reduce_op.cc`、`all_gather_op.cc`、`broadcast_op.cc`、`reduce_op.cc` 等）和拓扑匹配文件（`topo_match_ubx.cc`、`topo_match_1d.cc` 等）中重复出现。

### 2.4 编译时兼容的局限性

编译时兼容只能根据**编译时**的SDK版本做决策。但HCCL编译出的 `.so` 可能被加载到安装了不同版本CANN运行库的环境中。例如用9.0.0 SDK编译的HCCL，可能运行在只安装了8.5.0 运行库的环境中（反之亦然）。此时编译时的 `#if` 分支已经固化在二进制中，无法再改变。这就要靠运行时兼容性来弥补。

---

## 3. 运行时兼容性

### 3.1 需求

运行时兼容性解决的问题是：**HCCL编译出的 `.so` 在加载到目标环境后，需要探测实际可用的底层API，并在API不存在时安全降级。** 具体包括：

- 某些接口可能在编译时存在（头文件中有声明）但运行时的 `.so` 中未导出（旧版本运行库）。
- 某些接口可能在编译时不存在（旧版SDK头文件）但运行时的 `.so` 中已存在（新版本运行库）。
- 需要在不崩溃的前提下选择正确的代码路径。

### 3.2 核心机制：dlopen + dlsym + weak symbol

整个运行时兼容方案的核心位于 `src/common/hcomm_dlsym/` 目录下。其基本思路是：

1. 运行时 `dlopen` 打开底层通信运行库（Host侧为 `libhcomm.so`，Device侧为 `libccl_kernel.so`）。
2. 对每个需要兼容的接口，用 `dlsym` 探测该符号是否存在。
3. 根据探测结果设置一个布尔标志位，供上层代码查询。
4. 同时提供一个weak symbol的桩函数作为后备实现，确保即使真实接口不存在也不会产生链接错误或运行时崩溃。

### 3.3 宏定义工具箱：dlsym_common.h

`src/common/hcomm_dlsym/dlsym_common.h` 定义了四个核心宏，构成了整个运行时兼容方案的基础设施：

#### （1）DECL_WEAK_FUNC — 声明弱符号（放在头文件中）

```c
#define DECL_WEAK_FUNC(type, func_name, ...) \
    type func_name(__VA_ARGS__) __attribute__((weak))
```

展开后是一个带 `__attribute__((weak))` 的函数声明。如果链接时找不到该符号的定义，弱符号会被解析为 `nullptr`/`0`，不会产生链接错误。

#### （2）DEFINE_WEAK_FUNC — 定义弱桩函数 + 支持查询函数（放在源文件中）

```c
#define DEFINE_WEAK_FUNC(type, func_name, ...) \
    static bool g_##func_name##Supported = false; \
    extern "C" bool HcommIsSupport##func_name(void) { \
        return g_##func_name##Supported; \
    } \
    type func_name(__VA_ARGS__) __attribute__((weak)); \
    type func_name(__VA_ARGS__) \
    { \
        HCCL_COMPAT_ERROR("[HcclWrapper] %s not supported", __func__); \
        return (type)(-1); \
    }
```

这个宏一次性生成三样东西：

- **静态布尔标志** `g_<func_name>Supported`：记录该接口是否被运行库支持，初始为 `false`。
- **支持查询函数** `HcommIsSupport<func_name>()`：`extern "C"` 强符号，返回标志的值。上层代码通过调用此函数来判断接口是否可用。
- **弱桩实现** `<func_name>()`：带 `weak` 属性的函数定义，当运行库中不存在真实实现时作为后备，打印错误日志并返回错误码。

关键点：`HcommIsSupport<func_name>` 是**强符号**（非weak），它必须有定义存在，否则链接报错。而 `<func_name>` 本身是**弱符号**，如果运行库提供了真实实现，链接器会优先选择真实实现覆盖弱桩。

#### （3）DECL_SUPPORT_FLAG — 声明支持查询函数（放在头文件中）

```c
#define DECL_SUPPORT_FLAG(func_name) \
    extern "C" bool HcommIsSupport##func_name(void)
```

仅声明查询函数，不声明弱符号。某些场景下上层只需要查询接口是否存在，不需要直接调用接口本身（直接调用走的是其他路径），就只声明support flag。

#### （4）INIT_SUPPORT_FLAG — 运行时探测并设置标志（放在初始化函数中）

```c
#define INIT_SUPPORT_FLAG(handle, func_name) \
    do { \
        void *ptr = (void *)dlsym(handle, #func_name); \
        if (ptr == nullptr) { \
            g_##func_name##Supported = false; \
        } else { \
            g_##func_name##Supported = true; \
        } \
    } while(0)
```

用 `dlsym` 在已打开的运行库句柄上查找符号。找到则置标志为 `true`，否则为 `false`。注意这里查找的是真实接口名（`#func_name` 字符串化），而不是 `HcommIsSupport` 开头的查询函数名。

### 3.4 打桩文件的组织

`src/common/hcomm_dlsym/` 目录下的文件按API领域分类，每个 `*_dl.h` / `*_dl.cc` 文件对应一组底层接口：

| 文件 | 负责的API领域 | 归属 |
|------|----------------|------|
| `hccl_res_dl.cc/h` | 资源获取（HcclThreadAcquire, HcclDevMemAcquire, HcclThreadAcquireWithConfig等） | Host侧 |
| `hccl_rank_graph_dl.cc/h` | 拓扑图查询（HcclRankGraphGetTopoInstsByLayer等） | Host侧 |
| `hcomm_primitives_dl.cc/h` | 通信原语（HcommWrite/Read/Notify/Fence等） | Host + Device共享 |
| `hccl_inner_dl.cc/h` | 内部接口（HcclCreateOpResCtxInner） | Host侧 |
| `hcomm_host_profiling_dl.cc/h` | Host侧profiling（HcommProfilingRegThread等） | Host侧 |
| `hccl_host_comm_dl.cc/h` | Host侧通信域管理（HcclCommGetStatus, HcclConfigGetInfo） | Host侧 |
| `hccl_res_expt_dl.cc/h` | 实验性资源接口（HcclCommAddExchangeInfo等） | Host侧 |
| `ccu_res_dl.cc/h` | CCU资源（HcommCcuGetMemToken） | Host侧 |
| `hccl_ccu_res_dl.cc/h` | CCU通信域查询（HcclCommQueryCcuIns） | Host侧 |
| `ccu_launch_dl.cc/h` | CCU kernel注册与下发（HcommCcuKernelLaunch等） | Host侧 |
| `ccu_primitives_impl_dl.cc/h` | CCU原语（CcuVariable/Address/Event/Buffer操作、控制流等） | Host侧 |
| `hcomm_device_profiling_dl.cc/h` | Device侧profiling（HcommProfilingReportKernelStartTask等） | Device侧 |
| `hcomm_diag_dl.cc/h` | 诊断接口（HcommRegOpInfo等） | Device侧 |
| `hccl_device_comm_dl.cc/h` | Device侧通信域（HcclCommGetStatus） | Device侧 |
| `hcomm_dlsym.cc/h` | Host侧聚合初始化 | Host侧 |
| `hcomm_device_dlsym.cc/h` | Device侧聚合初始化 | Device侧 |

每个 `*_dl.cc` 文件的结构高度统一：
1. 用 `DEFINE_WEAK_FUNC` 定义所有需要打桩的接口。
2. 提供一个 `XxxDlInit(void* libHandle)` 初始化函数，内部对每个接口调用 `INIT_SUPPORT_FLAG`。

此外，部分 `*_dl.h` 文件中还包含编译时的类型桩定义（用 `#if CANN_VERSION_NUM` 守卫），同时承担编译时兼容和运行时兼容的职责。例如 `hccl_res_dl.h` 中的 `ThreadType`、`ThreadConfig` 桩定义，以及 `HcclDfxOpInfoCompat` 兼容结构体。

### 3.5 两种运行时探测方案的对比

在HCCL工程中存在**两种**运行时接口探测方案，分别用于不同场景：

#### （1）方案A：HcommIsSupportXxx + DEFINE_WEAK_FUNC（hcomm_dlsym目录）

这是主流方案，覆盖了绝大部分底层接口。其工作流程为：

1. **初始化时**：`HcommDlInit()` / `HcommDeviceDlInit()` 通过 `dlopen` 打开运行库，然后调用各 `XxxDlInit()` 函数，内部用 `dlsym` 逐个探测符号并设置 `g_<func>Supported` 标志。
2. **使用时**：上层代码先调用 `HcommIsSupportXxx()` 检查接口是否可用，如果可用则直接调用 `Xxx()`（此时弱桩已被真实实现覆盖），如果不可用则走降级路径。
3. **安全保障**：即使上层忘记检查就直接调用，弱桩也会兜底——打印错误日志并返回错误码，不会崩溃。

初始化的完整调用链如下，Host侧和Device侧结构对称：

```text
Host侧:
  libhccl.so被加载
    └─ __attribute__((constructor)) InitCompat()              compat.cc:20
         └─ pthread_once → CompatSymInit()
              └─ HcommDlInit()                                hcomm_dlsym.cc:62
                   ├─ dlopen("libhcomm.so", RTLD_NOW)         hcomm_dlsym.cc:65
                   └─ XxxDlInit(gLibHandle) × 11              各 *_dl.cc
                        └─ INIT_SUPPORT_FLAG(handle, func) × N
                             └─ dlsym(handle, "func")         dlsym_common.h:92
                                  └─ g_funcSupported = true/false

Device侧:
  libscatter_aicpu_kernel.so被加载
    └─ __attribute__((constructor)) InitCompat()              device_compat.cc:20
         └─ pthread_once → CompatSymInit()
              └─ HcommDeviceDlInit()                          hcomm_device_dlsym.cc:26
                   ├─ dlopen("libccl_kernel.so", RTLD_NOW)    hcomm_device_dlsym.cc:29
                   └─ XxxDlInit(gLibHandle) × 4               各 *_dl.cc
                        └─ INIT_SUPPORT_FLAG(handle, func) × N
                             └─ dlsym(handle, "func")         dlsym_common.h:92
                                  └─ g_funcSupported = true/false
```

`__attribute__((constructor))` 标注的函数在 `.so` 被动态链接器加载时自动调用，不需要在源码中显式调用。`pthread_once` 保证初始化只执行一次且线程安全。

典型示例（`src/ops/op_common/template/aicpu/kernel_launch.cc:394-396`）：

```cpp
if (HcommIsSupportHcommThreadResAcquireTimeOut()) {
    CHK_RET(HcclThreadResAcquireTimeOut(resCtxPtr->fullTimeout));
}
```

此方案的特点是**调用方可以直接调用接口函数本身**，因为弱桩保证了安全性。适合接口数量多、调用点分散的场景。

#### （2）方案B：DlHcommFunction单例 + std::function（dlhcomm_function.h/.cc）

这是辅助方案，仅用于少数Host侧接口（`HcclThreadResGetInfo`、`HcclConfigGetInfo`）。其实现位于 `src/ops/op_common/dlhcomm_function.h` 和 `dlhcomm_function.cc`。

`DlHcommFunction` 是一个C++ 单例类，内部持有 `void*` 的dlopen句柄和若干 `std::function` 成员：

```cpp
class DlHcommFunction {
public:
    static DlHcommFunction &GetInstance();
    HcclResult DlHcommFunctionInit();
    std::function<HcclResult(HcclComm, ThreadHandle, void*, uint32_t, void**)> dlHcclThreadResGetInfo{};
    std::function<HcclResult(HcclComm, HcclConfigType, uint32_t, void*)> dlHcclConfigGetInfo{};
private:
    void* handle_{nullptr};
    // ...
};
```

初始化时（`dlhcomm_function.cc:44-56`）`dlopen("libhcomm.so")`，然后用 `dlsym` 将函数指针存入 `std::function` 成员：

```cpp
dlHcclThreadResGetInfo = (HcclResult(*)(HcclComm, ThreadHandle, void*, uint32_t, void**))dlsym(handle_,
    "HcclThreadResGetInfo");
```

如果 `dlsym` 返回 `nullptr`，`std::function` 保持默认构造的空状态。使用时调用方通过检查 `std::function` 是否为空来判断可用性（`src/ops/op_common/op_common.cc:738-739`）：

```cpp
auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo || param.opMode == OpMode::OFFLOAD) {
    ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
} else {
    HcclResult ret1 = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*),
        &unfoldStream);
    // ...
}
```

方案B的初始化不依赖constructor，而是通过单例的**懒初始化**触发。完整调用链如下：

```text
业务代码首次调用DlHcommFunction::GetInstance()                op_common.cc:738 / 2107
  └─ static局部变量首次构造DlHcommFunction实例                dlhcomm_function.cc:16-17
       └─ DlHcommFunctionInit()                             dlhcomm_function.cc:44-56
            ├─ lock_guard(handleMutex_)                     互斥保护
            ├─ if (handle_ != nullptr) return               已初始化则跳过
            ├─ dlopen("libhcomm.so", RTLD_NOW)              dlhcomm_function.cc:51
            └─ DlHcommFunctionInterInit()                   dlhcomm_function.cc:35-42
                 ├─ dlsym(handle_, "HcclThreadResGetInfo")
                 │    → 存入dlHcclThreadResGetInfo (std::function)
                 └─ dlsym(handle_, "HcclConfigGetInfo")
                      → 存入dlHcclConfigGetInfo (std::function)
```

后续调用 `GetInstance()` 时，static局部变量已构造，直接返回已有实例，`DlHcommFunctionInit` 内部 `if (handle_ != nullptr)` 短路返回，不会重复初始化。方案B仅Host侧使用，自己独立 `dlopen("libhcomm.so")`，不走 `compat.cc` 的constructor路径。

#### （3）两者的区别

| 维度 | 方案A（DEFINE_WEAK_FUNC） | 方案B（DlHcommFunction） |
|------|--------------------------|-------------------------|
| 触发时机 | `.so` 加载时自动触发 | 业务代码首次调用 `GetInstance()` 时懒触发 |
| 触发入口 | `__attribute__((constructor))` in `compat.cc` / `device_compat.cc` | 无constructor，纯首次访问单例 |
| 初始化方式 | `HcommDlInit()` / `HcommDeviceDlInit()` 聚合调用各 `XxxDlInit` | `DlHcommFunctionInit()` 内部直接 `dlsym` |
| dlopen | `HcommDlInit` / `HcommDeviceDlInit` 中 `dlopen` | `DlHcommFunctionInit` 中独立 `dlopen` |
| dlsym结果存储 | 布尔标志 `g_xxxSupported`（true/false） | 原始函数指针存入 `std::function`（空/非空） |
| 语言风格 | C风格宏，`extern "C"` | C++ 类，`std::function` |
| 安全兜底 | 有弱桩，直接调用也安全 | 无兜底，必须先检查 `if (func)` |
| 探测方式 | `INIT_SUPPORT_FLAG` + `dlsym` 设置bool标志 | `dlsym` 直接存入 `std::function` |
| 查询方式 | `HcommIsSupportXxx()` | `if (obj.dlXxx)` |
| 调用方式 | 直接调用函数名 | 通过 `std::function` 间接调用 |
| 适用场景 | 接口多、调用点分散 | 接口少、调用点集中 |
| 代码位置 | `src/common/hcomm_dlsym/` | `src/ops/op_common/` |
| 编译进 | `libhccl_compat.so`（Host）/ `libhccl_kernel_compat.so`（Device） | `libhccl.so`（Host主库） |
| 归属层 | 兼容层基础设施 | 业务层自行管理 |
| 归属侧 | Host + Device各一套 | 仅Host |

方案A是统一的基础设施，方案B是业务层（`op_common`）针对少数几个特殊接口的轻量级补充。两种方案在实际代码中并存：大部分接口通过 `HcommIsSupportXxx` 探测，少数Host侧接口通过 `DlHcommFunction` 探测。

### 3.6 运行时版本探测：GetHcommVersion()

除了逐个接口的dlsym探测外，`hcomm_dlsym.cc:32-41` 还提供了运行时版本号查询：

```cpp
int GetHcommVersion(void) {
    if (gHcommVersion == 0) {
        char hcommPkgName[] = "hcomm";
        if (aclsysGetVersionNum(hcommPkgName, &gHcommVersion) != ACL_SUCCESS) {
            gHcommVersion = 0;
        }
    }
    return gHcommVersion;
}
```

它通过ACL运行时接口 `aclsysGetVersionNum` 查询hcomm包的版本号，返回值与 `CANN_VERSION()` 宏计算的格式一致。这提供了一种**运行时**的版本号判断能力，用于编译时宏无法覆盖的场景。

例如 `src/ops/all_reduce/all_reduce_op.cc:27-29` 在运行时检查版本：

```cpp
if (GetHcommVersion() < CANN_VERSION(9, 0, 0)) {
    return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
}
```

以及 `hcomm_dlsym.cc:43-58` 中的组合判断，将运行时版本号与dlsym探测结果结合：

```cpp
bool HcommIsExportThreadSupported() {
    if (GetHcommVersion() >= CANN_VERSION(9, 0, 0) && HcommIsSupportHcclThreadExportToCommEngine()) {
        return true;
    }
    return false;
}
```

这里同时要求版本号 >= 9.0.0（运行时探测）和dlsym找到符号，双保险地确认接口可用。

---

## 4. Host侧与Device侧的隔离

### 4.1 为什么需要隔离

HCCL产出的 `.so` 分为两类：

- **Host侧库**：`libhccl.so`（主库）+ `libhccl_compat.so`（兼容层）。运行在Host CPU上，通过 `libhcomm.so` 获取底层能力。
- **Device侧库**：`libscatter_aicpu_kernel.so`（算子kernel）+ `libhccl_kernel_compat.so`（兼容层）。运行在Device AICPU上，通过 `libccl_kernel.so` 获取底层能力。

两侧的底层运行库不同（`libhcomm.so` vs `libccl_kernel.so`），可用的API集合不同，dlopen的目标也不同。如果把Host侧的打桩代码编译进Device侧库，Device侧 `dlopen("libhcomm.so")` 会失败（Device上没有这个库），或者即使成功也拿不到正确的符号。反之亦然。

因此兼容层在CMake中被拆分为两个独立的共享库。

### 4.2 CMake中的拆分

`src/common/hcomm_dlsym/CMakeLists.txt` 定义了两个目标：

**`hccl_compat`（Host侧）**（第88-165 行），包含以下源文件：

```text
hccl_rank_graph_dl.cc       — 拓扑图查询
hccl_res_dl.cc              — 资源获取（含HcclThreadAcquireWithConfig）
hcomm_primitives_dl.cc      — 通信原语
hcomm_dlsym.cc              — 聚合初始化（dlopen libhcomm.so）
hccl_inner_dl.cc            — 内部接口
hcomm_host_profiling_dl.cc  — Host profiling
hccl_host_comm_dl.cc        — Host通信域管理
hccl_res_expt_dl.cc         — 实验性资源接口
ccu_res_dl.cc               — CCU资源
hccl_ccu_res_dl.cc          — CCU通信域查询
ccu_launch_dl.cc            — CCU kernel下发
ccu_primitives_impl_dl.cc   — CCU原语
```

初始化入口 `HcommDlInit()`（`hcomm_dlsym.cc:62-84`）依次调用11 个子模块的 `XxxDlInit()`，并 `dlopen("libhcomm.so")`。

**`hccl_kernel_compat`（Device侧）**（第169-175 行），包含以下源文件：

```text
hcomm_device_dlsym.cc         — 聚合初始化（dlopen libccl_kernel.so）
hcomm_primitives_dl.cc        — 通信原语（与Host侧共享同一源文件）
hcomm_diag_dl.cc              — 诊断接口
hcomm_device_profiling_dl.cc  — Device profiling
hccl_device_comm_dl.cc        — Device通信域管理
```

初始化入口 `HcommDeviceDlInit()`（`hcomm_device_dlsym.cc:26-41`）只调用4 个子模块的 `XxxDlInit()`，并 `dlopen("libccl_kernel.so")`。

注意 `hcomm_primitives_dl.cc` 是两侧共享的——通信原语（read/write/notify）在Host和Device上都需要使用，但两侧分别通过各自的运行库获取实现。

### 4.3 链接关系

**Host侧**（`src/CMakeLists.txt:568-589`）：

```text
libhccl.so  →  链接hccl_compat  →  运行时dlopen libhcomm.so
```

**Device侧**（`src/CMakeLists.txt:488-494`）：

```text
libscatter_aicpu_kernel.so  →  链接hccl_kernel_compat + ccl_kernel  →  运行时dlopen libccl_kernel.so
```

### 4.4 Host侧接口不能在Device侧判断的原因

以 `HcclThreadAcquireWithConfig` 为例：

1. `HcommIsSupportHcclThreadAcquireWithConfig` 和 `HcclThreadAcquireWithConfig` 的弱桩定义都在 `hccl_res_dl.cc` 中，通过 `DEFINE_WEAK_FUNC` 生成。
2. `hccl_res_dl.cc` 只被编译进 `hccl_compat`（Host侧），**不在** `hccl_kernel_compat`（Device侧）的源文件列表中。
3. Device侧的 `libscatter_aicpu_kernel.so` 只链接 `hccl_kernel_compat`，不链接 `hccl_compat`。
4. `HcclThreadAcquireWithConfig` 本身是弱符号，未定义时链接器解析为0，不报错。但 `HcommIsSupportHcclThreadAcquireWithConfig` 是 `extern "C"` 的**强符号**（由 `DEFINE_WEAK_FUNC` 生成，非weak），未定义时直接报undefined symbol。

这是设计上的刻意隔离：`HcclThreadAcquireWithConfig` 是Host侧资源获取接口，它的符号定义、初始化逻辑（`HcclResDlInit`）、dlsym探测目标（`libhcomm.so`）全部属于Host侧。Device侧代码不应引用这些符号，否则既无法链接通过，即使强行加入也无法正确初始化（Device侧的 `HcommDeviceDlInit` 不会调用 `HcclResDlInit`，标志永远为 `false`）。

如果Device侧代码确实需要知道Host侧是否走了某条代码路径，正确做法是通过已有的参数结构体（如 `param`、`resCtxPtr`）由Host侧传递标志位下来，而不是在Device侧直接探测Host侧接口。

---

## 5. 分层架构总览

```text
┌─────────────────────────────────────────────────────────────┐
│                      业务代码层                               │
│  src/ops/op_common/op_common.cc                             │
│  src/ops/all_reduce/all_reduce_op.cc                        │
│  src/ops/op_common/template/aicpu/kernel_launch.cc          │
│                                                             │
│  使用的兼容手段：                                             │
│  · #if CANN_VERSION_NUM >= CANN_VERSION(x,y,z)  (编译时)     │
│  · HcommIsSupportXxx()                          (运行时-A)   │
│  · DlHcommFunction::GetInstance().dlXxx         (运行时-B)   │
│  · GetHcommVersion()                            (运行时版本) │
│  · #ifdef MACRO_DEV_TYPE_NEW                    (编译时)     │
└────────────┬────────────────────────────┬───────────────────┘
             │ Host                       │ Device
┌────────────▼──────────┐  ┌──────────────▼──────────────────┐
│  hccl_compat (.so)    │  │  hccl_kernel_compat (.so)       │
│  Host侧兼容层         │  │  Device侧兼容层                 │
│                       │  │                                 │
│  hccl_res_dl.cc       │  │  hcomm_device_dlsym.cc          │
│  hccl_rank_graph_dl   │  │  hcomm_primitives_dl.cc (共享)  │
│  hcomm_primitives_dl  │  │  hcomm_diag_dl.cc               │
│  hcomm_dlsym.cc       │  │  hcomm_device_profiling_dl.cc   │
│  hccl_inner_dl        │  │  hccl_device_comm_dl.cc         │
│  hccl_host_comm_dl    │  │                                 │
│  hcomm_host_profiling │  │  初始化: HcommDeviceDlInit()    │
│  hccl_res_expt_dl     │  │  dlopen("libccl_kernel.so")    │
│  ccu_res_dl           │  │  调用4 个子模块DlInit         │
│  hccl_ccu_res_dl      │  │                                 │
│  ccu_launch_dl        │  │  触发: device_compat.cc         │
│  ccu_primitives_impl  │  │  __attribute__((constructor))   │
│                       │  │                                 │
│  初始化: HcommDlInit()│  │                                 │
│  dlopen("libhcomm.so")│  │                                 │
│  调用11 个子模块DlInit│  │                                 │
│                       │  │                                 │
│  触发: compat.cc      │  │                                 │
│  __attribute__((constructor))│  │                          │
└──────────┬────────────┘  └──────────────┬──────────────────┘
           │                              │
┌──────────▼────────────┐  ┌──────────────▼──────────────────┐
│  libhcomm.so          │  │  libccl_kernel.so               │
│  Host侧运行库         │  │  Device侧运行库                 │
│  (由CANN安装提供)    │  │  (由CANN安装提供)              │
└───────────────────────┘  └─────────────────────────────────┘
```

---

## 6. 新增兼容接口的步骤

当需要在新代码中引用一个可能不存在于所有CANN版本的底层接口时，应按以下步骤操作：

### 6.1 判断归属侧

首先确定该接口属于Host侧还是Device侧：

- 如果接口由 `libhcomm.so` 提供、在Host CPU上调用 → Host侧。
- 如果接口由 `libccl_kernel.so` 提供、在AICPU上调用 → Device侧。

### 6.2 添加打桩定义

根据是复用已有文件还是新建文件，所需步骤不同：

**情况一：在已有的 `*_dl.h` / `*_dl.cc` 文件中添加接口**

只需完成以下两步：

1. **头文件中**：用 `DECL_WEAK_FUNC` 声明弱符号，用 `DECL_SUPPORT_FLAG` 声明查询函数。如果旧版SDK头文件缺少该接口的类型定义，用 `#if CANN_VERSION_NUM < CANN_VERSION(...)` 补桩。
2. **源文件中**：用 `DEFINE_WEAK_FUNC` 定义弱桩和查询函数，在已有的 `XxxDlInit` 函数体中添加 `INIT_SUPPORT_FLAG`。

此情况下，该文件的 `XxxDlInit` 已被聚合初始化函数（`HcommDlInit` 或 `HcommDeviceDlInit`）调用，新加的 `INIT_SUPPORT_FLAG` 会随之自动执行，无需改动其他文件。

**情况二：新建 `*_dl.h` / `*_dl.cc` 文件**

需要完成全部四步：

1. **头文件中**：用 `DECL_WEAK_FUNC` 声明弱符号，用 `DECL_SUPPORT_FLAG` 声明查询函数。如果旧版SDK头文件缺少该接口的类型定义，用 `#if CANN_VERSION_NUM < CANN_VERSION(...)` 补桩。
2. **源文件中**：用 `DEFINE_WEAK_FUNC` 定义弱桩和查询函数，新建 `XxxDlInit` 函数，内部调用 `INIT_SUPPORT_FLAG`。
3. **聚合初始化中**：在 `hcomm_dlsym.cc`（Host）的 `HcommDlInit()` 或 `hcomm_device_dlsym.cc`（Device）的 `HcommDeviceDlInit()` 中添加对新 `XxxDlInit` 的调用。否则新接口永远不会被 `dlsym` 探测，标志永远是 `false`。
4. **CMake中**：将新源文件加入 `hccl_compat`（Host）或 `hccl_kernel_compat`（Device）的 `target_sources`。

### 6.3 在业务代码中使用

优先使用方案A（`HcommIsSupportXxx`）：

```cpp
if (HcommIsSupportXxx()) {
    Xxx(args...);
} else {
    // 降级路径
}
```

如果接口仅Host侧少量调用点使用，也可使用方案B（`DlHcommFunction`），在 `dlhcomm_function.h` 中添加 `std::function` 成员并在 `dlhcomm_function.cc` 中 `dlsym` 初始化。

### 6.4 常见陷阱

- **不要在Device侧代码中调用Host侧的 `HcommIsSupportXxx`**：对应的强符号定义不在 `hccl_kernel_compat` 中，会导致undefined symbol。
- **不要遗漏 `INIT_SUPPORT_FLAG`**：如果只定义了 `DEFINE_WEAK_FUNC` 但没有在初始化函数中调用 `INIT_SUPPORT_FLAG`，标志永远为 `false`，接口即使存在也会被认为不支持。
- **不要在Device侧的 `HcommDeviceDlInit` 中调用Host侧的 `XxxDlInit`**：Device侧 `dlopen` 的是 `libccl_kernel.so`，不是 `libhcomm.so`，探测到的符号集合不同。
- **注意weak符号的链接行为**：`DEFINE_WEAK_FUNC` 生成的接口函数本身是弱符号，如果运行库提供了真实实现，链接器会自动覆盖弱桩。但 `HcommIsSupportXxx` 是强符号，必须确保它被编译进了同一个 `.so`。
- **注意处理与接口可用性关联的周边逻辑**：接口的可用性不仅影响其自身的调用与降级路径，还可能影响其他代码分支的走向。例如，当某接口可用时，相关的参数预处理、资源分配、路径选择等逻辑会走一条分支；不可用时则走另一条分支。处理兼容性时应全面排查所有依赖于该接口可用性的代码路径，确保降级分支与正常分支的周边逻辑保持一致，避免遗漏导致行为异常。
- **弱符号方案仅适用于具有外部链接的函数**：`DEFINE_WEAK_FUNC` 方案依赖函数在共享库中产生动态符号、可被 `dlsym` 探测且可被强符号覆盖。若目标函数在SDK头文件中以 `static inline` 或 `inline` 定义，则具有内部链接，不产生动态符号，弱符号机制完全失效。此类函数只能通过编译时 `#if CANN_VERSION_NUM` 桩定义来处理（参见2.2 节方式四）。

---

## 7. 关键文件索引

| 文件 | 作用 |
|------|------|
| `src/common/hcomm_dlsym/dlsym_common.h` | 核心宏定义（DEFINE_WEAK_FUNC等）+ CANN版本号宏 + 编译时类型桩 |
| `src/common/hcomm_dlsym/hcomm_dlsym.cc` | Host侧聚合初始化，dlopen libhcomm.so，GetHcommVersion() |
| `src/common/hcomm_dlsym/hcomm_device_dlsym.cc` | Device侧聚合初始化，dlopen libccl_kernel.so |
| `src/common/hcomm_dlsym/hccl_res_dl.cc/h` | 资源获取接口打桩（HcclThreadAcquireWithConfig等） |
| `src/common/hcomm_dlsym/hcomm_primitives_dl.cc/h` | 通信原语打桩（Host/Device共享） |
| `src/common/hcomm_dlsym/hccl_host_comm_dl.cc/h` | Host通信域管理打桩（HcclConfigGetInfo等） |
| `src/common/hcomm_dlsym/hccl_device_comm_dl.cc/h` | Device通信域管理打桩 |
| `src/common/hcomm_dlsym/hcomm_host_profiling_dl.cc/h` | Host profiling打桩 |
| `src/common/hcomm_dlsym/hcomm_device_profiling_dl.cc/h` | Device profiling打桩 |
| `src/common/hcomm_dlsym/hccl_rank_graph_dl.cc/h` | 拓扑图查询打桩 |
| `src/common/hcomm_dlsym/hccl_inner_dl.cc/h` | 内部接口打桩 |
| `src/common/hcomm_dlsym/hccl_res_expt_dl.cc/h` | 实验性资源接口打桩 |
| `src/common/hcomm_dlsym/ccu_*.cc/h` | CCU相关接口打桩 |
| `src/common/hcomm_dlsym/hcomm_diag_dl.cc/h` | 诊断接口打桩 |
| `src/common/hcomm_dlsym/CMakeLists.txt` | hccl_compat和hccl_kernel_compat的构建定义 |
| `src/common/compat.cc` | Host侧constructor自动初始化 |
| `src/common/device_compat.cc` | Device侧constructor自动初始化 |
| `src/ops/op_common/dlhcomm_function.h/cc` | DlHcommFunction单例（方案B） |
| `src/CMakeLists.txt:14-72` | CANN_VERSION_NUM解析与注入逻辑 |
| `src/ops/op_common/op_common.cc` | 业务层兼容性使用范例 |
| `src/ops/op_common/template/aicpu/kernel_launch.cc` | Device侧业务代码兼容性使用范例 |

# RFC：hccl4py — HCCL 通信算子 Python 绑定

- 起始日期：2026-10-14
- RFC PR编号：[cann/hccl#3798](https://gitcode.com/cann/hccl/merge_requests/3798)
- 相关Issue：社区任务「实现 hccl4py」；讨论帖 https://gitcode.com/cann/hccl/discussions/19

---

## 概要

在 `experimental/eco_system/python` 以 ctypes 封装 HCCL 侧 14 个集合/点对点 C API，形成 Python 包 `hccl4py` 的算子层，语义与 C 接口一致，额外调用开销目标 ≤ 10µs。对外边界见同目录 `PUBLIC_API.md`，可通过 `setup.py` 构建 whl。

## 背景与动机

- HCCL 当前仅提供 C 头文件 + 动态库；Python 用户需手写 ctypes，易错且难维护。
- 社区任务要求将对接 AI 框架层的通信算子 C API 封装为 Python，功能与 C 保持一致。
- 预期场景：自研训练框架、HPC 脚本、模型调优工具在 Python 侧直接调用集合/点对点通信。

## 术语表

| 术语 | 含义 |
|------|------|
| hccl4py | 本仓 Python 包名（算子；域 API 依赖 hcomm4py） |
| hcomm4py | 姊妹仓域名权威包 |
| 对外边界 | `PUBLIC_API.md` 与 `__all__` 共同约束的稳定 API 面 |

## 架构与接口契约

### 整体架构

```text
experimental/eco_system/python/
├── PUBLIC_API.md / PUBLIC_API_en.md   # 对外边界清单（管控）
├── setup.py                           # whl 打包
├── hccl4py/
│   ├── __init__.py      # 导出算子、类型与自 hcomm4py 重导出的域 API
│   ├── _hcomm_dep.py    # 硬依赖 hcomm4py（无本地 comm.py）
│   ├── _lib.py          # 加载 libhccl.so / libhcomm.so / libascendcl.so
│   ├── _validate.py     # 算子侧入参校验（与 hcomm4py 语义对齐）
│   ├── _bindings.py     # 14 算子 ctypes argtypes/restype 表
│   ├── _ops_util.py     # 句柄/指针归一化等算子辅助
│   ├── types.py         # 自 hcomm4py.types 重导出
│   └── ops.py           # 14 个通信算子封装（本 RFC 范围）
├── examples/
└── tests/
```

本仓**不保留** `hccl4py/comm.py`，也**没有** in-tree 域兜底；通信域 API 完全由姊妹仓 `cann/hcomm` 的 `hcomm4py` 提供，经 `hccl4py` 顶层导入。算子与域须共用同一次 `libhcomm`/`libhccl` 加载（跨仓混用两套 CDLL 句柄会挂死集合通信）。

数据流：Python 调用 → 入参格式校验 → ctypes 参数转换（一次性 argtypes 绑定）→ `libhccl.so` 原 C 入口 → NPU 执行。

本仓负责 **§3.1.2 通信算子 14 个**；通信域 5 个接口见姊妹仓 PR [cann/hcomm#6331](https://gitcode.com/cann/hcomm/merge_requests/6331) 中的 RFC 文档（合入 master 前请以该 PR 为准）。

### 对外接口

稳定符号以 `hccl4py.__all__` / `PUBLIC_API.md` 为准，包括类型枚举、自 `hcomm4py` 导入的域 API、14 个算子及 `make_send_recv_item`。

约定：

- `comm` / `stream` 接受不透明句柄（`int`、ctypes 指针或 `Comm`）。
- `send`/`recv` 为 device 指针（`int` 地址或 `c_void_p`）。
- 返回值为 `HcclResult` 整型；非 0 时抛出带 API 名的异常（`check`）。
- 调用前校验 count/rank/dtype/reduce_op 等格式，非法入参在进入 native 前失败。

### 依赖接口

依赖运行时 `libhcomm.so` / `libhccl.so` / `libascendcl.so`。库搜索顺序与 `hccl4py/_lib.py` 一致：`HCCL4PY_LIB_DIR` / `HCCL4PY_HCCL_LIB` / `HCCL4PY_HCOMM_LIB` / `HCCL4PY_ASCENDCL_LIB`，再 `ASCEND_LIBRARY_PATH` / `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME`（及 `lib64`），以及 `LD_LIBRARY_PATH`。

## 影响分析

- 性能：绑定层相对同进程 ctypes 直调额外开销目标 ≤ 10µs。
- 功能范围：主体为 `experimental/` 纯新增；本 PR 另含一处 `src/` 拓扑 UBG 门控宏修复（配合 CANN 9.2 beta 的 `CANN_VERSION(9,2,0,0)` 窗口），以便 Ascend 950PR 验收，不改变既有 C API 语义。
- 构建发布：新增可选 whl（`python -m build`），不强制改主仓 CMake 产物。

## 兼容性考虑

- 纯新增路径，向后兼容。
- 不引入特性开关；依赖已安装的 CANN 动态库。
- `HcclCommConfig` 布局由姊妹仓 `hcomm4py` 权威维护：运行时按 `HcclGetCommConfigCapability` 选型，离线回退 v11/2248；本仓 `types.py` 仅重导出。

## 详细设计

### 类型与结构体

- `HcclSendRecvItem` 与 C 布局对齐；枚举取值与 `hccl_types.h` 一致（经 `hcomm4py.types` 重导出）。
- `HcclCommConfig` 使用可扩展 raw buffer + `_OFF` 偏移表，支持 v11/v12 字段门控（实现位于 hcomm 仓）。

### 加载与绑定

1. `_lib.get_hccl()` 先确保 `libhcomm` 已加载，再加载 `libhccl`。
2. ctypes 原型表落在 `_bindings.py`；`ops._bind()` 一次性调用完成 `argtypes`/`restype` 绑定，`ops.py` 仅保留语义封装（控制单文件体量）。
3. 不改变 HCCL 内部算法选择与执行路径；绑定层不做额外拷贝。

### 打包

```bash
cd experimental/eco_system/python
python3 -m build  # via setup.py
pip install dist/hccl4py-*.whl
```

## 测试方案

| 项 | 方法 |
|----|------|
| 离线 | `tests/test_offline_bindings.py`：Config 门控、入参校验、`_as_comm` |
| 功能 | `tests/run_ops_suite.py` 等：单机 ≥2 device |
| 性能 | `tests/run_perf_bench.py`：同进程 ctypes 直调对照的 **host-enqueue** wrapper 开销 ≤10µs（sync 在计时窗外） |
| 验收环境 | A3 或 A5，CANN ≥ 9.2.0，≥2 卡 |

## 风险评估

| 风险 | 应对 |
|------|------|
| ctypes 结构体对齐错误 | 以运行环境 sizeof 为准 |
| CANN 版本字段扩展 | config 缓冲预留；version/size 与 toolkit 对齐 |
| 跨包双 CDLL 句柄不互通 | 文档约束联用方式；验收走同一进程内 hcomm4py + hccl4py |

## 替代方案

| 方案 | 优点 | 缺点 |
|------|------|------|
| **ctypes（采用）** | 无编译依赖、易随 so 升级 | 需手工维护布局 |
| pybind11 / cffi | 类型更安全 | 需编译链路 |
| 仅文档示例不入库 | 无合入成本 | 不满足任务交付 |

## 开放问题

1. 正式验收机资源申请与长期维护归属。
2. 合入后 examples 是否按算子拆文件。

---

## 评审记录

评审过程在 PR 评论区进行，详见 cann/hccl#3798。

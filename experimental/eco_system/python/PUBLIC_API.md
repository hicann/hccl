# hccl4py 对外 API 边界

简体中文 | [English](./PUBLIC_API_en.md)

本文件是 **hccl4py 对外边界清单**，供合入后管控与评审对照。  
**仅 `__init__.py` 的 `__all__` 中符号为稳定对外 API**；以下划线开头的模块/符号（如 `_lib`、`ops._as_*`）为内部实现，不纳入对外契约。

## 边界文件映射

| 职责 | 对应文件 | 说明 |
|---|---|---|
| 对外导出入口 | `hccl4py/__init__.py` | 唯一稳定导入面（`__all__`） |
| 类型 / 枚举 / 配置布局 | `hccl4py/types.py` | **re-export** `hcomm4py.types`（无本地副本） |
| 通信域 API | （无本地 `comm.py`） | 顶层从 `hcomm4py.comm` 导入（依赖姊妹仓） |
| 集合 / 点对点算子 | `hccl4py/ops.py` | 任务书 §3.1.2，14 个算子 |
| 动态库加载 | `hccl4py/_lib.py` | **内部**，非对外 API；`get_hcomm` 优先复用 `hcomm4py` 已加载的 CDLL |
| 打包入口 | `setup.py` | whl 构建入口（`python -m build` / `pip wheel .`） |

## 对外符号 ↔ C API

### 类型

| Python | C / 头文件概念 |
|---|---|
| `HcclResult` / `HcclError` | `HcclResult` |
| `HcclDataType` | `HcclDataType` |
| `HcclReduceOp` | `HcclReduceOp` |
| `HcclSendRecvType` | `HcclSendRecvType` |
| `HcclRootInfo` | `HcclRootInfo` |
| `HcclCommConfig` | `HcclCommConfig` |
| `HcclSendRecvItem` | `HcclSendRecvItem` |
| `HCCL_ROOT_INFO_BYTES` | `HCCL_ROOT_INFO_BYTES`（4108） |
| `HCCL_COMM_CONFIG_BYTES` | Python raw buffer 容量（2304） |
| `HCCL_COMM_CONFIG_MAGIC_WORD` | Config magic `0xF0F0F0F0` |
| `HCCL_COMM_CONFIG_SIZE_V10` / `V11` / `V12` | 逻辑 size 门限（2240 / 2248 / 2248） |
| `HCCL_COMM_CONFIG_VERSION_DEFAULT` | 离线回退 version（11） |
| `HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT` | 离线回退 logical_size（= V11） |

### 通信域（依赖 hcomm4py）

通信域实现位于姊妹仓 `cann/hcomm` 的 `hcomm4py`。本仓不保留 `hccl4py/comm.py`；`hccl4py` 顶层与 `hccl4py.types` 从 `hcomm4py` 导入，不维护第二份域实现。  
域符号门控：这些符号走 V2/op_base 导出路径，以 hcomm 仓 `PUBLIC_API.md` 为准；与 toolkit 小版本号（如 9.1.0）无关。详细说明见该文档对 V2/op_base 的描述。

| Python | C API |
|---|---|
| `get_root_info` | `HcclGetRootInfo` |
| `comm_config_init` | `HcclCommConfigInit`（头文件 inline，Python 重实现于 hcomm4py） |
| `comm_init_root_info_config` / `init_rank_config` | `HcclCommInitRootInfoConfig` |
| `comm_init_cluster_info_config` | `HcclCommInitClusterInfoConfig` |
| `comm_destroy` | `HcclCommDestroy` |
| `Comm` | 句柄上下文管理器（Python 侧） |

### 算子

| Python | C API |
|---|---|
| `all_reduce` | `HcclAllReduce` |
| `reduce` | `HcclReduce` |
| `broadcast` | `HcclBroadcast` |
| `all_gather` | `HcclAllGather` |
| `all_gather_v` | `HcclAllGatherV` |
| `scatter` | `HcclScatter` |
| `reduce_scatter` | `HcclReduceScatter` |
| `reduce_scatter_v` | `HcclReduceScatterV` |
| `all_to_all` | `HcclAlltoAll` |
| `all_to_all_v` | `HcclAlltoAllV` |
| `all_to_all_vc` | `HcclAlltoAllVC` |
| `send` / `recv` | `HcclSend` / `HcclRecv` |
| `batch_send_recv` / `make_send_recv_item` | `HcclBatchSendRecv` |

## 变更管控约定

1. 新增/删除/重命名对外符号：必须同步更新本文件与 `__all__`。
2. 仅内部重构（`_lib` / 私有 helper）：可不改本文件，但不得泄漏到 `__all__`。
3. 跨仓联用：域名权威包为 hcomm 仓 `hcomm4py`；本仓顶层导入依赖之，勿再复制域实现。

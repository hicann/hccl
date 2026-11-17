# hccl4py Public API Boundary

English | [简体中文](./PUBLIC_API.md)

This file is the **hccl4py external API boundary checklist** for post-merge governance.  
**Only symbols listed in `hccl4py/__init__.py` `__all__` are stable public APIs.** Underscore-prefixed modules/symbols (e.g. `_lib`, `ops._as_*`) are internal and not part of the contract.

## Boundary File Mapping

| Role | File | Notes |
|---|---|---|
| Public export entry | `hccl4py/__init__.py` | Sole stable import surface (`__all__`) |
| Types / enums / config layout | `hccl4py/types.py` | **Re-exports** `hcomm4py.types` (no local copy) |
| Domain APIs | (no local `comm.py`) | Top-level imports from `hcomm4py.comm` (sister repo) |
| Collective / P2P ops | `hccl4py/ops.py` | Task §3.1.2, 14 ops |
| Dynamic library loading | `hccl4py/_lib.py` | **Internal**; `get_hcomm` prefers the CDLL already loaded by `hcomm4py` |
| Packaging entry | `setup.py` | Wheel build entry (`python -m build` / `pip wheel .`) |

## Public Symbols ↔ C API

### Types

| Python | C / header concept |
|---|---|
| `HcclResult` / `HcclError` | `HcclResult` |
| `HcclDataType` | `HcclDataType` |
| `HcclReduceOp` | `HcclReduceOp` |
| `HcclSendRecvType` | `HcclSendRecvType` |
| `HcclRootInfo` | `HcclRootInfo` |
| `HcclCommConfig` | `HcclCommConfig` |
| `HcclSendRecvItem` | `HcclSendRecvItem` |
| `HCCL_ROOT_INFO_BYTES` | `HCCL_ROOT_INFO_BYTES` (4108) |
| `HCCL_COMM_CONFIG_BYTES` | Python raw buffer capacity (2304) |
| `HCCL_COMM_CONFIG_MAGIC_WORD` | Config magic `0xF0F0F0F0` |
| `HCCL_COMM_CONFIG_SIZE_V10` / `V11` / `V12` | Logical size gates (2240 / 2248 / 2248) |
| `HCCL_COMM_CONFIG_VERSION_DEFAULT` | Offline fallback version (11) |
| `HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT` | Offline fallback logical_size (= V11) |

### Domain (via hcomm4py)

Domain implementation lives in sister repo `cann/hcomm` as `hcomm4py`. This package has no `hccl4py/comm.py`; top-level `hccl4py` and `hccl4py.types` import from `hcomm4py` and do not keep a second domain copy.  
Domain symbol gating: these symbols come from the V2/op_base export path; see hcomm `PUBLIC_API.md`. This is unrelated to toolkit minor versions such as 9.1.0.

| Python | C API |
|---|---|
| `get_root_info` | `HcclGetRootInfo` |
| `comm_config_init` | `HcclCommConfigInit` (header inline; Python reimplementation in hcomm4py) |
| `comm_init_root_info_config` / `init_rank_config` | `HcclCommInitRootInfoConfig` |
| `comm_init_cluster_info_config` | `HcclCommInitClusterInfoConfig` |
| `comm_destroy` | `HcclCommDestroy` |
| `Comm` | Handle context manager (Python-side) |

### Ops

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

## Change Control

1. Add/remove/rename public symbols: update this file and `__all__` together.
2. Internal-only refactors (`_lib` / private helpers): this file may stay unchanged; do not leak into `__all__`.
3. Cross-repo use: canonical domain package is hcomm `hcomm4py`; this package depends on it via top-level import — do not copy the domain implementation again.

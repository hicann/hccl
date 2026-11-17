# eco_system

简体中文 | [English](./README_en.md)

本目录存放生态相关实验代码。`python/` 子目录提供 **hccl4py**：HCCL 通信算子的 Python 绑定（任务书 §3.1.2，14 个算子）。

## hccl4py 覆盖算子

`all_reduce` / `reduce` / `broadcast` / `all_gather` / `all_gather_v` / `scatter` /
`reduce_scatter` / `reduce_scatter_v` / `all_to_all` / `all_to_all_v` / `all_to_all_vc` /
`send` / `recv` / `batch_send_recv`

通信域接口见姊妹仓 `cann/hcomm` 的 `hcomm4py`（任务 Issue：
https://gitcode.com/cann/hcomm/issues/1157）。`hccl4py` **直接依赖** `hcomm4py`：
`hccl4py` 顶层与 `hccl4py.types` 从 `hcomm4py` 导入域 API/类型，本仓不保留 `comm.py` 副本。联用时把两仓
`experimental/eco_system/python` 都放进 `PYTHONPATH`（或先安装 `hcomm4py`）。

## 目录

```text
experimental/eco_system/python/
├── PUBLIC_API.md / PUBLIC_API_en.md  # 对外 API 边界（管控）
├── setup.py                    # whl 打包
├── hccl4py/           # 算子；域 API 顶层导入自 hcomm4py
├── examples/01_basic/ # AllReduce / Broadcast / AllGather / SendRecv
└── tests/             # ops / dtype / reduce_op / boundary / cluster / perf
```

方案文档：[`docs/zh/rfcs/0004-hccl4py-python-bindings.md`](../../docs/zh/rfcs/0004-hccl4py-python-bindings.md)

## 用法

```bash
source /usr/local/Ascend/cann/set_env.sh   # CANN >= 9.2.0
# 需同时能 import hcomm4py（姊妹仓路径或已 pip install）
export PYTHONPATH="/path/to/hcomm/experimental/eco_system/python:/path/to/hccl/experimental/eco_system/python:${PYTHONPATH}"
cd /path/to/hccl/experimental/eco_system/python
# 可选：构建并安装 whl
# python3 -m build  # via setup.py && pip install dist/hccl4py-*.whl
python3 examples/01_basic/01_allreduce.py --nranks 2
```

### 测试入口（tests/）

| 脚本 | 说明 |
|---|---|
| `run_npu_smoke.py` | 库加载 + RootInfo 冒烟 |
| `run_ops_suite.py` | 14 算子功能正确性 |
| `run_dtype_suite.py` | 多 dtype |
| `run_reduce_op_suite.py` | reduce op |
| `run_boundary_suite.py` | 边界 |
| `run_cluster_suite.py` | 多卡集群（需 ranktable） |
| `run_comm_domain_suite.py` | 通信域 5 API |
| `run_ops_extra_suite.py` | 多 item batch / 变长 *V / 多 rank |
| `run_stress_suite.py` | 大 count 短循环（夜间） |
| `run_perf_bench.py` | Python wrapper 相对 ctypes 直调的额外开销（≤10µs） |
| `c_allreduce_bench.c` | 可选原生 C 对照（需自行编译；验收主路径见下） |

集群 / 性能示例（套件自建 ranktable；默认 2 卡，无 CLI 规模参数）：

```bash
export HCCL_IF_IP=<本机 IP>
# optional: export HCCL4PY_CHIP=a5
python3 tests/run_cluster_suite.py
python3 tests/run_perf_bench.py
# perf knobs: HCCL4PY_PERF_WARMUP/ITERS/COUNT/ROUNDS/LIMIT_US/JITTER（见 tests/README.md）
```

## 验收口径

| 项 | 要求 | 私仓 A5 结果 |
|---|---|---|
| 硬件 | A3 **或** A5，≥2 卡 | Ascend 950PR ×2 |
| CANN | ≥ 9.2.0（含 9.2.0-beta.x） | 9.2.0-beta.2 |
| 功能 | 14 算子结果正确 | ops：14 个 C API / 13 个套件条目（send/recv 合并）全部 PASS（含 A5 `all_gather_v`/`reduce_scatter_v`，无 soft-SKIP）；dtype：仅官方不支持 dtype 预跳（静态预跳见 `_platform.dtype_skip_cases`） |
| 性能 | Py wrapper 相对同进程 ctypes 直调 `lib.Hccl*` **主机下发**额外开销 ≤10µs（±10%） | host-enqueue 窗口实测 ≤11µs（含 ±10% 容差） |

说明：`run_perf_bench.py` 的 “C 路径”是同一 Python 进程内用 ctypes 直接调 `lib.Hccl*`（不含 hccl4py wrapper 函数体）。计时窗口为 warmup+预 sync 之后的主机提交循环，`aclrtSynchronizeStream` 放在收表之后，避免设备执行为瓶颈时把 wrapper 开销测成约 0。

Issue：https://gitcode.com/cann/hccl/issues/875

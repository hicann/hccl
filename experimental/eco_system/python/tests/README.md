# hccl4py 测试套件

在 `experimental/eco_system/python` 下执行；需能 `import hcomm4py`，并先 `source` CANN 环境。

## 脚本

| 脚本 | 需要 NPU | 用途 |
|---|---|---|
| `run_npu_smoke.py` | ≥1 | 库加载 + RootInfo 冒烟 |
| `run_ops_suite.py` | ≥2 | 14 个 C API / 13 条目（send/recv 合并）功能正确性 |
| `run_dtype_suite.py` | ≥2 | op × dtype 矩阵（仅官方不支持 dtype 预跳） |
| `run_reduce_op_suite.py` | ≥2 | reduce op 变体 |
| `run_boundary_suite.py` | 空缓冲 MP 需 ≥2 | CPU 配置布局 + 空指针/非法 count 边界 |
| `run_cluster_suite.py` | ≥2 | 自写 ranktable 多卡 |
| `run_comm_domain_suite.py` | ≥1（`<2` 时多卡用例自跳） | 通信域 5 API |
| `run_perf_bench.py` | ≥2 | wrapper 相对同进程 ctypes 主机下发开销 |
| `run_stress_suite.py` | ≥2 | 大 count + 短循环 AllReduce（夜间） |
| `run_ops_extra_suite.py` | ≥2（4-rank 需 device_count≥4） | 多 item batch、变长 *V、可选多 rank AllReduce |
| `probe_platform.py` | 可选 | npu-smi / CANN install.info 诊断 |
| `_platform.py` | — | 芯片探测与 ranktable 辅助 |

可选原生对照：本地编译 `c_allreduce_bench.c`（产物已在 `.gitignore`）。

## 环境变量

| 变量 | 作用 |
|---|---|
| `HCCL4PY_CHIP` | `a2` / `a3` / `a5` / `auto`（默认） |
| `HCCL4PY_ASSUME_CHIP` | 自动探测无产品 tag 时的兜底 |
| `HCCL_IF_IP` | HCCL 运行时网卡选择（不写入 ranktable；`host_ip` 来自 `hostname -I` / UDP 探测） |
| `HCCL4PY_PERF_WARMUP` / `ITERS` / `COUNT` / `ROUNDS` / `LIMIT_US` / `JITTER` | 性能旋钮 |
| `HCCL4PY_NRANKS` | ops / extra 套件 rank 数（默认 2；extra 在卡数≥4 时可跑 4-rank AllReduce） |
| `HCCL4PY_STRESS_COUNT` / `HCCL4PY_STRESS_REPEAT` | stress 规模 / 循环次数 |

套件需要时会自建 ranktable。验收默认 2 卡；可用 `HCCL4PY_NRANKS` 或 `run_ops_extra_suite.py` 覆盖多 rank / 变长 *V / 多 item batch。

## 验收口径

- ops suite：14 个 C API（13 条目，send/recv 合并）全部 PASS（含 A5 `all_gather_v` / `reduce_scatter_v`，无 soft-SKIP）
- dtype suite（A5）：仅官方不支持 dtype 预跳（`_platform.dtype_skip_cases("a5")`）
- cluster suite：ClusterInfo 集合通信必跑（含 A5）
- perf：相对同进程 `lib.Hccl*` ctypes 的主机下发额外开销 ≤10µs（`aclrtSynchronizeStream` 在计时窗外）
- 通信域：`hccl4py` 从 `hcomm4py` 导入域 API（无本地 `comm.py`）；两仓 `python` 树均需在 `PYTHONPATH`

## 覆盖矩阵（验收 / 夜间）

| 套件 | API | dtype | count | ranks | 正/负向 | 备注 |
|---|---|---|---|---|---|---|
| `run_ops_suite` | 14 | FP32 | 8（V=1） | `HCCL4PY_NRANKS`（默认 2） | 正向 | A5 *V 必须 PASS |
| `run_ops_extra_suite` | batch×4 / 变长 *V / AllReduce | FP32 | 小 | 2 或 4 | 多 item + 变长 counts | A5 *V 必须 PASS |
| `run_dtype_suite` | 14×12 | 矩阵 | 4 | 2 | 正向 + 官方 dtype 跳过 | 见 `_platform.dtype_skip_cases` |
| `run_reduce_op_suite` | reduce 族 | FP32 | 小 | 2 | SUM/PROD/MAX/MIN | A5 跑 ReduceScatterV |
| `run_boundary_suite` | 抽样 | FP32 | 边界 | 2 | 空指针 / 非法 count / 非法 rank | 离线 + MP |
| `run_cluster_suite` | AllReduce 路径 | FP32 | 8 | 2 | ClusterInfo | A5：建域+集合；不强制 ccu_sched |
| `run_comm_domain_suite` | 5 域 API | — | — | 1 或 2 | RootInfo / ClusterInfo 建域+销毁 | 单卡跑 test_1/2；`<2` 时多卡用例自跳；ClusterInfo 用默认 cfg |
| `run_perf_bench` | 14 | FP32 | 旋钮 | 2 | host enqueue vs ctypes | sync 在计时窗外 |
| offline UT | types/ops/comm | — | — | 0 | float/bool 拒绝、偏移金表、Comm.destroy 幂等 | 需 `hcomm4py` 在 path |

## ranktable

不单独维护静态 fixture。验收套件统一由 `_platform.write_ranktable_*` 运行时生成（`host_ip` 来自 `hostname -I` / UDP 探测；A3 单机可省略 `device_ip`，多机从 `/etc/hccn.conf` 填充）。Schema 以生成器与官方 `rank_table_config_*.md` 为准。

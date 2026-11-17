# hccl4py test suites

Run from `experimental/eco_system/python` after `hcomm4py` is importable and CANN env is sourced.

## Scripts

| Script | Needs NPU | Purpose |
|---|---|---|
| `run_npu_smoke.py` | ≥1 | Load libs + RootInfo smoke |
| `run_ops_suite.py` | ≥2 | 14 C APIs / 13 entries (send/recv merged) correctness |
| `run_dtype_suite.py` | ≥2 | op × dtype matrix (official unsupported dtypes only are pre-skipped) |
| `run_reduce_op_suite.py` | ≥2 | reduce op variants |
| `run_boundary_suite.py` | ≥2 for null-buffer MP tests | CPU config layout + null/bad-count boundaries |
| `run_cluster_suite.py` | ≥2 | Multi-card via self-written ranktable |
| `run_comm_domain_suite.py` | ≥1 (multi-rank cases self-skip when device_count<2) | Comm-domain APIs |
| `run_perf_bench.py` | ≥2 | Wrapper vs same-process ctypes host-enqueue overhead |
| `run_stress_suite.py` | ≥2 | Large count + short loop AllReduce (nightly) |
| `run_ops_extra_suite.py` | ≥2 (4-rank when device_count≥4) | Multi-item batch, uneven *V, optional multi-rank AllReduce |
| `probe_platform.py` | optional | npu-smi / CANN install.info diagnostics |
| `_platform.py` | — | Shared chip detect + ranktable helpers |

Optional native baseline: compile `c_allreduce_bench.c` locally (binary ignored by `.gitignore`).

## Environment

| Variable | Effect |
|---|---|
| `HCCL4PY_CHIP` | `a2` / `a3` / `a5` / `auto` (default) |
| `HCCL4PY_ASSUME_CHIP` | Fallback only when auto-detect finds no product tag |
| `HCCL_IF_IP` | HCCL runtime NIC selection (not written into ranktable; ranktable `host_ip` comes from `hostname -I` / UDP probe) |
| `HCCL4PY_PERF_WARMUP` / `ITERS` / `COUNT` / `ROUNDS` / `LIMIT_US` / `JITTER` | Perf knobs |
| `HCCL4PY_NRANKS` | Ops / extra suite rank count (default 2; extra may run 4-rank AllReduce when ≥4 devices) |
| `HCCL4PY_STRESS_COUNT` / `HCCL4PY_STRESS_REPEAT` | Stress suite size / loops |

Suites write their own ranktable when needed. Acceptance still defaults to 2 ranks; set `HCCL4PY_NRANKS` or run `run_ops_extra_suite.py` for multi-rank / uneven *V / multi-item batch.

## Acceptance notes

- ops suite: all 14 C APIs (13 entries; send/recv merged) must PASS (incl. A5 `all_gather_v` / `reduce_scatter_v`; no soft-SKIP)
- dtype suite on A5: only official unsupported dtypes are pre-skipped (`_platform.dtype_skip_cases("a5")`)
- cluster suite: ClusterInfo collectives always run (incl. A5)
- perf: host-enqueue wrapper overhead ≤10µs vs same-process `lib.Hccl*` ctypes call (`aclrtSynchronizeStream` is outside the timed window)
- domain: `hccl4py` imports domain APIs from `hcomm4py` (no local `comm.py`); put both python trees on `PYTHONPATH`

## Coverage matrix (acceptance / nightly)

| Suite | APIs | dtype | count | ranks | normal / negative | Notes |
|---|---|---|---|---|---|---|
| `run_ops_suite` | 14 | FP32 | 8 (V=1) | `HCCL4PY_NRANKS` (default 2) | normal | A5 *V must PASS |
| `run_ops_extra_suite` | batch×4 / uneven *V / AllReduce | FP32 | small | 2 or 4 | multi-item + uneven counts | A5 *V must PASS |
| `run_dtype_suite` | 14×12 | matrix | 4 | 2 | normal + official dtype skips | see `_platform.dtype_skip_cases` |
| `run_reduce_op_suite` | reduce family | FP32 | small | 2 | SUM/PROD/MAX/MIN | A5 runs ReduceScatterV |
| `run_boundary_suite` | sample ops | FP32 | edge | 2 | null / bad count / bad rank | offline + MP |
| `run_cluster_suite` | AllReduce path | FP32 | 8 | 2 | ClusterInfo | A5: init+collective; no forced ccu_sched |
| `run_comm_domain_suite` | 5 domain | — | — | 1 or 2 | RootInfo / ClusterInfo init+destroy | single-card runs test_1/2; multi-rank self-skips when device_count<2; ClusterInfo uses default cfg |
| `run_perf_bench` | 14 | FP32 | knob | 2 | host enqueue vs ctypes | sync outside timer |
| offline UT | types/ops/comm | — | — | 0 | float/bool reject, offsets gold, Comm.destroy idempotent | needs `hcomm4py` on path |

## ranktable

No static ranktable fixtures. Suites always generate ranktables at runtime via `_platform.write_ranktable_*` (`host_ip` from `hostname -I` / UDP probe; A3 single-host may omit `device_ip`, multi-host fills from `/etc/hccn.conf`). Schema follows the generators and the official `rank_table_config_*.md` docs.

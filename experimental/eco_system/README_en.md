# eco_system

[简体中文](./README.md) | English

Experimental ecosystem code. The `python/` subdirectory hosts **hccl4py**: Python bindings for HCCL collective operators (task book §3.1.2, 14 ops).

## Operators covered by hccl4py

`all_reduce` / `reduce` / `broadcast` / `all_gather` / `all_gather_v` / `scatter` /
`reduce_scatter` / `reduce_scatter_v` / `all_to_all` / `all_to_all_v` / `all_to_all_vc` /
`send` / `recv` / `batch_send_recv`

Domain APIs live in the sibling repo `cann/hcomm` as `hcomm4py` (task issue:
https://gitcode.com/cann/hcomm/issues/1157). `hccl4py` **depends on** `hcomm4py`:
Top-level `hccl4py` and `hccl4py.types` import domain APIs/types from `hcomm4py` — no in-tree `comm.py`. Put both
`experimental/eco_system/python` trees on `PYTHONPATH` (or install `hcomm4py` first).

## Layout

```text
experimental/eco_system/python/
├── PUBLIC_API.md / PUBLIC_API_en.md  # public API boundary
├── setup.py                    # wheel packaging
├── hccl4py/           # ops; domain APIs imported from hcomm4py at top level
├── examples/01_basic/ # AllReduce / Broadcast / AllGather / SendRecv
└── tests/             # ops / dtype / reduce_op / boundary / cluster / perf
```

Design RFC: [`docs/en/rfcs/0004-hccl4py-python-bindings.md`](../../docs/en/rfcs/0004-hccl4py-python-bindings.md)

## Usage

```bash
source /usr/local/Ascend/cann/set_env.sh   # CANN >= 9.2.0
# hcomm4py must be importable (sibling path or pip install)
export PYTHONPATH="/path/to/hcomm/experimental/eco_system/python:/path/to/hccl/experimental/eco_system/python:${PYTHONPATH}"
cd /path/to/hccl/experimental/eco_system/python
# optional wheel: python3 -m build  # via setup.py && pip install dist/hccl4py-*.whl
python3 examples/01_basic/01_allreduce.py --nranks 2
```

### Test entry points (`tests/`)

| Script | Purpose |
|---|---|
| `run_npu_smoke.py` | Library load + RootInfo smoke |
| `run_ops_suite.py` | Correctness of 14 ops |
| `run_dtype_suite.py` | Dtype coverage |
| `run_reduce_op_suite.py` | Reduce ops |
| `run_boundary_suite.py` | Boundary cases |
| `run_cluster_suite.py` | Multi-card (needs ranktable) |
| `run_comm_domain_suite.py` | 5 domain APIs |
| `run_ops_extra_suite.py` | Multi-item batch / uneven *V / multi-rank |
| `run_stress_suite.py` | Large-count short loop (nightly) |
| `run_perf_bench.py` | Python wrapper overhead vs same-process ctypes `lib.Hccl*` (≤10µs) |
| `c_allreduce_bench.c` | Optional native C baseline (compile yourself; main acceptance path below) |

Cluster / perf example (suites write their own ranktable; default 2 ranks, no scale CLI):

```bash
export HCCL_IF_IP=<host IP>
# optional: export HCCL4PY_CHIP=a5
python3 tests/run_cluster_suite.py
python3 tests/run_perf_bench.py
# perf knobs: HCCL4PY_PERF_WARMUP/ITERS/COUNT/ROUNDS/LIMIT_US/JITTER (see tests/README.md)
```

## Acceptance criteria

| Item | Requirement | Private A5 result |
|---|---|---|
| Hardware | A3 **or** A5, ≥2 cards | Ascend 950PR ×2 |
| CANN | ≥ 9.2.0 (incl. 9.2.0-beta.x) | 9.2.0-beta.2 |
| Function | 14 ops correct | ops: all 14 C APIs / 13 suite entries (send/recv merged) PASS (incl. A5 `all_gather_v`/`reduce_scatter_v`; no soft-SKIP); dtype: only official unsupported dtypes pre-skipped (see `_platform.dtype_skip_cases`) |
| Perf | Py wrapper **host-enqueue** overhead vs same-process ctypes-direct `lib.Hccl*` ≤10µs (±10%) | measured ≤11µs in the host-enqueue window (incl. ±10% tolerance) |

Note: the “C path” in `run_perf_bench.py` is raw `lib.Hccl*` via ctypes in the same Python process (no hccl4py wrapper body). Timing covers the host submit loop after warmup + pre-sync; `aclrtSynchronizeStream` runs after the stopwatch so device-bound work cannot zero out wrapper overhead.

Issue: https://gitcode.com/cann/hccl/issues/875

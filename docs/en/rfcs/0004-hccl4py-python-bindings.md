# RFC: hccl4py — HCCL Collective/P2P Python Bindings

- Start date: 2026-10-14
- RFC PR: [cann/hccl#3798](https://gitcode.com/cann/hccl/merge_requests/3798)
- Related: community task “implement hccl4py”; discussion https://gitcode.com/cann/hccl/discussions/19

---

## Summary

Provide ctypes bindings for 14 HCCL collective/P2P C APIs under `experimental/eco_system/python` as package `hccl4py`, matching C semantics with wrapper overhead target ≤ 10µs. Public surface is governed by `PUBLIC_API.md`; wheels are built via `setup.py`.

## Motivation

- HCCL ships C headers + shared libraries only; hand-written ctypes is error-prone.
- Community task requires Python wrappers for framework-facing communication ops.
- Target users: custom trainers, HPC scripts, tuning tools.

## Terminology

| Term | Meaning |
|------|---------|
| hccl4py | This-repo Python package (ops + re-exported hcomm4py domain APIs) |
| hcomm4py | Sister-repo canonical domain package |
| Public boundary | Stable API surface constrained by `PUBLIC_API.md` and `__all__` |

## Architecture and Contracts

### Layout

```text
experimental/eco_system/python/
├── PUBLIC_API.md / PUBLIC_API_en.md
├── setup.py
├── hccl4py/
│   ├── __init__.py      # ops + types + re-exported hcomm4py domain APIs
│   ├── _hcomm_dep.py    # hard dependency on hcomm4py (no local comm.py)
│   ├── _lib.py          # load libhccl / libhcomm / libascendcl
│   ├── _validate.py     # ops-side argument checks
│   ├── _bindings.py     # ctypes argtypes/restype tables for 14 ops
│   ├── _ops_util.py     # handle/pointer helpers
│   ├── types.py         # re-export from hcomm4py.types
│   └── ops.py           # 14 collective/P2P wrappers (this RFC)
├── examples/
└── tests/
```

Flow: Python call → argument validation → ctypes conversion (one-time argtypes) → `libhccl.so` → NPU.

This RFC covers **§3.1.2 ops (14)**. Domain APIs are covered by sister PR [cann/hcomm#6331](https://gitcode.com/cann/hcomm/merge_requests/6331) (use that PR until the RFC lands on master). There is **no** in-tree domain fallback.

### Public API

Stable symbols are those in `hccl4py.__all__` / `PUBLIC_API.md` (types, re-exported domain APIs / ABI size constants, 14 ops, `make_send_recv_item`).

### Dependencies

Runtime `libhcomm.so` / `libhccl.so` / `libascendcl.so`. Search order matches `hccl4py/_lib.py`: `HCCL4PY_LIB_DIR` / `HCCL4PY_HCCL_LIB` / `HCCL4PY_HCOMM_LIB` / `HCCL4PY_ASCENDCL_LIB`, then `ASCEND_LIBRARY_PATH` / `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` (plus `lib64`), and `LD_LIBRARY_PATH`.

## Impact

- Perf: wrapper overhead ≤ 10µs vs same-process ctypes direct call.
- Scope: primarily additive under `experimental/`; this PR also includes a small
  `src/` topo UBG gate fix needed for Ascend 950PR acceptance on CANN 9.2 beta.
- Release: optional wheel (`python -m build`); CMake artifacts unchanged.

## Compatibility

Additive only; no feature flag. `HcclCommConfig` layout is owned by sister `hcomm4py` (runtime capability → version/size; offline fallback v11/2248); this package only re-exports the types/constants.

## Detailed Design

- ctypes prototypes live in `_bindings.py`; one-shot `ops._bind()` wires argtypes/restype while `ops.py` keeps semantic wrappers under the ~500-line guidance.
- Validate count/rank/dtype/reduce_op before native calls (`_validate.py` / `_ops_util.py`).
- No in-tree `hccl4py/comm.py`: domain APIs come from `hcomm4py` (cann/hcomm) via top-level import so ops and domain share one `libhcomm`/`libhccl` load (cross-CDLL handles hang collectives).

### Packaging

```bash
cd experimental/eco_system/python
python3 -m build  # via setup.py
pip install dist/hccl4py-*.whl
```

## Testing

| Item | Method |
|------|--------|
| Offline | `tests/test_offline_bindings.py` |
| Functional | `tests/run_*_suite.py` on ≥2 devices |
| Perf | `tests/run_perf_bench.py` |
| Env | A3 or A5, CANN ≥ 9.2.0 |

## Alternatives

ctypes (chosen) vs pybind11/cffi vs docs-only examples.

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| ctypes struct layout / alignment mistakes | Trust runtime `sizeof`; gold-offset UT + optional header offsetof probe |
| CANN version field growth | Config buffer reserved; layout owned by sister `hcomm4py` (`explicit > capability-as-version > DEFAULT v11/2248`) |
| Dual-CDLL handles not interchangeable across packages | Document same-process `hcomm4py` + `hccl4py` coupling; acceptance uses one process |

## Open Questions

1. Long-term ownership of acceptance hardware.
2. Whether examples should split per op after merge.

---

## Review Notes

See discussion on cann/hccl#3798.

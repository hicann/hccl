#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reduce-op coverage: SUM/PROD/MAX/MIN on AllReduce/Reduce/ReduceScatter/ReduceScatterV (FP32)."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import sys
import traceback

HERE = os.environ.get(
    "HCCL4PY_PYTHON",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

# rank0=2.0, rank1=3.0  → SUM=5 MAX=3 MIN=2 PROD=6
CASES = []
for op in ("all_reduce", "reduce", "reduce_scatter", "reduce_scatter_v"):
    CASES.extend(
        [
            (op, "SUM", 5.0),
            (op, "PROD", 6.0),
            (op, "MAX", 3.0),
            (op, "MIN", 2.0),
        ]
    )


def _acl(device=0):
    from _platform import ensure_acl

    return ensure_acl(device)



def _worker(rank, nranks, root_bytes, cases, q):
    try:
        import hccl4py as h

        acl = _acl(rank)
        stream = ctypes.c_void_p()
        assert acl.aclrtCreateStream(ctypes.byref(stream)) == 0
        root = h.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        from _platform import make_test_comm_config

        cfg = make_test_comm_config(h)
        comm = h.comm_init_root_info_config(nranks, root, rank, config=cfg)

        dt = h.HcclDataType.FP32
        val = 2.0 if rank == 0 else 3.0
        results = []
        for op, rop_name, expect in cases:
            rop = getattr(h.HcclReduceOp, rop_name)
            label = "%s/%s" % (op, rop_name)
            try:
                if op in ("all_reduce", "reduce"):
                    count = 4
                    nbytes = count * 4
                    sh = (ctypes.c_float * count)(*([val] * count))
                    rh = (ctypes.c_float * count)()
                    sbuf = ctypes.c_void_p()
                    rbuf = ctypes.c_void_p()
                    assert acl.aclrtMalloc(ctypes.byref(sbuf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
                    assert acl.aclrtMalloc(ctypes.byref(rbuf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
                    assert acl.aclrtMemcpy(sbuf, nbytes, sh, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
                    if op == "all_reduce":
                        h.all_reduce(
                            comm, sbuf, rbuf, count, data_type=dt, reduce_op=rop, stream=stream
                        )
                    else:
                        h.reduce(
                            comm,
                            sbuf,
                            rbuf,
                            count,
                            data_type=dt,
                            reduce_op=rop,
                            root=0,
                            stream=stream,
                        )
                    assert acl.aclrtSynchronizeStream(stream) == 0
                    assert acl.aclrtMemcpy(rh, nbytes, rbuf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
                    if op == "reduce" and rank != 0:
                        ok = True
                    else:
                        ok = all(abs(rh[i] - expect) < 1e-3 for i in range(count))
                    detail = {"out0": float(rh[0]), "expect": expect}
                    acl.aclrtFree(sbuf)
                    acl.aclrtFree(rbuf)
                elif op == "reduce_scatter":
                    send_n = nranks
                    sh = (ctypes.c_float * send_n)(*([val] * send_n))
                    rh = (ctypes.c_float * 1)()
                    sbuf = ctypes.c_void_p()
                    rbuf = ctypes.c_void_p()
                    assert (
                        acl.aclrtMalloc(ctypes.byref(sbuf), send_n * 4, ACL_MEM_MALLOC_HUGE_FIRST)
                        == 0
                    )
                    assert acl.aclrtMalloc(ctypes.byref(rbuf), 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
                    assert (
                        acl.aclrtMemcpy(sbuf, send_n * 4, sh, send_n * 4, ACL_MEMCPY_HOST_TO_DEVICE)
                        == 0
                    )
                    h.reduce_scatter(
                        comm, sbuf, rbuf, 1, data_type=dt, reduce_op=rop, stream=stream
                    )
                    assert acl.aclrtSynchronizeStream(stream) == 0
                    assert acl.aclrtMemcpy(rh, 4, rbuf, 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
                    ok = abs(rh[0] - expect) < 1e-3
                    detail = {"out0": float(rh[0]), "expect": expect}
                    acl.aclrtFree(sbuf)
                    acl.aclrtFree(rbuf)
                else:  # reduce_scatter_v
                    send_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
                    send_displs = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
                    sh = (ctypes.c_float * nranks)(*([val] * nranks))
                    rh = (ctypes.c_float * 1)()
                    sbuf = ctypes.c_void_p()
                    rbuf = ctypes.c_void_p()
                    assert (
                        acl.aclrtMalloc(ctypes.byref(sbuf), nranks * 4, ACL_MEM_MALLOC_HUGE_FIRST)
                        == 0
                    )
                    assert acl.aclrtMalloc(ctypes.byref(rbuf), 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
                    assert (
                        acl.aclrtMemcpy(sbuf, nranks * 4, sh, nranks * 4, ACL_MEMCPY_HOST_TO_DEVICE)
                        == 0
                    )
                    h.reduce_scatter_v(
                        comm,
                        sbuf,
                        send_counts,
                        send_displs,
                        rbuf,
                        1,
                        data_type=dt,
                        reduce_op=rop,
                        stream=stream,
                    )
                    assert acl.aclrtSynchronizeStream(stream) == 0
                    assert acl.aclrtMemcpy(rh, 4, rbuf, 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
                    ok = abs(rh[0] - expect) < 1e-3
                    detail = {"out0": float(rh[0]), "expect": expect}
                    acl.aclrtFree(sbuf)
                    acl.aclrtFree(rbuf)
                results.append({"label": label, "ok": ok, "detail": detail, "unsupported": False})
            except Exception as e:
                from _platform import is_not_supported

                unsupported = is_not_supported(e)
                results.append(
                    {
                        "label": label,
                        "ok": False,
                        "unsupported": unsupported,
                        "error": traceback.format_exc(),
                    }
                )

        h.comm_destroy(comm)
        q.put({"rank": rank, "ok": True, "results": results})
    except Exception:
        q.put({"rank": rank, "ok": False, "error": traceback.format_exc(), "results": []})


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    acl = _acl()
    n = ctypes.c_uint32()
    assert acl.aclrtGetDeviceCount(ctypes.byref(n)) == 0
    if int(n.value) < 2:
        print("NEED >=2 devices")
        return 2

    import hccl4py as h
    from _platform import detect_chip, reduce_op_skip_cases

    chip = detect_chip()
    skip_map = reduce_op_skip_cases(chip)
    cases = []
    skipped = 0
    for op, rop_name, expect in CASES:
        reason = skip_map.get((op, rop_name))
        if reason:
            print("[SKIP] %s/%s %s" % (op, rop_name, reason), flush=True)
            skipped += 1
            continue
        cases.append((op, rop_name, expect))
    print("chip=%s planned=%d (+%d pre-skip)" % (chip, len(cases), skipped), flush=True)

    root = h.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from _platform import mp_collect, start_processes

    procs = [ctx.Process(target=_worker, args=(r, 2, root_bytes, cases, q)) for r in range(2)]
    procs = start_processes(procs)
    payloads = mp_collect(q, 2, 600, procs)

    if not all(p.get("ok") for p in payloads):
        print("[FAIL] worker init")
        for p in payloads:
            if not p.get("ok"):
                print(p.get("error"))
        return 1

    by_label = {}
    for p in payloads:
        for r in p["results"]:
            by_label.setdefault(r["label"], []).append(r)

    passed = 0
    failed = []
    for label, rows in by_label.items():
        if all(r.get("ok") for r in rows):
            print("[PASS]", label, rows[0].get("detail"), flush=True)
            passed += 1
        elif all(r.get("unsupported") for r in rows):
            print("[SKIP]", label, "NOT_SUPPORT", flush=True)
            skipped += 1
        else:
            print("[FAIL]", label, flush=True)
            for r in rows:
                if not r.get("ok"):
                    print(" ", str(r.get("error") or r)[:400], flush=True)
            failed.append(label)

    print(
        "=== SUMMARY pass=%d skip=%d fail=%d failed=%s ==="
        % (passed, skipped, len(failed), failed or "none"),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

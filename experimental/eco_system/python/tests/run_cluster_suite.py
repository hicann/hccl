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

"""Happy-path ClusterInfo (ranktable) init + multi-op on 2 devices.

Auto-selects A2 / A3 / A5 ranktable via tests._platform (override: HCCL4PY_CHIP).
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import sys
import tempfile
import traceback

HERE = os.environ.get(
    "HCCL4PY_PYTHON",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_TESTS = os.path.join(HERE, "tests")
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _acl(device=0):
    from _platform import ensure_acl

    return ensure_acl(device)



def _device_count():
    acl = _acl(0)
    n = ctypes.c_uint32()
    assert acl.aclrtGetDeviceCount(ctypes.byref(n)) == 0
    return int(n.value)


def write_ranktable(path: str, logic_ids) -> str:
    from _platform import write_ranktable as _wr

    p, chip = _wr(path, logic_ids)
    print("chip=%s ranktable_schema=%s" % (chip, {"a5": "v2.0+EID", "a3": "v1.0 server", "a2": "v1.0"}[chip]), flush=True)
    return p


def _worker(rank, nranks, ranktable_path, q):
    try:
        import hccl4py as h

        acl = _acl(rank)
        stream = ctypes.c_void_p()
        assert acl.aclrtCreateStream(ctypes.byref(stream)) == 0

        # ClusterInfo path on A5 is unstable with forced ccu_sched; use toolkit default.
        comm = h.comm_init_cluster_info_config(ranktable_path, rank)
        count = 8
        nbytes = count * 4
        dt = h.HcclDataType.FP32
        details = {}

        # 1) AllReduce
        send_h = (ctypes.c_float * count)(*[float(rank + 1) for _ in range(count)])
        recv_h = (ctypes.c_float * count)(*([0.0] * count))
        send_buf = ctypes.c_void_p()
        recv_buf = ctypes.c_void_p()
        assert acl.aclrtMalloc(ctypes.byref(send_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
        assert acl.aclrtMalloc(ctypes.byref(recv_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
        assert acl.aclrtMemcpy(send_buf, nbytes, send_h, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
        assert acl.aclrtMemset(recv_buf, nbytes, 0, nbytes) == 0
        h.all_reduce(comm, send_buf, recv_buf, count, data_type=dt, stream=stream)
        assert acl.aclrtSynchronizeStream(stream) == 0
        assert acl.aclrtMemcpy(recv_h, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
        ar_expect = float(sum(range(1, nranks + 1)))
        ok_ar = all(abs(recv_h[i] - ar_expect) < 1e-3 for i in range(count))
        details["all_reduce"] = {"out0": float(recv_h[0]), "expect": ar_expect, "ok": ok_ar}

        # 2) Broadcast (root=0 value 7)
        bval = 7.0 if rank == 0 else 0.0
        bh = (ctypes.c_float * count)(*([bval] * count))
        assert acl.aclrtMemcpy(send_buf, nbytes, bh, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
        h.broadcast(comm, send_buf, count, data_type=dt, root=0, stream=stream)
        assert acl.aclrtSynchronizeStream(stream) == 0
        assert acl.aclrtMemcpy(recv_h, nbytes, send_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
        ok_bc = all(abs(recv_h[i] - 7.0) < 1e-3 for i in range(count))
        details["broadcast"] = {"out0": float(recv_h[0]), "expect": 7.0, "ok": ok_bc}

        # 3) AllGather (1 elem each)
        one = (ctypes.c_float * 1)(float(rank + 1))
        ag_out = (ctypes.c_float * nranks)()
        assert acl.aclrtMemcpy(send_buf, 4, one, 4, ACL_MEMCPY_HOST_TO_DEVICE) == 0
        h.all_gather(comm, send_buf, recv_buf, 1, data_type=dt, stream=stream)
        assert acl.aclrtSynchronizeStream(stream) == 0
        assert acl.aclrtMemcpy(ag_out, nranks * 4, recv_buf, nranks * 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
        ag_expect = [float(i + 1) for i in range(nranks)]
        ok_ag = all(abs(ag_out[i] - ag_expect[i]) < 1e-3 for i in range(nranks))
        details["all_gather"] = {
            "out": [float(ag_out[i]) for i in range(nranks)],
            "expect": ag_expect,
            "ok": ok_ag,
        }

        # 4) ReduceScatter
        rs_send = (ctypes.c_float * nranks)(*([float(rank + 1)] * nranks))
        rs_recv = (ctypes.c_float * 1)()
        assert acl.aclrtMemcpy(send_buf, nranks * 4, rs_send, nranks * 4, ACL_MEMCPY_HOST_TO_DEVICE) == 0
        h.reduce_scatter(comm, send_buf, recv_buf, 1, data_type=dt, stream=stream)
        assert acl.aclrtSynchronizeStream(stream) == 0
        assert acl.aclrtMemcpy(rs_recv, 4, recv_buf, 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
        ok_rs = abs(rs_recv[0] - ar_expect) < 1e-3
        details["reduce_scatter"] = {"out0": float(rs_recv[0]), "expect": ar_expect, "ok": ok_rs}

        h.comm_destroy(comm)
        ok = ok_ar and ok_bc and ok_ag and ok_rs
        q.put({"rank": rank, "ok": ok, "details": details})
    except Exception:
        q.put({"rank": rank, "ok": False, "error": traceback.format_exc()})


def main():
    from _platform import detect_chip, platform_summary

    print("platform=", platform_summary(), flush=True)
    chip = detect_chip()
    n = _device_count()
    print("device_count=%d chip=%s" % (n, chip))
    if n < 2:
        print("NEED >=2 devices")
        return 2
    nranks = 2
    logic_ids = list(range(nranks))
    fd, path = tempfile.mkstemp(prefix="hccl4py_ranktable_", suffix=".json")
    os.close(fd)
    try:
        # Parent must have ACL device context for phy-id query
        write_ranktable(path, logic_ids)
        print("ranktable=", path)
        with open(path, encoding="utf-8") as f:
            print(f.read())

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        from _platform import mp_collect, start_processes

        procs = [
            ctx.Process(target=_worker, args=(r, nranks, path, q)) for r in range(nranks)
        ]
        procs = start_processes(procs)

        results = mp_collect(q, nranks, 240, procs)

        ok = all(r.get("ok") for r in results)
        status = "PASS" if ok else "FAIL"
        print("[%s] cluster_info_multi_op" % status)
        for r in results:
            if r.get("ok"):
                print("  rank", r["rank"], r.get("details"))
            else:
                print("  rank", r.get("rank"), r.get("error") or r)
        print("=== SUMMARY failed=%s ===" % ("none" if ok else "cluster_info"))
        return 0 if ok else 1
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

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

"""Larger-count / short-loop stress for AllReduce (nightly-friendly).

Env:
  HCCL4PY_STRESS_COUNT   elements (default 1048576 ≈ 4MiB FP32)
  HCCL4PY_STRESS_REPEAT  loop iterations (default 32)
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import sys
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _platform import make_test_comm_config, mp_collect  # noqa: E402

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _acl(device=0):
    from _platform import ensure_acl

    return ensure_acl(device)



def _worker(rank, nranks, root_bytes, count, repeat, q):
    send_buf = recv_buf = None
    stream = None
    acl = None
    try:
        import hccl4py as h

        acl = _acl(rank)
        stream = ctypes.c_void_p()
        assert acl.aclrtCreateStream(ctypes.byref(stream)) == 0
        root = h.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        cfg = make_test_comm_config(h)
        comm = h.comm_init_root_info_config(nranks, root, rank, config=cfg)

        nbytes = count * 4
        host = (ctypes.c_float * count)(*([float(rank + 1)] * count))
        send_buf = ctypes.c_void_p()
        recv_buf = ctypes.c_void_p()
        assert acl.aclrtMalloc(ctypes.byref(send_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
        assert acl.aclrtMalloc(ctypes.byref(recv_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
        assert (
            acl.aclrtMemcpy(send_buf, nbytes, host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
        )

        for _ in range(repeat):
            h.all_reduce(
                comm,
                send_buf,
                recv_buf,
                count,
                data_type=h.HcclDataType.FP32,
                stream=stream,
            )
        assert acl.aclrtSynchronizeStream(stream) == 0

        ncopy = min(8, count)
        out = (ctypes.c_float * ncopy)()
        nbytes_out = ncopy * ctypes.sizeof(ctypes.c_float)
        assert (
            acl.aclrtMemcpy(
                out, nbytes_out, recv_buf, nbytes_out, ACL_MEMCPY_DEVICE_TO_HOST
            )
            == 0
        )
        expect = float(sum(range(1, nranks + 1)))
        ok = all(abs(out[i] - expect) < 1e-2 for i in range(ncopy))
        h.comm_destroy(comm)
        q.put({"rank": rank, "ok": ok, "expect": expect, "out0": float(out[0])})
    except Exception:
        q.put({"rank": rank, "ok": False, "error": traceback.format_exc()})
    finally:
        if acl is not None:
            if send_buf:
                acl.aclrtFree(send_buf)
            if recv_buf:
                acl.aclrtFree(recv_buf)


def main():
    import hccl4py as h

    count = int(os.environ.get("HCCL4PY_STRESS_COUNT", "1048576"))
    repeat = int(os.environ.get("HCCL4PY_STRESS_REPEAT", "32"))
    print("stress count=%d repeat=%d" % (count, repeat))

    acl = _acl(0)
    n = ctypes.c_uint32()
    assert acl.aclrtGetDeviceCount(ctypes.byref(n)) == 0
    if int(n.value) < 2:
        print("NEED >=2 devices")
        return 2
    assert acl.aclrtSetDevice(0) == 0
    root = h.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from _platform import start_processes

    procs = [
        ctx.Process(target=_worker, args=(r, 2, root_bytes, count, repeat, q))
        for r in range(2)
    ]
    procs = start_processes(procs)
    results = mp_collect(q, 2, 600, procs, join_timeout=60)
    for r in results:
        print(r)
    ok = all(r.get("ok") for r in results)
    print("=== SUMMARY stress failed=%s ===" % ("none" if ok else "allreduce_stress"))
    return 0 if ok else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

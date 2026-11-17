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

"""2-rank Send/Recv smoke. Requires >=2 NPU + CANN>=9.2."""

import ctypes
import multiprocessing as mp
import os
import sys


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_EX = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _EX not in sys.path:
    sys.path.insert(0, _EX)


def _worker(rank, nranks, root_bytes):
    import hccl4py as h
    from _acl_runtime import ensure_acl

    ACL_MEM_MALLOC_HUGE_FIRST = 0
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2

    acl = ensure_acl(rank)
    stream = None
    comm = None
    sbuf = None
    rbuf = None
    try:
        stream = ctypes.c_void_p()
        _check(acl.aclrtCreateStream(ctypes.byref(stream)) == 0, "acl.aclrtCreateStream failed")
        root = h.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        comm = h.comm_init_root_info_config(nranks, root, rank)

        count = 4
        nbytes = count * 4
        peer = 1 - rank
        host = (ctypes.c_float * count)(*([float(rank + 1)] * count))
        sbuf = ctypes.c_void_p()
        rbuf = ctypes.c_void_p()
        _check(
            acl.aclrtMalloc(ctypes.byref(sbuf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0,
            "acl.aclrtMalloc sbuf failed",
        )
        _check(
            acl.aclrtMalloc(ctypes.byref(rbuf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0,
            "acl.aclrtMalloc rbuf failed",
        )
        _check(
            acl.aclrtMemcpy(sbuf, nbytes, host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0,
            "host-to-device memcpy failed",
        )
        _check(acl.aclrtMemset(rbuf, nbytes, 0, nbytes) == 0, "acl.aclrtMemset failed")
        dt = h.HcclDataType.FP32
        if rank == 0:
            _check(h.send(comm, sbuf, count, dt, dest_rank=peer, stream=stream) == 0, "send failed")
            _check(h.recv(comm, rbuf, count, dt, src_rank=peer, stream=stream) == 0, "recv failed")
        else:
            _check(h.recv(comm, rbuf, count, dt, src_rank=peer, stream=stream) == 0, "recv failed")
            _check(h.send(comm, sbuf, count, dt, dest_rank=peer, stream=stream) == 0, "send failed")
        _check(acl.aclrtSynchronizeStream(stream) == 0, "acl.aclrtSynchronizeStream failed")
        out = (ctypes.c_float * count)()
        _check(
            acl.aclrtMemcpy(out, nbytes, rbuf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0,
            "device-to-host memcpy failed",
        )
        expect = float(peer + 1)
        print("rank=%d out0=%.1f expect=%.1f" % (rank, out[0], expect))
        _check(abs(out[0] - expect) < 1e-3, "value check failed: abs(out[0] - expect) < 1e-3")
        _check(h.comm_destroy(comm) == 0, "comm_destroy failed")
        comm = None
    finally:
        if sbuf:
            acl.aclrtFree(sbuf)
        if rbuf:
            acl.aclrtFree(rbuf)
        if stream:
            acl.aclrtDestroyStream(stream)
        if comm is not None:
            try:
                h.comm_destroy(comm)
            except Exception:
                pass
        acl.aclrtResetDevice(rank)


def main():
    import hccl4py as h
    from _acl_runtime import ensure_acl

    ensure_acl(0)
    root = h.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, 2, root_bytes)) for r in range(2)]
    for p in procs:
        p.start()
    rc = 0
    for p in procs:
        p.join()
        if p.exitcode:
            rc = p.exitcode
    return rc


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

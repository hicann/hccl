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

"""Minimal 2-rank AllReduce smoke (requires >=2 NPU + CANN).

Usage (on device, after source set_env.sh):
  # hccl4py depends on hcomm4py — put both trees on PYTHONPATH
  export PYTHONPATH=<hcomm/.../python>:<hccl/.../python>:$PYTHONPATH
  python3 01_allreduce.py
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import sys

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_EX = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _EX not in sys.path:
    sys.path.insert(0, _EX)


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def _worker(rank: int, nranks: int, root_bytes: bytes, count: int) -> None:
    import hccl4py as h4
    from _acl_runtime import ensure_acl

    ACL_MEM_MALLOC_HUGE_FIRST = 0
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2

    acl = ensure_acl(rank)
    stream = None
    handle = None
    send_buf = None
    recv_buf = None
    try:
        stream = ctypes.c_void_p()
        _check(acl.aclrtCreateStream(ctypes.byref(stream)) == 0, "acl.aclrtCreateStream failed")
        root = h4.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        # Default layout follows runtime capability (legacy v12/2248, V2 v10/2240).
        cfg = h4.comm_config_init()
        handle = h4.comm_init_root_info_config(nranks, root, rank, config=cfg)

        nbytes = count * 4
        host = (ctypes.c_float * count)(*[float(rank + i) for i in range(count)])
        send_buf = ctypes.c_void_p()
        recv_buf = ctypes.c_void_p()
        _check(
            acl.aclrtMalloc(ctypes.byref(send_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0,
            "acl.aclrtMalloc send failed",
        )
        _check(
            acl.aclrtMalloc(ctypes.byref(recv_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0,
            "acl.aclrtMalloc recv failed",
        )
        _check(
            acl.aclrtMemcpy(send_buf, nbytes, host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0,
            "ACL memcpy H2D failed",
        )
        _check(
            h4.all_reduce(
                handle,
                send_buf,
                recv_buf,
                count,
                data_type=h4.HcclDataType.FP32,
                reduce_op=h4.HcclReduceOp.SUM,
                stream=stream,
            )
            == 0,
            "all_reduce failed",
        )
        _check(acl.aclrtSynchronizeStream(stream) == 0, "acl.aclrtSynchronizeStream failed")
        out = (ctypes.c_float * count)()
        _check(
            acl.aclrtMemcpy(out, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0,
            "ACL memcpy D2H failed",
        )
        print(f"rank={rank} out={[out[i] for i in range(min(count, 8))]}")
        _check(h4.comm_destroy(handle) == 0, "comm_destroy failed")
        handle = None
    finally:
        if send_buf:
            acl.aclrtFree(send_buf)
        if recv_buf:
            acl.aclrtFree(recv_buf)
        if stream:
            acl.aclrtDestroyStream(stream)
        if handle is not None:
            try:
                h4.comm_destroy(handle)
            except Exception:
                pass
        acl.aclrtResetDevice(rank)


def main() -> int:
    p = argparse.ArgumentParser(description="hccl4py AllReduce 2-rank smoke")
    p.add_argument("--nranks", type=int, default=2)
    p.add_argument("--count", type=int, default=8)
    args = p.parse_args()

    import hccl4py as h4
    from _acl_runtime import ensure_acl

    ensure_acl(0)
    root = h4.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(r, args.nranks, root_bytes, args.count))
        for r in range(args.nranks)
    ]
    for proc in procs:
        proc.start()
    rc = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            rc = proc.exitcode or 1
    return rc


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

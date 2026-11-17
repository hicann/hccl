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

"""NPU smoke for hccl4py on single-machine multi-device (device_count>=1).

AllReduce path reuses ``run_ops_suite.run_op`` (same worker as the ops matrix) so
smoke cannot drift into a second, subtly broken spawn/init recipe. Default-stream
(``stream=None``) is covered by a dedicated 2-rank check after that.
"""

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


def _acl(device=0):
    from _platform import ensure_acl

    return ensure_acl(device)



def test_cpu_config():
    import hccl4py as h
    from hccl4py import types as T

    cfg = h.comm_config_init()
    assert ctypes.sizeof(cfg) >= T.HCCL_COMM_CONFIG_SIZE_V10
    magic = ctypes.c_uint32.from_buffer(cfg._raw, T._OFF["magic"]).value
    ver = ctypes.c_uint32.from_buffer(cfg._raw, T._OFF["version"]).value
    print("[PASS] cpu_config magic=0x%x version=%d" % (magic, ver))
    assert magic == T.HCCL_COMM_CONFIG_MAGIC_WORD
    assert ver >= 10


def test_load_and_rootinfo():
    import hccl4py as h

    acl = _acl(0)
    n = ctypes.c_uint32()
    assert acl.aclrtGetDeviceCount(ctypes.byref(n)) == 0
    print("[INFO] device_count=%d" % n.value)
    assert n.value >= 1
    info = h.get_root_info()
    assert ctypes.sizeof(info) == 4108
    assert h.HCCL_ROOT_INFO_BYTES == 4108
    print("[PASS] get_root_info sizeof=%d" % ctypes.sizeof(info))
    return int(n.value)


def test_allreduce_2dev(nranks):
    """Delegate to ops suite — same RootInfo + CommConfig + spawn recipe."""
    if nranks < 2:
        print("[SKIP] allreduce_2dev need device_count>=2, got %d" % nranks)
        return False
    import run_ops_suite as ros

    ok = ros.run_op("all_reduce", 2)
    print("[%s] allreduce_2dev" % ("PASS" if ok else "FAIL"))
    return bool(ok)


def _default_stream_worker(rank, nranks, root_bytes, q):
    try:
        import hccl4py as h
        from _platform import make_test_comm_config

        acl = _acl(rank)

        # Resolve the same default stream hccl4py uses for stream=None, then sync it.
        get_def = acl.aclrtCtxGetCurrentDefaultStream
        get_def.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        get_def.restype = ctypes.c_int
        default_stream = ctypes.c_void_p()
        assert get_def(ctypes.byref(default_stream)) == 0
        assert default_stream.value

        root = h.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        cfg = make_test_comm_config(h)
        comm = h.comm_init_root_info_config(nranks, root, rank, config=cfg)

        count = 8
        nbytes = count * 4
        host = (ctypes.c_float * count)(*([float(rank + 1)] * count))
        send_buf = ctypes.c_void_p()
        recv_buf = ctypes.c_void_p()
        assert acl.aclrtMalloc(ctypes.byref(send_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
        assert acl.aclrtMalloc(ctypes.byref(recv_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
        assert acl.aclrtMemcpy(send_buf, nbytes, host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
        assert acl.aclrtMemset(recv_buf, nbytes, 0, nbytes) == 0

        h.all_reduce(
            comm,
            int(send_buf.value or 0),
            int(recv_buf.value or 0),
            count,
            data_type=h.HcclDataType.FP32,
            stream=None,
        )
        assert acl.aclrtSynchronizeStream(default_stream) == 0
        out = (ctypes.c_float * count)()
        assert acl.aclrtMemcpy(out, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
        expect = float(sum(range(1, nranks + 1)))
        ok = all(abs(out[i] - expect) < 1e-3 for i in range(count))
        h.comm_destroy(comm)
        q.put({"rank": rank, "ok": ok, "out0": float(out[0]), "expect": expect})
    except Exception:
        q.put({"rank": rank, "ok": False, "error": traceback.format_exc()})


def test_allreduce_default_stream(nranks):
    import hccl4py as h
    from _platform import mp_collect

    if nranks < 2:
        print("[SKIP] allreduce_default_stream need >=2 devices")
        return True

    acl = _acl()
    assert acl.aclrtSetDevice(0) == 0
    root = h.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from _platform import start_processes

    procs = [
        ctx.Process(target=_default_stream_worker, args=(r, nranks, root_bytes, q))
        for r in range(nranks)
    ]
    procs = start_processes(procs)
    results = mp_collect(q, nranks, 180, procs)
    ok = len(results) == nranks and all(r.get("ok") for r in results)
    print("[%s] allreduce_default_stream" % ("PASS" if ok else "FAIL"))
    for r in results:
        print("  rank", r.get("rank"), r.get("error") or r)
    return ok


def main():
    print("=== hccl4py NPU smoke ===")
    test_cpu_config()
    n = test_load_and_rootinfo()
    ok = test_allreduce_2dev(min(n, 2))
    ok_def = test_allreduce_default_stream(min(n, 2)) if n >= 2 else True
    print("=== DONE n=%d allreduce_ok=%s default_stream_ok=%s ===" % (n, ok, ok_def))
    return 0 if (ok and ok_def) or n < 2 else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

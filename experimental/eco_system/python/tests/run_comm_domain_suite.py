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

"""Explicit self-test for the 5 hcomm communication-domain APIs (task §3.1.1)."""

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
_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

def _ensure_acl(device: int = 0):
    from _platform import ensure_acl

    return ensure_acl(device)


def _device_count():
    acl = _ensure_acl(0)
    n = ctypes.c_uint32()
    assert acl.aclrtGetDeviceCount(ctypes.byref(n)) == 0
    return int(n.value)


def _rootinfo_worker(rank, root_bytes, q):
    try:
        import hccl4py as hh

        _ensure_acl(rank)
        r = hh.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(r), root_bytes, len(root_bytes))
        from _platform import make_test_comm_config

        c = make_test_comm_config(hh)
        comm = hh.comm_init_root_info_config(2, r, rank, config=c)
        assert comm
        rc = int(hh.comm_destroy(comm))
        q.put({"rank": rank, "ok": rc == 0, "destroy_rc": rc, "comm": int(comm)})
    except Exception:
        q.put({"rank": rank, "ok": False, "error": traceback.format_exc()})


def _cluster_worker(rank, path, q):
    try:
        import hccl4py as hh

        _ensure_acl(rank)
        # ClusterInfo on A5 is unstable with forced ccu_sched (same as
        # run_cluster_suite); use toolkit default CommConfig.
        comm = hh.comm_init_cluster_info_config(path, rank)
        assert comm
        rc = int(hh.comm_destroy(comm))
        q.put({"rank": rank, "ok": rc == 0, "destroy_rc": rc})
    except Exception:
        q.put({"rank": rank, "ok": False, "error": traceback.format_exc()})


def test_1_get_root_info():
    import hccl4py as h

    _ensure_acl(0)
    root = h.get_root_info()
    assert ctypes.sizeof(root) > 0
    return {"api": "HcclGetRootInfo", "sizeof_rootinfo": ctypes.sizeof(root), "ok": True}


def test_2_comm_config_init():
    import hccl4py as h
    from hccl4py.types import _OFF

    cfg = h.comm_config_init()
    assert ctypes.sizeof(cfg) >= h.HCCL_COMM_CONFIG_SIZE_V10
    raw = bytes(cfg._raw)
    magic = int.from_bytes(raw[_OFF["magic"] : _OFF["magic"] + 4], "little")
    version = int.from_bytes(raw[_OFF["version"] : _OFF["version"] + 4], "little")
    ok = magic == h.HCCL_COMM_CONFIG_MAGIC_WORD and version != 0
    return {
        "api": "HcclCommConfigInit",
        "sizeof_config": ctypes.sizeof(cfg),
        "magic": hex(magic),
        "version": version,
        "ok": ok,
    }


def test_3_and_5_rootinfo_init_destroy():
    import hccl4py as h

    if _device_count() < 2:
        return {"api": "HcclCommInitRootInfoConfig+Destroy", "ok": True, "skipped": True, "error": "NEED>=2"}

    root = h.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from _platform import mp_collect, start_processes

    procs = [ctx.Process(target=_rootinfo_worker, args=(r, root_bytes, q)) for r in range(2)]
    procs = start_processes(procs)
    rows = mp_collect(q, 2, 180, procs)
    ok = all(r.get("ok") for r in rows)
    return {
        "api": "HcclCommInitRootInfoConfig + HcclCommDestroy",
        "ok": ok,
        "ranks": rows,
    }


def test_4_cluster_info_init_destroy():
    from _platform import write_ranktable

    if _device_count() < 2:
        return {"api": "HcclCommInitClusterInfoConfig", "ok": True, "skipped": True, "error": "NEED>=2"}

    fd, path = tempfile.mkstemp(prefix="hccl4py_comm_rt_", suffix=".json")
    os.close(fd)
    try:
        write_ranktable(path, [0, 1])
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        from _platform import mp_collect, start_processes

        procs = [ctx.Process(target=_cluster_worker, args=(r, path, q)) for r in range(2)]
        procs = start_processes(procs)
        rows = mp_collect(q, 2, 240, procs)
        ok = all(r.get("ok") for r in rows)
        return {
            "api": "HcclCommInitClusterInfoConfig + HcclCommDestroy",
            "ok": ok,
            "ranks": rows,
        }
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print("=== COMM DOMAIN SUITE (5 APIs) ===", flush=True)
    n = _device_count()
    print("device_count=%d" % n, flush=True)
    if n < 2:
        print(
            "device_count<2: run single-device APIs; multi-rank cases self-skip",
            flush=True,
        )
    cases = [
        ("1_HcclGetRootInfo", test_1_get_root_info),
        ("2_HcclCommConfigInit", test_2_comm_config_init),
        ("3_5_RootInfoInit_Destroy", test_3_and_5_rootinfo_init_destroy),
        ("4_ClusterInfoInit_Destroy", test_4_cluster_info_init_destroy),
    ]
    failed = []
    for name, fn in cases:
        try:
            detail = fn()
            ok = bool(detail.get("ok"))
            print("[%s] %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
            if not ok:
                failed.append(name)
        except Exception:
            print("[FAIL] %s" % name, flush=True)
            print(traceback.format_exc(), flush=True)
            failed.append(name)
    print(
        "=== SUMMARY domain_apis fail=%d failed=%s ==="
        % (len(failed), failed or "none"),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

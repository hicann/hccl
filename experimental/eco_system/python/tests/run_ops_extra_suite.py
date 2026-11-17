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

"""Extra E2E coverage: multi-item batch, uneven *V counts, optional 4-rank AllReduce.

Env:
  HCCL4PY_NRANKS  default 4-rank when device_count>=4, else 2-rank; need >=2
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

from _platform import make_test_comm_config, mp_collect, safe_ops_skips  # noqa: E402

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _acl(device=0):
    from _platform import ensure_acl

    return ensure_acl(device)



def _ensure_acl_parent():
    from _platform import ensure_acl

    return ensure_acl(0)


def _device_count():
    acl = _ensure_acl_parent()
    n = ctypes.c_uint32()
    assert acl.aclrtGetDeviceCount(ctypes.byref(n)) == 0
    return int(n.value)


def _root_bytes():
    import hccl4py as h

    _ensure_acl_parent()
    root = h.get_root_info()
    return ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))


def _worker(case, rank, nranks, root_bytes, q):
    try:
        import hccl4py as h

        acl = _acl(rank)
        stream = ctypes.c_void_p()
        assert acl.aclrtCreateStream(ctypes.byref(stream)) == 0
        root = h.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        cfg = make_test_comm_config(h)
        comm = h.comm_init_root_info_config(nranks, root, rank, config=cfg)
        dt = h.HcclDataType.FP32
        detail = {}
        ok = False

        if case == "batch_multi_item":
            # 4 items: two SEND + two RECV (two payload slots each way).
            peer = 1 - rank
            count = 2
            nbytes = count * 4
            s1 = (ctypes.c_float * count)(float(rank + 1), float(rank + 1) + 10)
            s2 = (ctypes.c_float * count)(float(rank + 1) + 20, float(rank + 1) + 30)
            r1 = (ctypes.c_float * count)()
            r2 = (ctypes.c_float * count)()
            sb1 = ctypes.c_void_p()
            sb2 = ctypes.c_void_p()
            rb1 = ctypes.c_void_p()
            rb2 = ctypes.c_void_p()
            for buf in (sb1, sb2, rb1, rb2):
                assert acl.aclrtMalloc(ctypes.byref(buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sb1, nbytes, s1, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            assert acl.aclrtMemcpy(sb2, nbytes, s2, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            items = [
                h.make_send_recv_item(h.HcclSendRecvType.SEND, sb1, count, dt, peer),
                h.make_send_recv_item(h.HcclSendRecvType.SEND, sb2, count, dt, peer),
                h.make_send_recv_item(h.HcclSendRecvType.RECV, rb1, count, dt, peer),
                h.make_send_recv_item(h.HcclSendRecvType.RECV, rb2, count, dt, peer),
            ]
            h.batch_send_recv(comm, items, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(r1, nbytes, rb1, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            assert acl.aclrtMemcpy(r2, nbytes, rb2, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            e1 = (float(peer + 1), float(peer + 1) + 10)
            e2 = (float(peer + 1) + 20, float(peer + 1) + 30)
            ok = (
                abs(r1[0] - e1[0]) < 1e-3
                and abs(r1[1] - e1[1]) < 1e-3
                and abs(r2[0] - e2[0]) < 1e-3
                and abs(r2[1] - e2[1]) < 1e-3
            )
            detail = {"r1": [float(r1[0]), float(r1[1])], "r2": [float(r2[0]), float(r2[1])]}

        elif case == "all_gather_v_uneven":
            # rank0 sends 1 elem, rank1 sends 2 elems (nranks must be 2).
            send_count = 1 if rank == 0 else 2
            recv_counts = (ctypes.c_uint64 * 2)(1, 2)
            recv_displs = (ctypes.c_uint64 * 2)(0, 1)
            total = 3
            sh = (ctypes.c_float * send_count)(
                *([float(rank + 1)] * send_count)
            )
            out = (ctypes.c_float * total)()
            sbuf = ctypes.c_void_p()
            rbuf = ctypes.c_void_p()
            assert acl.aclrtMalloc(ctypes.byref(sbuf), send_count * 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(rbuf), total * 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sbuf, send_count * 4, sh, send_count * 4, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            h.all_gather_v(
                comm, sbuf, send_count, rbuf, recv_counts, recv_displs, data_type=dt, stream=stream
            )
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(out, total * 4, rbuf, total * 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            # rank0 value at [0], rank1 value at [1],[2]
            expect = [1.0, 2.0, 2.0]
            ok = all(abs(out[i] - expect[i]) < 1e-3 for i in range(total))
            detail = {"out": [float(out[i]) for i in range(total)], "expect": expect}

        elif case == "reduce_scatter_v_uneven":
            # each rank contributes uneven slices; recv_count at root-local is 1 or 2
            send_counts = (ctypes.c_uint64 * 2)(1, 2)
            send_displs = (ctypes.c_uint64 * 2)(0, 1)
            total = 3
            # fill send buf with rank+1 everywhere
            sh = (ctypes.c_float * total)(*([float(rank + 1)] * total))
            recv_count = 1 if rank == 0 else 2
            rh = (ctypes.c_float * recv_count)()
            sbuf = ctypes.c_void_p()
            rbuf = ctypes.c_void_p()
            assert acl.aclrtMalloc(ctypes.byref(sbuf), total * 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(rbuf), recv_count * 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sbuf, total * 4, sh, total * 4, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            h.reduce_scatter_v(
                comm,
                sbuf,
                send_counts,
                send_displs,
                rbuf,
                recv_count,
                data_type=dt,
                stream=stream,
            )
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(rh, recv_count * 4, rbuf, recv_count * 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            # SUM over ranks of (rank+1) = 3
            expect = 3.0
            ok = all(abs(rh[i] - expect) < 1e-3 for i in range(recv_count))
            detail = {"out": [float(rh[i]) for i in range(recv_count)], "expect": expect}

        elif case == "all_to_all_v_uneven":
            # send 1 to peer0 path / 2 to peer1 path depending on rank layout
            send_counts = (ctypes.c_uint64 * 2)(1, 2) if rank == 0 else (ctypes.c_uint64 * 2)(2, 1)
            # compact layout: displs cumulative
            sc = [int(send_counts[i]) for i in range(2)]
            sdispls = (ctypes.c_uint64 * 2)(0, sc[0])
            recv_counts = (ctypes.c_uint64 * 2)(1, 2) if rank == 0 else (ctypes.c_uint64 * 2)(2, 1)
            # Actually for alltoallv each rank sends sc[j] to rank j; so rank0 sends 1 to r0, 2 to r1
            # rank1 sends 2 to r0, 1 to r1. Then rank0 receives: 1 from r0 + 2 from r1 = counts (1,2)
            # rank1 receives: 2 from r0 + 1 from r1 = counts (2,1)
            rc = [int(recv_counts[i]) for i in range(2)]
            rdispls = (ctypes.c_uint64 * 2)(0, rc[0])
            send_total = sum(sc)
            recv_total = sum(rc)
            sh = (ctypes.c_float * send_total)(*([float(rank + 1)] * send_total))
            rh = (ctypes.c_float * recv_total)()
            sbuf = ctypes.c_void_p()
            rbuf = ctypes.c_void_p()
            assert acl.aclrtMalloc(ctypes.byref(sbuf), send_total * 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(rbuf), recv_total * 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sbuf, send_total * 4, sh, send_total * 4, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            h.all_to_all_v(
                comm,
                sbuf,
                send_counts,
                sdispls,
                dt,
                rbuf,
                recv_counts,
                rdispls,
                dt,
                stream=stream,
            )
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(rh, recv_total * 4, rbuf, recv_total * 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            # rank0 gets 1 from self (=1) then 2 from rank1 (=2,2)
            # rank1 gets 2 from rank0 (=1,1) then 1 from self (=2)
            if rank == 0:
                expect = [1.0, 2.0, 2.0]
            else:
                expect = [1.0, 1.0, 2.0]
            ok = all(abs(rh[i] - expect[i]) < 1e-3 for i in range(recv_total))
            detail = {"out": [float(rh[i]) for i in range(recv_total)], "expect": expect}

        elif case == "all_reduce_nranks":
            count = 8
            nbytes = count * 4
            send_h = (ctypes.c_float * count)(*([float(rank + 1)] * count))
            recv_h = (ctypes.c_float * count)()
            send_buf = ctypes.c_void_p()
            recv_buf = ctypes.c_void_p()
            assert acl.aclrtMalloc(ctypes.byref(send_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(recv_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(send_buf, nbytes, send_h, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            h.all_reduce(comm, send_buf, recv_buf, count, data_type=dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(recv_h, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = float(sum(range(1, nranks + 1)))
            ok = all(abs(recv_h[i] - expect) < 1e-3 for i in range(count))
            detail = {"out0": float(recv_h[0]), "expect": expect, "nranks": nranks}

        else:
            raise RuntimeError("unknown case %s" % case)

        h.comm_destroy(comm)
        q.put({"case": case, "rank": rank, "ok": ok, "detail": detail})
    except Exception:
        q.put({"case": case, "rank": rank, "ok": False, "error": traceback.format_exc()})


def _run(case, nranks):
    root_bytes = _root_bytes()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from _platform import start_processes

    procs = [
        ctx.Process(target=_worker, args=(case, r, nranks, root_bytes, q))
        for r in range(nranks)
    ]
    procs = start_processes(procs)
    results = mp_collect(q, nranks, 180, procs, join_timeout=30)
    ok = len(results) == nranks and all(r.get("ok") for r in results)
    print("[%s] %s nranks=%d" % ("PASS" if ok else "FAIL", case, nranks))
    for r in results:
        if not r.get("ok"):
            print("  rank", r.get("rank"), r.get("error") or r.get("detail"))
        else:
            print("  rank", r.get("rank"), r.get("detail"))
    return ok


def main():
    n = _device_count()
    print("device_count=%d" % n)
    if n < 2:
        print("NEED >=2 devices")
        return 2

    skips = safe_ops_skips()
    failed = []
    skipped = []
    passed = []

    # Always: multi-item batch on 2 ranks
    if not _run("batch_multi_item", 2):
        failed.append("batch_multi_item")
    else:
        passed.append("batch_multi_item")

    # Uneven *V on 2 ranks (respect A5 skip table)
    for case, skip_key in (
        ("all_gather_v_uneven", "all_gather_v"),
        ("reduce_scatter_v_uneven", "reduce_scatter_v"),
        ("all_to_all_v_uneven", "all_to_all_v"),
    ):
        reason = skips.get(skip_key)
        if reason:
            print("[SKIP] %s %s" % (case, reason))
            skipped.append(case)
            continue
        if not _run(case, 2):
            failed.append(case)
        else:
            passed.append(case)

    # Multi-rank AllReduce when enough devices
    want = int(os.environ.get("HCCL4PY_NRANKS", "0") or "0")
    if want <= 0:
        want = 4 if n >= 4 else 2
    want = min(want, n)
    if want >= 3:
        if not _run("all_reduce_nranks", want):
            failed.append("all_reduce_nranks_%d" % want)
        else:
            passed.append("all_reduce_nranks_%d" % want)
    else:
        print("[SKIP] all_reduce_nranks need >=3 devices for non-2 coverage (got %d)" % n)
        skipped.append("all_reduce_nranks")

    print(
        "=== SUMMARY passed=%d skipped=%d failed=%s skipped_ops=%s ==="
        % (len(passed), len(skipped), failed or "none", skipped or "none")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

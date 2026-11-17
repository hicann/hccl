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

"""Python vs C (ctypes-direct) overhead microbench for ALL 14 task-book ops.

Task-book test_cases: 性能验证 / 所有算子 — 相同样例对比 C 接口，额外耗时 ≤10µs（±10%）。
Each op:
  - C path: raw lib.Hccl* via ctypes (no hccl4py wrapper body)
  - Python path: hccl4py.ops.*
Reports per-op mean and (py - c); FAIL if any op overhead > limit*(1+jitter).
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import statistics
import sys
import time

HERE = os.environ.get(
    "HCCL4PY_PYTHON",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
WARMUP = int(os.environ.get("HCCL4PY_PERF_WARMUP", "20"))
ITERS = int(os.environ.get("HCCL4PY_PERF_ITERS", "100"))
COUNT = int(os.environ.get("HCCL4PY_PERF_COUNT", "8"))
ROUNDS = int(os.environ.get("HCCL4PY_PERF_ROUNDS", "3"))
LIMIT_US = float(os.environ.get("HCCL4PY_PERF_LIMIT_US", "10.0"))
JITTER = float(os.environ.get("HCCL4PY_PERF_JITTER", "0.10"))
NRANKS = 2  # task book: 2-card

# Order matches task_book test_cases/README.md
OPS = [
    "broadcast",
    "all_gather",
    "all_gather_v",
    "reduce",
    "all_reduce",
    "scatter",
    "reduce_scatter",
    "reduce_scatter_v",
    "all_to_all",
    "all_to_all_v",
    "all_to_all_vc",
    "send_recv",
    "batch_send_recv",
]


def _acl(device=0):
    from _platform import ensure_acl

    return ensure_acl(device)



def _timed_loop(fn, stream, acl, warmup, iters):
    """Host-side enqueue overhead only: sync is outside the timed window.

    Warmup + a pre-window sync drain the device so the measured loop is
    dominated by Python/ctypes submit cost, not device execution. A post-
    window sync still waits for completion (correctness) but is not billed
    into the per-iter µs used for the ≤10µs wrapper budget.
    """
    for _ in range(warmup):
        fn()
    assert acl.aclrtSynchronizeStream(stream) == 0
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host_us = (time.perf_counter() - t0) * 1e6 / float(iters)
    assert acl.aclrtSynchronizeStream(stream) == 0
    return host_us


def _bind_c(lib, HcclComm, aclrtStream):
    """Bind all Hccl* used by C path."""
    specs = {
        "HcclAllReduce": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            HcclComm,
            aclrtStream,
        ],
        "HcclReduce": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint32,
            HcclComm,
            aclrtStream,
        ],
        "HcclBroadcast": [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_uint32,
            HcclComm,
            aclrtStream,
        ],
        "HcclAllGather": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            HcclComm,
            aclrtStream,
        ],
        "HcclAllGatherV": [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            HcclComm,
            aclrtStream,
        ],
        "HcclScatter": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_uint32,
            HcclComm,
            aclrtStream,
        ],
        "HcclReduceScatter": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            HcclComm,
            aclrtStream,
        ],
        "HcclReduceScatterV": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            HcclComm,
            aclrtStream,
        ],
        "HcclAlltoAll": [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            HcclComm,
            aclrtStream,
        ],
        "HcclAlltoAllV": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            HcclComm,
            aclrtStream,
        ],
        "HcclAlltoAllVC": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            HcclComm,
            aclrtStream,
        ],
        "HcclSend": [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_uint32,
            HcclComm,
            aclrtStream,
        ],
        "HcclRecv": [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_uint32,
            HcclComm,
            aclrtStream,
        ],
        "HcclBatchSendRecv": [
            ctypes.c_void_p,
            ctypes.c_uint32,
            HcclComm,
            aclrtStream,
        ],
    }
    for name, args in specs.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_int


def _check(rc, where):
    if rc != 0:
        raise RuntimeError("%s C path rc=%d" % (where, rc))


def _make_pair_fns(h, lib, comm, comm_handle, stream, stream_p, rank, nranks, acl):
    """Return {op: (c_call, py_call)} with buffers allocated."""
    from hccl4py.types import HcclSendRecvItem

    dtype = int(h.HcclDataType.FP32)
    rop = int(h.HcclReduceOp.SUM)
    count_c = ctypes.c_uint64(COUNT)
    nbytes = COUNT * 4

    send_buf = ctypes.c_void_p()
    recv_buf = ctypes.c_void_p()
    assert acl.aclrtMalloc(ctypes.byref(send_buf), max(nbytes, nranks * 4), ACL_MEM_MALLOC_HUGE_FIRST) == 0
    assert acl.aclrtMalloc(ctypes.byref(recv_buf), max(nbytes, nranks * 4), ACL_MEM_MALLOC_HUGE_FIRST) == 0
    host = (ctypes.c_float * COUNT)(*([float(rank + 1)] * COUNT))
    assert acl.aclrtMemcpy(send_buf, nbytes, host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
    assert acl.aclrtMemset(recv_buf, max(nbytes, nranks * 4), 0, max(nbytes, nranks * 4)) == 0

    # scatter / reduce_scatter sized buffers
    sc_n = nranks
    sc_bytes = sc_n * 4
    sc_send = ctypes.c_void_p()
    sc_recv = ctypes.c_void_p()
    assert acl.aclrtMalloc(ctypes.byref(sc_send), sc_bytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
    assert acl.aclrtMalloc(ctypes.byref(sc_recv), 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
    sc_host = (ctypes.c_float * sc_n)(*([float(rank + 1)] * sc_n))
    assert acl.aclrtMemcpy(sc_send, sc_bytes, sc_host, sc_bytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0

    # all_gather 1-elem
    ag_send = ctypes.c_void_p()
    ag_recv = ctypes.c_void_p()
    assert acl.aclrtMalloc(ctypes.byref(ag_send), 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
    assert acl.aclrtMalloc(ctypes.byref(ag_recv), nranks * 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
    one = (ctypes.c_float * 1)(float(rank + 1))
    assert acl.aclrtMemcpy(ag_send, 4, one, 4, ACL_MEMCPY_HOST_TO_DEVICE) == 0

    recv_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
    recv_displs = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
    send_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
    send_displs = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
    matrix = (ctypes.c_uint64 * (nranks * nranks))(*([1] * (nranks * nranks)))

    # alltoall buffers
    a2a_n = nranks
    a2a_bytes = a2a_n * 4
    a2a_s = ctypes.c_void_p()
    a2a_r = ctypes.c_void_p()
    assert acl.aclrtMalloc(ctypes.byref(a2a_s), a2a_bytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
    assert acl.aclrtMalloc(ctypes.byref(a2a_r), a2a_bytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
    a2a_h = (ctypes.c_float * a2a_n)(*([float(rank + 1)] * a2a_n))
    assert acl.aclrtMemcpy(a2a_s, a2a_bytes, a2a_h, a2a_bytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0

    peer = 1 - rank
    next_rank = (rank + 1) % nranks
    prev_rank = (rank - 1 + nranks) % nranks

    # BatchSendRecv items (reuse send/recv bufs)
    item_send = HcclSendRecvItem()
    item_send.sendRecvType = int(h.HcclSendRecvType.SEND)
    item_send.buf = send_buf
    item_send.count = COUNT
    item_send.dataType = dtype
    item_send.remoteRank = next_rank
    item_recv = HcclSendRecvItem()
    item_recv.sendRecvType = int(h.HcclSendRecvType.RECV)
    item_recv.buf = recv_buf
    item_recv.count = COUNT
    item_recv.dataType = dtype
    item_recv.remoteRank = prev_rank
    items_arr = (HcclSendRecvItem * 2)(item_send, item_recv)

    pairs = {}

    def add(name, c_fn, py_fn):
        pairs[name] = (c_fn, py_fn)

    add(
        "all_reduce",
        lambda: _check(
            lib.HcclAllReduce(send_buf, recv_buf, count_c, dtype, rop, comm, stream), "AllReduce"
        ),
        lambda: h.all_reduce(
            comm_handle, send_buf, recv_buf, COUNT, data_type=dtype, reduce_op=rop, stream=stream_p.value
        ),
    )
    add(
        "reduce",
        lambda: _check(
            lib.HcclReduce(send_buf, recv_buf, count_c, dtype, rop, 0, comm, stream), "Reduce"
        ),
        lambda: h.reduce(
            comm_handle,
            send_buf,
            recv_buf,
            COUNT,
            data_type=dtype,
            reduce_op=rop,
            root=0,
            stream=stream_p.value,
        ),
    )
    add(
        "broadcast",
        lambda: _check(lib.HcclBroadcast(send_buf, count_c, dtype, 0, comm, stream), "Broadcast"),
        lambda: h.broadcast(comm_handle, send_buf, COUNT, data_type=dtype, root=0, stream=stream_p.value),
    )
    add(
        "all_gather",
        lambda: _check(
            lib.HcclAllGather(ag_send, ag_recv, ctypes.c_uint64(1), dtype, comm, stream), "AllGather"
        ),
        lambda: h.all_gather(comm_handle, ag_send, ag_recv, 1, data_type=dtype, stream=stream_p.value),
    )
    add(
        "all_gather_v",
        lambda: _check(
            lib.HcclAllGatherV(
                ag_send,
                ctypes.c_uint64(1),
                ag_recv,
                recv_counts,
                recv_displs,
                dtype,
                comm,
                stream,
            ),
            "AllGatherV",
        ),
        lambda: h.all_gather_v(
            comm_handle,
            ag_send,
            1,
            ag_recv,
            recv_counts,
            recv_displs,
            data_type=dtype,
            stream=stream_p.value,
        ),
    )
    add(
        "scatter",
        lambda: _check(
            lib.HcclScatter(sc_send, sc_recv, ctypes.c_uint64(1), dtype, 0, comm, stream), "Scatter"
        ),
        lambda: h.scatter(
            comm_handle, sc_send, sc_recv, 1, data_type=dtype, root=0, stream=stream_p.value
        ),
    )
    add(
        "reduce_scatter",
        lambda: _check(
            lib.HcclReduceScatter(sc_send, sc_recv, ctypes.c_uint64(1), dtype, rop, comm, stream),
            "ReduceScatter",
        ),
        lambda: h.reduce_scatter(
            comm_handle, sc_send, sc_recv, 1, data_type=dtype, reduce_op=rop, stream=stream_p.value
        ),
    )
    add(
        "reduce_scatter_v",
        lambda: _check(
            lib.HcclReduceScatterV(
                sc_send,
                send_counts,
                send_displs,
                sc_recv,
                ctypes.c_uint64(1),
                dtype,
                rop,
                comm,
                stream,
            ),
            "ReduceScatterV",
        ),
        lambda: h.reduce_scatter_v(
            comm_handle,
            sc_send,
            send_counts,
            send_displs,
            sc_recv,
            1,
            data_type=dtype,
            reduce_op=rop,
            stream=stream_p.value,
        ),
    )
    add(
        "all_to_all",
        lambda: _check(
            lib.HcclAlltoAll(
                a2a_s,
                ctypes.c_uint64(1),
                dtype,
                a2a_r,
                ctypes.c_uint64(1),
                dtype,
                comm,
                stream,
            ),
            "AlltoAll",
        ),
        lambda: h.all_to_all(
            comm_handle, a2a_s, 1, dtype, a2a_r, 1, dtype, stream=stream_p.value
        ),
    )
    add(
        "all_to_all_v",
        lambda: _check(
            lib.HcclAlltoAllV(
                a2a_s,
                send_counts,
                send_displs,
                dtype,
                a2a_r,
                recv_counts,
                recv_displs,
                dtype,
                comm,
                stream,
            ),
            "AlltoAllV",
        ),
        lambda: h.all_to_all_v(
            comm_handle,
            a2a_s,
            send_counts,
            send_displs,
            dtype,
            a2a_r,
            recv_counts,
            recv_displs,
            dtype,
            stream=stream_p.value,
        ),
    )
    add(
        "all_to_all_vc",
        lambda: _check(
            lib.HcclAlltoAllVC(a2a_s, matrix, dtype, a2a_r, dtype, comm, stream), "AlltoAllVC"
        ),
        lambda: h.all_to_all_vc(comm_handle, a2a_s, matrix, dtype, a2a_r, dtype, stream=stream_p.value),
    )

    def c_send_recv():
        if rank == 0:
            _check(lib.HcclSend(send_buf, count_c, dtype, peer, comm, stream), "Send")
            _check(lib.HcclRecv(recv_buf, count_c, dtype, peer, comm, stream), "Recv")
        else:
            _check(lib.HcclRecv(recv_buf, count_c, dtype, peer, comm, stream), "Recv")
            _check(lib.HcclSend(send_buf, count_c, dtype, peer, comm, stream), "Send")

    def py_send_recv():
        if rank == 0:
            h.send(comm_handle, send_buf, COUNT, dtype, dest_rank=peer, stream=stream_p.value)
            h.recv(comm_handle, recv_buf, COUNT, dtype, src_rank=peer, stream=stream_p.value)
        else:
            h.recv(comm_handle, recv_buf, COUNT, dtype, src_rank=peer, stream=stream_p.value)
            h.send(comm_handle, send_buf, COUNT, dtype, dest_rank=peer, stream=stream_p.value)

    add("send_recv", c_send_recv, py_send_recv)

    items_ptr = ctypes.cast(items_arr, ctypes.POINTER(HcclSendRecvItem))

    add(
        "batch_send_recv",
        lambda: _check(
            lib.HcclBatchSendRecv(items_ptr, ctypes.c_uint32(2), comm, stream), "BatchSendRecv"
        ),
        # Reuse prebuilt items_arr (same as C path) — do not rebuild each iter.
        lambda: h.batch_send_recv(comm_handle, items_arr, stream=stream_p.value),
    )
    return pairs


def _worker(rank, nranks, root_bytes, q):
    try:
        import hccl4py as h
        from hccl4py._lib import get_hccl
        from hccl4py.types import HcclComm, aclrtStream

        acl = _acl(rank)
        assert acl.aclrtSetDevice(rank) == 0
        stream_p = ctypes.c_void_p()
        assert acl.aclrtCreateStream(ctypes.byref(stream_p)) == 0

        root = h.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        from _platform import make_test_comm_config

        cfg = make_test_comm_config(h)
        comm_handle = h.comm_init_root_info_config(nranks, root, rank, config=cfg)
        comm = HcclComm(comm_handle)
        stream = aclrtStream(stream_p.value)

        lib = get_hccl()
        _bind_c(lib, HcclComm, aclrtStream)
        # also bind python wrappers once
        import hccl4py.ops as ops

        ops._bind()

        pairs = _make_pair_fns(h, lib, comm, comm_handle, stream, stream_p, rank, nranks, acl)
        op_results = {}
        for op in OPS:
            c_fn, py_fn = pairs[op]
            c_us = []
            py_us = []
            for _ in range(ROUNDS):
                c_us.append(_timed_loop(c_fn, stream_p, acl, WARMUP, ITERS))
                py_us.append(_timed_loop(py_fn, stream_p, acl, WARMUP, ITERS))
                py_us.append(_timed_loop(py_fn, stream_p, acl, WARMUP, ITERS))
                c_us.append(_timed_loop(c_fn, stream_p, acl, WARMUP, ITERS))
            op_results[op] = {
                "c_mean": statistics.mean(c_us),
                "py_mean": statistics.mean(py_us),
                "c_stdev": statistics.pstdev(c_us) if len(c_us) > 1 else 0.0,
                "py_stdev": statistics.pstdev(py_us) if len(py_us) > 1 else 0.0,
                "overhead": statistics.mean(py_us) - statistics.mean(c_us),
            }
            print(
                "[RANK %d] %-18s C=%.3fus Py=%.3fus overhead=%.3fus"
                % (
                    rank,
                    op,
                    op_results[op]["c_mean"],
                    op_results[op]["py_mean"],
                    op_results[op]["overhead"],
                ),
                flush=True,
            )

        h.comm_destroy(comm_handle)
        q.put({"rank": rank, "ok": True, "ops": op_results})
    except Exception as e:
        import traceback

        q.put({"rank": rank, "ok": False, "error": repr(e), "tb": traceback.format_exc()})


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    acl = _acl(0)
    n = ctypes.c_uint32()
    assert acl.aclrtGetDeviceCount(ctypes.byref(n)) == 0
    ndev = int(n.value)
    print(
        "[INFO] devices=%d use_nranks=%d warmup=%d iters=%d count=%d rounds=%d limit=%.1fus ops=%d"
        % (ndev, NRANKS, WARMUP, ITERS, COUNT, ROUNDS, LIMIT_US, len(OPS)),
        flush=True,
    )
    if ndev < NRANKS:
        print("[FAIL] need device_count>=%d" % NRANKS)
        return 1

    import hccl4py as h

    assert acl.aclrtSetDevice(0) == 0
    root = h.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from _platform import mp_collect, start_processes

    procs = [ctx.Process(target=_worker, args=(r, NRANKS, root_bytes, q)) for r in range(NRANKS)]
    procs = start_processes(procs)
    results = mp_collect(q, NRANKS, 3600, procs)

    failed = [r for r in results if not r.get("ok")]
    if failed:
        for r in failed:
            print("[FAIL] rank=%s %s" % (r.get("rank"), r.get("error")))
            print(r.get("tb", ""))
        return 1

    r0 = sorted(results, key=lambda x: x["rank"])[0]
    limit_hi = LIMIT_US * (1.0 + JITTER)
    bad = []
    print("==== PER-OP OVERHEAD (rank0) ====", flush=True)
    for op in OPS:
        row = r0["ops"][op]
        oh = row["overhead"]
        status = "PASS" if oh <= limit_hi else "FAIL"
        print(
            "[%s] %-18s C=%.3fus Py=%.3fus overhead=%.3fus (limit %.3f)"
            % (status, op, row["c_mean"], row["py_mean"], oh, limit_hi),
            flush=True,
        )
        if oh > limit_hi:
            bad.append(op)

    print(
        "=== SUMMARY ops=%d fail=%d failed=%s limit=%.3fus ==="
        % (len(OPS), len(bad), bad or "none", limit_hi),
        flush=True,
    )
    return 0 if not bad else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

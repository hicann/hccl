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

"""Correctness of all 14 ops (13 entries; send/recv merged) on single-host multi-device."""

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

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _acl():
    from _platform import bind_acl

    return bind_acl()



_ACL_READY = False


def _ensure_acl_parent():
    global _ACL_READY
    acl = _acl()
    if not _ACL_READY:
        rc = acl.aclInit(None)
        # 0 = ok; non-zero may mean already initialized — only accept if SetDevice works.
        if rc != 0:
            set_rc = acl.aclrtSetDevice(0)
            if set_rc != 0:
                raise RuntimeError(
                    "aclInit failed rc=%s and aclrtSetDevice(0) rc=%s" % (rc, set_rc)
                )
        else:
            assert acl.aclrtSetDevice(0) == 0
        _ACL_READY = True
    return acl


def _device_count():
    acl = _ensure_acl_parent()
    n = ctypes.c_uint32()
    assert acl.aclrtGetDeviceCount(ctypes.byref(n)) == 0
    return int(n.value)


def _mk_root_bytes():
    import hccl4py as h

    _ensure_acl_parent()
    root = h.get_root_info()
    return ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))


def _worker(op, rank, nranks, root_bytes, q):
    try:
        import hccl4py as h

        from _platform import ensure_acl, check_acl

        acl = ensure_acl(rank)
        stream = ctypes.c_void_p()
        assert acl.aclrtCreateStream(ctypes.byref(stream)) == 0

        root = h.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        from _platform import make_test_comm_config

        cfg = make_test_comm_config(h)
        comm = h.comm_init_root_info_config(nranks, root, rank, config=cfg)

        count = 8
        nbytes = count * 4
        dt = h.HcclDataType.FP32
        send_h = (ctypes.c_float * count)(*[float(rank + 1) for _ in range(count)])
        recv_h = (ctypes.c_float * count)(*([0.0] * count))
        send_buf = ctypes.c_void_p()
        recv_buf = ctypes.c_void_p()
        assert acl.aclrtMalloc(ctypes.byref(send_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
        assert acl.aclrtMalloc(ctypes.byref(recv_buf), nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
        assert acl.aclrtMemcpy(send_buf, nbytes, send_h, nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
        assert acl.aclrtMemset(recv_buf, nbytes, 0, nbytes) == 0

        ok = False
        detail = {}

        if op == "all_reduce":
            h.all_reduce(comm, send_buf, recv_buf, count, data_type=dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(recv_h, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = float(sum(range(1, nranks + 1)))
            ok = all(abs(recv_h[i] - expect) < 1e-3 for i in range(count))
            detail = {"out0": float(recv_h[0]), "expect": expect}

        elif op == "broadcast":
            # root=0 broadcasts its buffer; non-root uses recv as inout
            buf = send_buf if rank == 0 else recv_buf
            h.broadcast(comm, buf, count, data_type=dt, root=0, stream=stream)
            check_buf = buf
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(recv_h, nbytes, check_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = 1.0  # rank0 value
            ok = all(abs(recv_h[i] - expect) < 1e-3 for i in range(count))
            detail = {"out0": float(recv_h[0]), "expect": expect}

        elif op == "reduce":
            h.reduce(comm, send_buf, recv_buf, count, data_type=dt, root=0, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(recv_h, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = float(sum(range(1, nranks + 1)))
            if rank == 0:
                ok = all(abs(recv_h[i] - expect) < 1e-3 for i in range(count))
            else:
                ok = True  # non-root result undefined/unused
            detail = {"out0": float(recv_h[0]), "expect": expect, "is_root": rank == 0}

        elif op == "all_gather":
            # each rank sends 1 element; recv has nranks elements
            send_count = 1
            send_nbytes = 4
            recv_count = nranks
            recv_nbytes = recv_count * 4
            one = (ctypes.c_float * 1)(float(rank + 1))
            out = (ctypes.c_float * recv_count)()
            sbuf = ctypes.c_void_p()
            rbuf = ctypes.c_void_p()
            assert acl.aclrtMalloc(ctypes.byref(sbuf), send_nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(rbuf), recv_nbytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sbuf, send_nbytes, one, send_nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            h.all_gather(comm, sbuf, rbuf, send_count, data_type=dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(out, recv_nbytes, rbuf, recv_nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = [float(i + 1) for i in range(nranks)]
            ok = all(abs(out[i] - expect[i]) < 1e-3 for i in range(nranks))
            detail = {"out": [float(out[i]) for i in range(nranks)], "expect": expect}

        elif op == "reduce_scatter":
            # send nranks elems, recv 1 elem = sum of that slice across ranks
            # each rank send[i] = (rank+1); after reduce_scatter recv = sum_r (r+1) = same for all
            send_n = nranks
            send_bytes = send_n * 4
            recv_n = 1
            sh = (ctypes.c_float * send_n)(*[float(rank + 1) for _ in range(send_n)])
            rh = (ctypes.c_float * 1)()
            sbuf = ctypes.c_void_p()
            rbuf = ctypes.c_void_p()
            assert acl.aclrtMalloc(ctypes.byref(sbuf), send_bytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(rbuf), 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sbuf, send_bytes, sh, send_bytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            h.reduce_scatter(comm, sbuf, rbuf, recv_n, data_type=dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(rh, 4, rbuf, 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = float(sum(range(1, nranks + 1)))
            ok = abs(rh[0] - expect) < 1e-3
            detail = {"out0": float(rh[0]), "expect": expect}

        elif op == "scatter":
            # root=0 scatters nranks floats: [10,20,...]; each rank gets one.
            # A5 AICPU path rejects nullptr sendBuf even on non-root (param invalid).
            send_n = nranks
            send_bytes = send_n * 4
            sbuf = ctypes.c_void_p()
            assert acl.aclrtMalloc(ctypes.byref(sbuf), send_bytes, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            if rank == 0:
                sh = (ctypes.c_float * send_n)(*[float((i + 1) * 10) for i in range(send_n)])
                assert acl.aclrtMemcpy(sbuf, send_bytes, sh, send_bytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            else:
                assert acl.aclrtMemset(sbuf, send_bytes, 0, send_bytes) == 0
            rbuf = ctypes.c_void_p()
            rh = (ctypes.c_float * 1)()
            assert acl.aclrtMalloc(ctypes.byref(rbuf), 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            h.scatter(comm, sbuf, rbuf, 1, data_type=dt, root=0, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(rh, 4, rbuf, 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = float((rank + 1) * 10)
            ok = abs(rh[0] - expect) < 1e-3
            detail = {"out0": float(rh[0]), "expect": expect}

        elif op == "all_to_all":
            # each rank sends 1 float per peer (= rank+1), receives nranks floats
            send_count = 1
            send_n = nranks * send_count
            sh = (ctypes.c_float * send_n)(*[float(rank + 1) for _ in range(send_n)])
            rh = (ctypes.c_float * send_n)()
            sbuf = ctypes.c_void_p()
            rbuf = ctypes.c_void_p()
            nb = send_n * 4
            assert acl.aclrtMalloc(ctypes.byref(sbuf), nb, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(rbuf), nb, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sbuf, nb, sh, nb, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            h.all_to_all(comm, sbuf, send_count, dt, rbuf, send_count, dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(rh, nb, rbuf, nb, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = [float(i + 1) for i in range(nranks)]
            ok = all(abs(rh[i] - expect[i]) < 1e-3 for i in range(nranks))
            detail = {"out": [float(rh[i]) for i in range(nranks)], "expect": expect}

        elif op == "send_recv":
            # Same ring as batch_send_recv: SEND to next, RECV from prev.
            next_rank = (rank + 1) % nranks
            prev_rank = (rank - 1 + nranks) % nranks
            if rank % 2 == 0:
                h.send(comm, send_buf, count, dt, dest_rank=next_rank, stream=stream)
                h.recv(comm, recv_buf, count, dt, src_rank=prev_rank, stream=stream)
            else:
                h.recv(comm, recv_buf, count, dt, src_rank=prev_rank, stream=stream)
                h.send(comm, send_buf, count, dt, dest_rank=next_rank, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(recv_h, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = float(prev_rank + 1)
            ok = all(abs(recv_h[i] - expect) < 1e-3 for i in range(count))
            detail = {"out0": float(recv_h[0]), "expect": expect}

        elif op == "batch_send_recv":
            # Match official ring sample: SEND to next, RECV from prev.
            next_rank = (rank + 1) % nranks
            prev_rank = (rank - 1 + nranks) % nranks
            items = [
                h.make_send_recv_item(h.HcclSendRecvType.SEND, send_buf, count, dt, next_rank),
                h.make_send_recv_item(h.HcclSendRecvType.RECV, recv_buf, count, dt, prev_rank),
            ]
            h.batch_send_recv(comm, items, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(recv_h, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = float(prev_rank + 1)  # recv from prev, whose value is prev+1
            ok = all(abs(recv_h[i] - expect) < 1e-3 for i in range(count))
            detail = {"out0": float(recv_h[0]), "expect": expect}

        elif op == "all_gather_v":
            # equal counts path: each sends 1; counts/displs are host uint64 (official hccl_test)
            send_count = 1
            recv_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
            recv_displs = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
            one = (ctypes.c_float * 1)(float(rank + 1))
            out = (ctypes.c_float * nranks)()
            sbuf = ctypes.c_void_p()
            rbuf = ctypes.c_void_p()
            assert acl.aclrtMalloc(ctypes.byref(sbuf), 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(rbuf), nranks * 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sbuf, 4, one, 4, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            h.all_gather_v(
                comm,
                sbuf,
                send_count,
                rbuf,
                recv_counts,
                recv_displs,
                data_type=dt,
                stream=stream,
            )
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(out, nranks * 4, rbuf, nranks * 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = [float(i + 1) for i in range(nranks)]
            ok = all(abs(out[i] - expect[i]) < 1e-3 for i in range(nranks))
            detail = {"out": [float(out[i]) for i in range(nranks)], "expect": expect}

        elif op == "reduce_scatter_v":
            send_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
            send_displs = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
            sh = (ctypes.c_float * nranks)(*[float(rank + 1) for _ in range(nranks)])
            rh = (ctypes.c_float * 1)()
            sbuf = ctypes.c_void_p()
            rbuf = ctypes.c_void_p()
            assert acl.aclrtMalloc(ctypes.byref(sbuf), nranks * 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(rbuf), 4, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sbuf, nranks * 4, sh, nranks * 4, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            h.reduce_scatter_v(
                comm,
                sbuf,
                send_counts,
                send_displs,
                rbuf,
                1,
                data_type=dt,
                stream=stream,
            )
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(rh, 4, rbuf, 4, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = float(sum(range(1, nranks + 1)))
            ok = abs(rh[0] - expect) < 1e-3
            detail = {"out0": float(rh[0]), "expect": expect}

        elif op == "all_to_all_v":
            send_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
            sdispls = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
            recv_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
            rdispls = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
            sh = (ctypes.c_float * nranks)(*[float(rank + 1) for _ in range(nranks)])
            rh = (ctypes.c_float * nranks)()
            sbuf = ctypes.c_void_p()
            rbuf = ctypes.c_void_p()
            nb = nranks * 4
            assert acl.aclrtMalloc(ctypes.byref(sbuf), nb, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(rbuf), nb, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sbuf, nb, sh, nb, ACL_MEMCPY_HOST_TO_DEVICE) == 0
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
            assert acl.aclrtMemcpy(rh, nb, rbuf, nb, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = [float(i + 1) for i in range(nranks)]
            ok = all(abs(rh[i] - expect[i]) < 1e-3 for i in range(nranks))
            detail = {"out": [float(rh[i]) for i in range(nranks)], "expect": expect}

        elif op == "all_to_all_vc":
            # sendCountMatrix[nranks][nranks], all 1s (host uint64)
            matrix = (ctypes.c_uint64 * (nranks * nranks))(*([1] * (nranks * nranks)))
            sh = (ctypes.c_float * nranks)(*[float(rank + 1) for _ in range(nranks)])
            rh = (ctypes.c_float * nranks)()
            sbuf = ctypes.c_void_p()
            rbuf = ctypes.c_void_p()
            nb = nranks * 4
            assert acl.aclrtMalloc(ctypes.byref(sbuf), nb, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMalloc(ctypes.byref(rbuf), nb, ACL_MEM_MALLOC_HUGE_FIRST) == 0
            assert acl.aclrtMemcpy(sbuf, nb, sh, nb, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            h.all_to_all_vc(comm, sbuf, matrix, dt, rbuf, dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            assert acl.aclrtMemcpy(rh, nb, rbuf, nb, ACL_MEMCPY_DEVICE_TO_HOST) == 0
            expect = [float(i + 1) for i in range(nranks)]
            ok = all(abs(rh[i] - expect[i]) < 1e-3 for i in range(nranks))
            detail = {"out": [float(rh[i]) for i in range(nranks)], "expect": expect}

        else:
            raise RuntimeError("unknown op %s" % op)

        destroy_rc = h.comm_destroy(comm)
        if int(destroy_rc) != 0:
            ok = False
            detail = dict(detail or {})
            detail["destroy_rc"] = int(destroy_rc)
        q.put({"op": op, "rank": rank, "ok": ok, "detail": detail})
    except Exception:
        q.put({"op": op, "rank": rank, "ok": False, "error": traceback.format_exc()})


def run_op(op, nranks=2):
    root_bytes = _mk_root_bytes()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from _platform import start_processes

    procs = [
        ctx.Process(target=_worker, args=(op, r, nranks, root_bytes, q))
        for r in range(nranks)
    ]
    procs = start_processes(procs)
    results = []
    timed_out = False
    try:
        for _ in range(nranks):
            results.append(q.get(timeout=120))
    except Exception as e:
        timed_out = True
        print("[%s] %s parent timeout/empty: %s" % ("FAIL", op, e))
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
    if timed_out or len(results) < nranks:
        return False
    ok = all(r.get("ok") for r in results)
    status = "PASS" if ok else "FAIL"
    print("[%s] %s" % (status, op))
    for r in results:
        if not r.get("ok"):
            print("  rank", r.get("rank"), r.get("error") or r.get("detail"))
        else:
            print("  rank", r.get("rank"), r.get("detail"))
    return ok


OPS = [
    "all_reduce",
    "broadcast",
    "reduce",
    "all_gather",
    "reduce_scatter",
    "scatter",
    "all_to_all",
    "send_recv",
    "batch_send_recv",
    "all_gather_v",
    "reduce_scatter_v",
    "all_to_all_v",
    "all_to_all_vc",
]


def main():
    # Atlas 350 / Ascend950: *V needs CCU schedule (CANN docs + hccl_test -a ccu_sched)
    n = _device_count()
    print("device_count=%d" % n)
    if n < 2:
        print("NEED >=2 devices")
        return 2
    from _platform import safe_ops_skips

    nranks = int(os.environ.get("HCCL4PY_NRANKS", "2") or "2")
    nranks = max(2, min(nranks, n))
    print("nranks=%d (override with HCCL4PY_NRANKS)" % nranks)
    skips = safe_ops_skips()
    failed = []
    skipped = []
    passed = []
    for op in OPS:
        reason = skips.get(op)
        if reason:
            print("[SKIP] %s %s" % (op, reason))
            skipped.append(op)
            continue
        try:
            if not run_op(op, nranks):
                failed.append(op)
            else:
                passed.append(op)
        except Exception:
            print("[FAIL] %s parent: %s" % (op, traceback.format_exc()))
            failed.append(op)
    print(
        "=== SUMMARY ops_entries=%d (14 C APIs; send/recv merged) passed=%d skipped=%d failed=%s skipped_ops=%s ==="
        % (len(OPS), len(passed), len(skipped), failed or "none", skipped or "none")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

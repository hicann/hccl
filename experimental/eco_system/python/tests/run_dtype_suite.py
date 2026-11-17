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

"""Multi-dtype coverage for all 14 hccl4py ops (RootInfo, 2 ranks).

``DTYPE_SPECS`` covers: INT8/16/32/64, UINT8/16/32/64, FP16/32/64, BFP16.
Reduce-family skips UINT8 (official AllReduce/Reduce on A5 has no UINT8);
chip-specific skips come from ``_platform.dtype_skip_cases``.
Unsupported runtime cases are reported as SKIP (NOT_SUPPORT), not FAIL.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import struct
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

DTYPE_SPECS = [
    ("INT8", "INT8", 1, "b"),
    ("INT16", "INT16", 2, "h"),
    ("INT32", "INT32", 4, "i"),
    ("INT64", "INT64", 8, "q"),
    ("UINT8", "UINT8", 1, "B"),
    ("UINT16", "UINT16", 2, "H"),
    ("UINT32", "UINT32", 4, "I"),
    ("UINT64", "UINT64", 8, "Q"),
    ("FP16", "FP16", 2, "FP16"),
    ("FP32", "FP32", 4, "f"),
    ("FP64", "FP64", 8, "d"),
    ("BFP16", "BFP16", 2, "BFP16"),
]

# Task-book 14 APIs: Send+Recv counted as send_recv; BatchSendRecv separate.
OPS = [
    "all_reduce",
    "reduce",
    "broadcast",
    "all_gather",
    "all_gather_v",
    "scatter",
    "reduce_scatter",
    "reduce_scatter_v",
    "all_to_all",
    "all_to_all_v",
    "all_to_all_vc",
    "send_recv",
    "batch_send_recv",
]

# Chip-aware skips (A2/A3/A5 official tables). Built at main() after detect.
SKIP_CASES = {}


def _acl(device=0):
    from _platform import ensure_acl

    return ensure_acl(device)



def _fp16_bytes(val: float) -> bytes:
    try:
        import numpy as np

        return np.float16(val).tobytes()
    except Exception:
        return struct.pack("<e", float(val))


def _bf16_bytes(val: float) -> bytes:
    try:
        import numpy as np

        return np.asarray(float(val), dtype=np.float32).view(np.uint16)[1].tobytes()
    except Exception:
        return struct.pack("<f", float(val))[2:4]


def _bf16_to_float(raw: bytes) -> float:
    try:
        import numpy as np

        return float(np.frombuffer(b"\x00\x00" + raw[:2], dtype=np.float32)[0])
    except Exception:
        return float(struct.unpack("<f", b"\x00\x00" + raw[:2])[0])


def _pack_vals(fmt, values):
    if fmt == "FP16":
        return b"".join(_fp16_bytes(v) for v in values)
    if fmt == "BFP16":
        return b"".join(_bf16_bytes(v) for v in values)
    code = "<" + fmt
    parts = []
    for v in values:
        if fmt in ("f", "d"):
            parts.append(struct.pack(code, float(v)))
        else:
            parts.append(struct.pack(code, int(v)))
    return b"".join(parts)


def _unpack_vals(name, fmt, data, n, esize):
    if name == "FP16":
        try:
            import numpy as np

            return [float(x) for x in np.frombuffer(data, dtype=np.float16, count=n)]
        except Exception:
            return [float(struct.unpack_from("<e", data, i * esize)[0]) for i in range(n)]
    if name == "BFP16":
        return [_bf16_to_float(data[i * esize : (i + 1) * esize]) for i in range(n)]
    return [float(struct.unpack_from("<" + fmt, data, i * esize)[0]) for i in range(n)]


def _close(a, b, name):
    if name in ("FP16", "BFP16", "FP32", "FP64"):
        return abs(a - b) < 5e-2
    return abs(a - b) < 0.5


def _malloc_copy(acl, host: bytes):
    buf = ctypes.c_void_p()
    n = len(host)
    assert acl.aclrtMalloc(ctypes.byref(buf), n, ACL_MEM_MALLOC_HUGE_FIRST) == 0
    assert acl.aclrtMemcpy(buf, n, host, n, ACL_MEMCPY_HOST_TO_DEVICE) == 0
    return buf, n


def _malloc_zero(acl, n: int):
    buf = ctypes.c_void_p()
    assert acl.aclrtMalloc(ctypes.byref(buf), n, ACL_MEM_MALLOC_HUGE_FIRST) == 0
    assert acl.aclrtMemset(buf, n, 0, n) == 0
    return buf


def _d2h(acl, buf, n, dtype_name, pack_fmt, count, esize):
    out_b = (ctypes.c_char * n)()
    assert acl.aclrtMemcpy(out_b, n, buf, n, ACL_MEMCPY_DEVICE_TO_HOST) == 0
    return _unpack_vals(dtype_name, pack_fmt, bytes(out_b), count, esize)


def _run_case(h, acl, comm, stream, op, dtype_name, dtype_attr, esize, pack_fmt, rank, nranks):
    dt = getattr(h.HcclDataType, dtype_attr)
    sum_expect = float(sum(range(1, nranks + 1)))
    bufs = []

    def alloc_copy(host):
        b, n = _malloc_copy(acl, host)
        bufs.append(b)
        return b, n

    def alloc_zero(n):
        b = _malloc_zero(acl, n)
        bufs.append(b)
        return b

    try:
        if op == "all_reduce":
            count = 4
            host = _pack_vals(pack_fmt, [float(rank + 1)] * count)
            sbuf, nbytes = alloc_copy(host)
            rbuf = alloc_zero(nbytes)
            h.all_reduce(comm, sbuf, rbuf, count, data_type=dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, nbytes, dtype_name, pack_fmt, count, esize)
            ok = all(_close(got[i], sum_expect, dtype_name) for i in range(count))
            return ok, {"out0": got[0], "expect": sum_expect}

        if op == "reduce":
            count = 4
            host = _pack_vals(pack_fmt, [float(rank + 1)] * count)
            sbuf, nbytes = alloc_copy(host)
            rbuf = alloc_zero(nbytes)
            h.reduce(comm, sbuf, rbuf, count, data_type=dt, root=0, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, nbytes, dtype_name, pack_fmt, count, esize)
            if rank == 0:
                ok = all(_close(got[i], sum_expect, dtype_name) for i in range(count))
            else:
                ok = True
            return ok, {"out0": got[0], "expect": sum_expect, "is_root": rank == 0}

        if op == "broadcast":
            count = 4
            host = _pack_vals(pack_fmt, [7.0 if rank == 0 else 0.0] * count)
            buf, nbytes = alloc_copy(host)
            h.broadcast(comm, buf, count, data_type=dt, root=0, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, buf, nbytes, dtype_name, pack_fmt, count, esize)
            ok = all(_close(got[i], 7.0, dtype_name) for i in range(count))
            return ok, {"out0": got[0], "expect": 7.0}

        if op == "all_gather":
            host = _pack_vals(pack_fmt, [float(rank + 1)])
            sbuf, snb = alloc_copy(host)
            rnb = nranks * esize
            rbuf = alloc_zero(rnb)
            h.all_gather(comm, sbuf, rbuf, 1, data_type=dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, rnb, dtype_name, pack_fmt, nranks, esize)
            expect = [float(i + 1) for i in range(nranks)]
            ok = all(_close(got[i], expect[i], dtype_name) for i in range(nranks))
            return ok, {"out": got, "expect": expect}

        if op == "all_gather_v":
            host = _pack_vals(pack_fmt, [float(rank + 1)])
            sbuf, snb = alloc_copy(host)
            rnb = nranks * esize
            rbuf = alloc_zero(rnb)
            recv_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
            recv_displs = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
            h.all_gather_v(comm, sbuf, 1, rbuf, recv_counts, recv_displs, data_type=dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, rnb, dtype_name, pack_fmt, nranks, esize)
            expect = [float(i + 1) for i in range(nranks)]
            ok = all(_close(got[i], expect[i], dtype_name) for i in range(nranks))
            return ok, {"out": got, "expect": expect}

        if op == "scatter":
            # root scatters [10, 20, ...]; each rank gets one element
            send_n = nranks
            send_bytes = send_n * esize
            sbuf = alloc_zero(send_bytes)
            if rank == 0:
                host = _pack_vals(pack_fmt, [float((i + 1) * 10) for i in range(send_n)])
                assert acl.aclrtMemcpy(sbuf, send_bytes, host, send_bytes, ACL_MEMCPY_HOST_TO_DEVICE) == 0
            rbuf = alloc_zero(esize)
            h.scatter(comm, sbuf, rbuf, 1, data_type=dt, root=0, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, esize, dtype_name, pack_fmt, 1, esize)
            expect = float((rank + 1) * 10)
            ok = _close(got[0], expect, dtype_name)
            return ok, {"out0": got[0], "expect": expect}

        if op == "reduce_scatter":
            send_n = nranks
            host = _pack_vals(pack_fmt, [float(rank + 1)] * send_n)
            sbuf, snb = alloc_copy(host)
            rbuf = alloc_zero(esize)
            h.reduce_scatter(comm, sbuf, rbuf, 1, data_type=dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, esize, dtype_name, pack_fmt, 1, esize)
            ok = _close(got[0], sum_expect, dtype_name)
            return ok, {"out0": got[0], "expect": sum_expect}

        if op == "reduce_scatter_v":
            send_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
            send_displs = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
            host = _pack_vals(pack_fmt, [float(rank + 1)] * nranks)
            sbuf, snb = alloc_copy(host)
            rbuf = alloc_zero(esize)
            h.reduce_scatter_v(comm, sbuf, send_counts, send_displs, rbuf, 1, data_type=dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, esize, dtype_name, pack_fmt, 1, esize)
            ok = _close(got[0], sum_expect, dtype_name)
            return ok, {"out0": got[0], "expect": sum_expect}

        if op == "all_to_all":
            send_count = 1
            send_n = nranks * send_count
            host = _pack_vals(pack_fmt, [float(rank + 1)] * send_n)
            sbuf, nb = alloc_copy(host)
            rbuf = alloc_zero(nb)
            h.all_to_all(comm, sbuf, send_count, dt, rbuf, send_count, dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, nb, dtype_name, pack_fmt, send_n, esize)
            expect = [float(i + 1) for i in range(nranks)]
            ok = all(_close(got[i], expect[i], dtype_name) for i in range(nranks))
            return ok, {"out": got, "expect": expect}

        if op == "all_to_all_v":
            send_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
            sdispls = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
            recv_counts = (ctypes.c_uint64 * nranks)(*([1] * nranks))
            rdispls = (ctypes.c_uint64 * nranks)(*list(range(nranks)))
            host = _pack_vals(pack_fmt, [float(rank + 1)] * nranks)
            sbuf, nb = alloc_copy(host)
            rbuf = alloc_zero(nb)
            h.all_to_all_v(
                comm, sbuf, send_counts, sdispls, dt, rbuf, recv_counts, rdispls, dt, stream=stream
            )
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, nb, dtype_name, pack_fmt, nranks, esize)
            expect = [float(i + 1) for i in range(nranks)]
            ok = all(_close(got[i], expect[i], dtype_name) for i in range(nranks))
            return ok, {"out": got, "expect": expect}

        if op == "all_to_all_vc":
            matrix = (ctypes.c_uint64 * (nranks * nranks))(*([1] * (nranks * nranks)))
            host = _pack_vals(pack_fmt, [float(rank + 1)] * nranks)
            sbuf, nb = alloc_copy(host)
            rbuf = alloc_zero(nb)
            h.all_to_all_vc(comm, sbuf, matrix, dt, rbuf, dt, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, nb, dtype_name, pack_fmt, nranks, esize)
            expect = [float(i + 1) for i in range(nranks)]
            ok = all(_close(got[i], expect[i], dtype_name) for i in range(nranks))
            return ok, {"out": got, "expect": expect}
        if op == "send_recv":
            count = 4
            peer = 1 - rank
            host = _pack_vals(pack_fmt, [float(rank + 1)] * count)
            sbuf, nbytes = alloc_copy(host)
            rbuf = alloc_zero(nbytes)
            if rank == 0:
                h.send(comm, sbuf, count, dt, dest_rank=peer, stream=stream)
                h.recv(comm, rbuf, count, dt, src_rank=peer, stream=stream)
            else:
                h.recv(comm, rbuf, count, dt, src_rank=peer, stream=stream)
                h.send(comm, sbuf, count, dt, dest_rank=peer, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, nbytes, dtype_name, pack_fmt, count, esize)
            expect = float(peer + 1)
            ok = all(_close(got[i], expect, dtype_name) for i in range(count))
            return ok, {"out0": got[0], "expect": expect}

        if op == "batch_send_recv":
            count = 4
            next_rank = (rank + 1) % nranks
            prev_rank = (rank - 1 + nranks) % nranks
            host = _pack_vals(pack_fmt, [float(rank + 1)] * count)
            sbuf, nbytes = alloc_copy(host)
            rbuf = alloc_zero(nbytes)
            items = [
                h.make_send_recv_item(h.HcclSendRecvType.SEND, sbuf, count, dt, next_rank),
                h.make_send_recv_item(h.HcclSendRecvType.RECV, rbuf, count, dt, prev_rank),
            ]
            h.batch_send_recv(comm, items, stream=stream)
            assert acl.aclrtSynchronizeStream(stream) == 0
            got = _d2h(acl, rbuf, nbytes, dtype_name, pack_fmt, count, esize)
            expect = float(prev_rank + 1)
            ok = all(_close(got[i], expect, dtype_name) for i in range(count))
            return ok, {"out0": got[0], "expect": expect}

        raise RuntimeError("unknown op %s" % op)

    finally:
        for b in bufs:
            try:
                if b:
                    acl.aclrtFree(b)
            except Exception:
                pass


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

        results = []
        try:
            for op, dtype_name, dtype_attr, esize, pack_fmt in cases:
                label = "%s/%s" % (op, dtype_name)
                try:
                    ok, detail = _run_case(
                        h, acl, comm, stream, op, dtype_name, dtype_attr, esize, pack_fmt, rank, nranks
                    )
                    results.append({"label": label, "ok": ok, "detail": detail, "unsupported": False})
                except Exception as e:
                    msg = str(e)
                    unsupported = (
                        ("NOT_SUPPORT" in msg)
                        or ("E_NOT_SUPPORT" in msg)
                        or ("code=5" in msg)
                        or ("HcclResult=5" in msg)
                    )
                    results.append(
                        {
                            "label": label,
                            "ok": False,
                            "unsupported": unsupported,
                            "error": traceback.format_exc(),
                        }
                    )
        finally:
            try:
                h.comm_destroy(comm)
            except Exception:
                pass
            try:
                acl.aclrtDestroyStream(stream)
            except Exception:
                pass
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
    n = int(n.value)
    print("device_count=%d" % n, flush=True)
    if n < 2:
        print("NEED >=2 devices", flush=True)
        return 2

    import hccl4py as h
    from _platform import detect_chip, dtype_skip_cases

    chip = detect_chip()
    SKIP_CASES.clear()
    SKIP_CASES.update(dtype_skip_cases(chip))
    print("chip=%s dtype_pre_skips=%d" % (chip, len(SKIP_CASES)), flush=True)

    root = h.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))

    cases = []
    skipped = 0
    for op in OPS:
        for name, attr, esize, fmt in DTYPE_SPECS:
            reason = SKIP_CASES.get((op, name))
            if reason:
                print("[SKIP] %s/%s %s" % (op, name, reason), flush=True)
                skipped += 1
                continue
            cases.append((op, name, attr, esize, fmt))

    print("planned_cases=%d (+%d pre-skip)" % (len(cases), skipped), flush=True)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from _platform import mp_collect, start_processes

    procs = [
        ctx.Process(target=_worker, args=(r, 2, root_bytes, cases, q)) for r in range(2)
    ]
    procs = start_processes(procs)

    payloads = mp_collect(q, 2, 1800, procs)

    if not all(p.get("ok") for p in payloads):
        print("[FAIL] worker init", flush=True)
        for p in payloads:
            if not p.get("ok"):
                print(p.get("error"), flush=True)
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
                    print(" ", str(r.get("error") or r.get("detail"))[:500], flush=True)
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

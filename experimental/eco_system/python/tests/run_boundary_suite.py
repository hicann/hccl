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

"""Exception / boundary cases for hccl4py (CPU + optional 2-device NPU)."""

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _platform import mp_collect  # noqa: E402

ACL_MEM_MALLOC_HUGE_FIRST = 0


def _acl(device=0):
    from _platform import ensure_acl

    return ensure_acl(device)



_PASSED = []
_FAILED = []


def expect_pass(name, fn):
    try:
        fn()
        print("[PASS]", name)
        _PASSED.append(name)
    except Exception as e:
        print("[FAIL]", name, repr(e))
        traceback.print_exc()
        _FAILED.append(name)


def expect_error(name, fn, exc_types=(Exception,)):
    try:
        fn()
        print("[FAIL]", name, "expected error, got success")
        _FAILED.append(name)
    except exc_types as e:
        print("[PASS]", name, "raised", type(e).__name__, ":", e)
        _PASSED.append(name)
    except Exception as e:
        print("[FAIL]", name, "wrong exception", repr(e))
        traceback.print_exc()
        _FAILED.append(name)


def _expected_config_version(h, T):
    """Runtime capability wins over offline DEFAULT (this 9.2-beta still reports 10)."""
    try:
        from hcomm4py.comm import get_comm_config_capability

        cap = get_comm_config_capability()
    except Exception:
        cap = None
    if cap is not None:
        return int(cap)
    return int(T.HCCL_COMM_CONFIG_VERSION_DEFAULT)


def test_cpu_config_layout():
    import hccl4py as h
    from hccl4py import types as T

    cfg = h.comm_config_init()
    assert ctypes.sizeof(cfg) >= T.HCCL_COMM_CONFIG_SIZE_V10
    magic = ctypes.c_uint32.from_buffer(cfg._raw, T._OFF["magic"]).value
    ver = ctypes.c_uint32.from_buffer(cfg._raw, T._OFF["version"]).value
    assert magic == T.HCCL_COMM_CONFIG_MAGIC_WORD, hex(magic)
    expect_ver = _expected_config_version(h, T)
    assert ver == expect_ver, (ver, expect_ver)
    if expect_ver >= 11:
        sq = ctypes.c_uint32.from_buffer(cfg._raw, T._OFF["hcclChannelSqDepth"]).value
        assert sq == T.HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET, hex(sq)
    h.comm_config_init(cfg)
    assert ctypes.c_uint32.from_buffer(cfg._raw, T._OFF["magic"]).value == T.HCCL_COMM_CONFIG_MAGIC_WORD


def test_send_recv_item_sizeof():
    import hccl4py as h

    assert ctypes.sizeof(h.HcclSendRecvItem) == 32
    item = h.make_send_recv_item(
        h.HcclSendRecvType.SEND, None, 1, h.HcclDataType.FP32, 0
    )
    assert item.sendRecvType == int(h.HcclSendRecvType.SEND)
    assert item.count == 1


def test_rootinfo_sizeof_after_acl():
    import hccl4py as h

    _acl()
    info = h.get_root_info()
    assert ctypes.sizeof(info) == 4108
    raw = (ctypes.c_char * ctypes.sizeof(info)).from_buffer_copy(info)
    assert len(raw) == 4108
    assert h.HCCL_ROOT_INFO_BYTES == 4108


def test_cluster_info_bad_path():
    import hccl4py as h

    _acl()
    # missing ranktable file must fail
    h.comm_init_cluster_info_config("/tmp/hccl4py_no_such_ranktable.json", 0)


def test_init_bad_rank():
    import hccl4py as h

    _acl()
    root = h.get_root_info()
    # rank >= nranks
    h.comm_init_root_info_config(2, root, 2)


def test_init_zero_nranks():
    import hccl4py as h

    _acl()
    root = h.get_root_info()
    h.comm_init_root_info_config(0, root, 0)


def _worker_allreduce_null(rank, nranks, root_bytes, q, mode):
    try:
        import hccl4py as h

        acl = _acl(rank)
        stream = ctypes.c_void_p()
        assert acl.aclrtCreateStream(ctypes.byref(stream)) == 0
        root = h.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        comm = h.comm_init_root_info_config(nranks, root, rank)
        buf = ctypes.c_void_p()
        if mode != "null_both":
            assert acl.aclrtMalloc(ctypes.byref(buf), 32, ACL_MEM_MALLOC_HUGE_FIRST) == 0
        try:
            if mode == "null_send":
                h.all_reduce(comm, None, buf, 4, stream=stream.value)
            elif mode == "null_recv":
                h.all_reduce(comm, buf, None, 4, stream=stream.value)
            elif mode == "null_both":
                h.all_reduce(comm, None, None, 4, stream=stream.value)
            elif mode == "bad_count":
                # count=0 is often accepted as no-op; use huge count to force PARA/RUNTIME
                h.all_reduce(comm, buf, buf, 2**62, stream=stream.value)
            else:
                raise RuntimeError("unknown mode")
            acl.aclrtSynchronizeStream(stream)
            q.put({"rank": rank, "ok": False, "error": "expected failure"})
        except Exception as e:
            q.put({"rank": rank, "ok": True, "error": repr(e)})
        finally:
            try:
                h.comm_destroy(comm)
            except Exception:
                pass
    except Exception as e:
        q.put({"rank": rank, "ok": False, "error": repr(e), "tb": traceback.format_exc()})


def _run_mp_expect_error(mode, require_all=True):
    import hccl4py as h

    acl = _acl()
    n = ctypes.c_uint32()
    assert acl.aclrtGetDeviceCount(ctypes.byref(n)) == 0
    avail = int(n.value)
    if avail < 2:
        raise RuntimeError("need >=2 devices")
    # Cap at 2 ranks: enough to exercise null-buffer reject; using all cards is flaky
    # after prior suite residual device state on multi-NPU hosts.
    nranks = 2
    root = h.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from _platform import start_processes

    procs = [
        ctx.Process(target=_worker_allreduce_null, args=(r, nranks, root_bytes, q, mode))
        for r in range(nranks)
    ]
    procs = start_processes(procs)
    results = mp_collect(q, nranks, 180, procs, join_timeout=30)
    raised = [r for r in results if r.get("ok")]
    if require_all:
        if len(raised) != nranks:
            raise RuntimeError(
                "expected all ranks to raise for mode=%s: %s" % (mode, results)
            )
    elif not raised:
        raise RuntimeError("no rank raised as expected: %s" % results)
    return results


def test_allreduce_null_send():
    _run_mp_expect_error("null_send", require_all=True)


def test_allreduce_null_recv():
    _run_mp_expect_error("null_recv", require_all=True)


def test_allreduce_null_both():
    _run_mp_expect_error("null_both", require_all=True)


def test_allreduce_bad_count():
    # Huge count may fail asymmetrically (PARA vs RUNTIME); require >=1 rank raise.
    _run_mp_expect_error("bad_count", require_all=False)


def test_destroy_null_handle():
    import hccl4py as h

    _acl()
    try:
        h.comm_destroy(0)
    except ValueError:
        return
    raise AssertionError("comm_destroy(0) must raise ValueError")


def test_batch_send_recv_empty_contract():
    """Null handle rejected by binding; empty item list never reaches C."""
    import hccl4py as h

    _acl()
    try:
        h.batch_send_recv(0, [], stream=0)
    except ValueError:
        return
    raise AssertionError("batch_send_recv(null, []) must raise ValueError")


def test_destroy_negative_handle():
    import hccl4py as h

    try:
        h.comm_destroy(-1)
    except ValueError:
        return
    raise AssertionError("comm_destroy(-1) must raise ValueError")


def test_ops_illegal_dtype_reduce_op():
    """Binding-layer rejects bad dtype/reduce_op before native call."""
    import hccl4py as h

    try:
        h.all_reduce(1, 0, 0, count=1, data_type=999)
    except ValueError:
        pass
    else:
        raise AssertionError("all_reduce bad dtype must raise ValueError")
    try:
        h.reduce(1, 0, 0, count=1, reduce_op=999)
    except ValueError:
        return
    raise AssertionError("reduce bad reduce_op must raise ValueError")


def test_config_fields_writable():
    """HcclCommConfig bufferSize / commName can be set without corrupting magic."""
    import hccl4py as h
    from hccl4py import types as T

    cfg = h.comm_config_init()
    T._u32(cfg, T._OFF["hcclBufferSize"], 20)
    name = b"hccl4py_cfg\0"
    for i, b in enumerate(name):
        cfg._raw[T._OFF["hcclCommName"] + i] = b
    magic = ctypes.c_uint32.from_buffer(cfg._raw, T._OFF["magic"]).value
    ver = ctypes.c_uint32.from_buffer(cfg._raw, T._OFF["version"]).value
    bufsz = ctypes.c_uint32.from_buffer(cfg._raw, T._OFF["hcclBufferSize"]).value
    assert magic == T.HCCL_COMM_CONFIG_MAGIC_WORD, hex(magic)
    expect_ver = _expected_config_version(h, T)
    assert ver == expect_ver, (ver, expect_ver)
    assert bufsz == 20, bufsz
    if expect_ver >= 11:
        sq = ctypes.c_uint32.from_buffer(cfg._raw, T._OFF["hcclChannelSqDepth"]).value
        assert sq == T.HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET, hex(sq)



def _worker_cfg_init(rank, nranks, root_bytes, q):
    try:
        import hccl4py as h
        from hccl4py import types as T

        acl = _acl(rank)
        cfg = h.comm_config_init()
        T._u32(cfg, T._OFF["hcclBufferSize"], 20)
        name = b"hccl4py_cfg\0"
        for i, b in enumerate(name):
            cfg._raw[T._OFF["hcclCommName"] + i] = b
        root = h.HcclRootInfo()
        ctypes.memmove(ctypes.addressof(root), root_bytes, len(root_bytes))
        comm = h.comm_init_root_info_config(nranks, root, rank, config=cfg)
        h.comm_destroy(comm)
        q.put({"rank": rank, "ok": True})
    except Exception:
        q.put({"rank": rank, "ok": False, "error": traceback.format_exc()})


def test_config_fields_rootinfo_init():
    """Custom config fields still allow RootInfo init + destroy on 2 ranks."""
    import hccl4py as h

    acl = _acl()
    n = ctypes.c_uint32()
    assert acl.aclrtGetDeviceCount(ctypes.byref(n)) == 0
    if int(n.value) < 2:
        raise RuntimeError("NEED >=2 devices")
    root = h.get_root_info()
    root_bytes = ctypes.string_at(ctypes.addressof(root), ctypes.sizeof(root))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from _platform import start_processes

    procs = [ctx.Process(target=_worker_cfg_init, args=(r, 2, root_bytes, q)) for r in range(2)]
    procs = start_processes(procs)
    results = mp_collect(q, 2, 180, procs, join_timeout=30)
    if not all(r.get("ok") for r in results):
        raise RuntimeError("config init failed: %s" % results)


def main():
    import hccl4py as h

    print("[INFO] hccl4py", getattr(h, "__version__", "?"))
    expect_pass("cpu_config_layout", test_cpu_config_layout)
    expect_pass("send_recv_item_sizeof32", test_send_recv_item_sizeof)
    expect_pass("rootinfo_sizeof", test_rootinfo_sizeof_after_acl)
    expect_pass("config_fields_writable", test_config_fields_writable)
    expect_error(
        "cluster_info_bad_path",
        test_cluster_info_bad_path,
        (h.HcclError,),
    )
    expect_error("init_bad_rank", test_init_bad_rank, (ValueError,))
    expect_error("init_zero_nranks", test_init_zero_nranks, (ValueError,))
    expect_pass("allreduce_null_send", test_allreduce_null_send)
    expect_pass("allreduce_null_recv", test_allreduce_null_recv)
    expect_pass("allreduce_null_both", test_allreduce_null_both)
    expect_pass("allreduce_bad_count", test_allreduce_bad_count)
    expect_pass("destroy_null_handle_contract", test_destroy_null_handle)
    expect_pass("destroy_negative_handle", test_destroy_negative_handle)
    expect_pass("batch_send_recv_empty_contract", test_batch_send_recv_empty_contract)
    expect_pass("ops_illegal_dtype_reduce_op", test_ops_illegal_dtype_reduce_op)
    expect_pass("config_fields_rootinfo_init", test_config_fields_rootinfo_init)

    print("=== SUMMARY passed=%d failed=%d ===" % (len(_PASSED), len(_FAILED)))
    if _FAILED:
        print("failed:", _FAILED)
        return 1
    return 0


if __name__ == "__main__":
    # fix nested expect in destroy test counting
    sys.exit(main())

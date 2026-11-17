# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Collective / P2P Python bindings (task 3.1.2 - hccl)."""

import ctypes
import threading
from typing import Sequence
from ._lib import get_hccl
from .types import (
    HcclDataType,
    HcclReduceOp,
    HcclSendRecvItem,
    check,
)
from ._validate import require_nonneg, require_u32, require_u64
from ._ops_util import (
    _as_comm,
    _as_ptr,
    _as_stream,
    _require_data_type,
    _require_reduce_op,
    _require_send_recv_type,
)
from ._bindings import bind_unlocked as _bind_unlocked

_BOUND = False
_BIND_LOCK = threading.Lock()


def _bind():
    global _BOUND
    lib = get_hccl()
    if _BOUND:
        return lib
    with _BIND_LOCK:
        if _BOUND:
            return lib
        _bind_unlocked(lib)
        _BOUND = True
        return lib


def all_reduce(
    hccl_comm,
    send_buf,
    recv_buf,
    count: int,
    data_type: int = HcclDataType.FP32,
    reduce_op: int = HcclReduceOp.SUM,
    stream=None,
) -> int:
    """AllReduce: reduce then broadcast.

    `count` is the element count (not bytes). Buffers are device pointers.
    Returns HcclResult code (0 on success); non-zero raises HcclError via check().
    """
    count = require_u64("count", count)
    data_type = _require_data_type(data_type)
    reduce_op = _require_reduce_op(reduce_op)
    send = _as_ptr(send_buf)
    recv = _as_ptr(recv_buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclAllReduce(
        send,
        recv,
        ctypes.c_uint64(count),
        data_type,
        reduce_op,
        hccl_comm,
        stream,
    )
    check(code, "HcclAllReduce")
    return int(code)


def reduce(
    hccl_comm,
    send_buf,
    recv_buf,
    count: int,
    data_type: int = HcclDataType.FP32,
    reduce_op: int = HcclReduceOp.SUM,
    root: int = 0,
    stream=None,
) -> int:
    """Reduce to `root`. `count` is element count; buffers are device pointers."""
    count = require_u64("count", count)
    data_type = _require_data_type(data_type)
    reduce_op = _require_reduce_op(reduce_op)
    root = require_u32("root", root)
    send = _as_ptr(send_buf)
    recv = _as_ptr(recv_buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclReduce(
        send,
        recv,
        ctypes.c_uint64(count),
        data_type,
        reduce_op,
        ctypes.c_uint32(root),
        hccl_comm,
        stream,
    )
    check(code, "HcclReduce")
    return int(code)


def broadcast(hccl_comm, buf, count: int, data_type: int = HcclDataType.FP32, root: int = 0, stream=None) -> int:
    """Broadcast `buf` from `root`. `count` is element count."""
    count = require_u64("count", count)
    data_type = _require_data_type(data_type)
    root = require_u32("root", root)
    buf = _as_ptr(buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclBroadcast(
        buf,
        ctypes.c_uint64(count),
        data_type,
        ctypes.c_uint32(root),
        hccl_comm,
        stream,
    )
    check(code, "HcclBroadcast")
    return int(code)


def all_gather(hccl_comm, send_buf, recv_buf, send_count: int, data_type: int = HcclDataType.FP32, stream=None) -> int:
    """AllGather. `send_count` is per-rank element count; recv holds nranks * send_count."""
    send_count = require_u64("send_count", send_count)
    data_type = _require_data_type(data_type)
    send = _as_ptr(send_buf)
    recv = _as_ptr(recv_buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclAllGather(
        send,
        recv,
        ctypes.c_uint64(send_count),
        data_type,
        hccl_comm,
        stream,
    )
    check(code, "HcclAllGather")
    return int(code)


def all_gather_v(
    hccl_comm,
    send_buf,
    send_count: int,
    recv_buf,
    recv_counts,
    recv_displs,
    data_type: int = HcclDataType.FP32,
    stream=None,
) -> int:
    """AllGatherV with per-rank recv counts/displacements (host uint64 arrays)."""
    send_count = require_u64("send_count", send_count)
    data_type = _require_data_type(data_type)
    if recv_counts is None or recv_displs is None:
        raise ValueError("recv_counts and recv_displs must not be None")
    send = _as_ptr(send_buf)
    recv = _as_ptr(recv_buf)
    counts = _as_ptr(recv_counts)
    displs = _as_ptr(recv_displs)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclAllGatherV(
        send,
        ctypes.c_uint64(send_count),
        recv,
        counts,
        displs,
        data_type,
        hccl_comm,
        stream,
    )
    check(code, "HcclAllGatherV")
    return int(code)


def scatter(
    hccl_comm,
    send_buf,
    recv_buf,
    recv_count: int,
    data_type: int = HcclDataType.FP32,
    root: int = 0,
    stream=None,
) -> int:
    """Scatter from `root`. `recv_count` is per-rank element count."""
    recv_count = require_u64("recv_count", recv_count)
    data_type = _require_data_type(data_type)
    root = require_u32("root", root)
    send = _as_ptr(send_buf)
    recv = _as_ptr(recv_buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclScatter(
        send,
        recv,
        ctypes.c_uint64(recv_count),
        data_type,
        ctypes.c_uint32(root),
        hccl_comm,
        stream,
    )
    check(code, "HcclScatter")
    return int(code)


def reduce_scatter(
    hccl_comm,
    send_buf,
    recv_buf,
    recv_count: int,
    data_type: int = HcclDataType.FP32,
    reduce_op: int = HcclReduceOp.SUM,
    stream=None,
) -> int:
    """ReduceScatter. `recv_count` is per-rank element count after scatter."""
    recv_count = require_u64("recv_count", recv_count)
    data_type = _require_data_type(data_type)
    reduce_op = _require_reduce_op(reduce_op)
    send = _as_ptr(send_buf)
    recv = _as_ptr(recv_buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclReduceScatter(
        send,
        recv,
        ctypes.c_uint64(recv_count),
        data_type,
        reduce_op,
        hccl_comm,
        stream,
    )
    check(code, "HcclReduceScatter")
    return int(code)


def reduce_scatter_v(
    hccl_comm,
    send_buf,
    send_counts,
    send_displs,
    recv_buf,
    recv_count: int,
    data_type: int = HcclDataType.FP32,
    reduce_op: int = HcclReduceOp.SUM,
    stream=None,
) -> int:
    """ReduceScatterV with per-rank send counts/displacements (host uint64 arrays)."""
    recv_count = require_u64("recv_count", recv_count)
    data_type = _require_data_type(data_type)
    reduce_op = _require_reduce_op(reduce_op)
    if send_counts is None or send_displs is None:
        raise ValueError("send_counts and send_displs must not be None")
    send = _as_ptr(send_buf)
    counts = _as_ptr(send_counts)
    displs = _as_ptr(send_displs)
    recv = _as_ptr(recv_buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclReduceScatterV(
        send,
        counts,
        displs,
        recv,
        ctypes.c_uint64(recv_count),
        data_type,
        reduce_op,
        hccl_comm,
        stream,
    )
    check(code, "HcclReduceScatterV")
    return int(code)


def all_to_all(
    hccl_comm,
    send_buf,
    send_count: int,
    send_type: int,
    recv_buf,
    recv_count: int,
    recv_type: int,
    stream=None,
) -> int:
    """AlltoAll with uniform counts. Counts are element counts per rank-peer."""
    send_count = require_u64("send_count", send_count)
    recv_count = require_u64("recv_count", recv_count)
    send_type = _require_data_type(send_type)
    recv_type = _require_data_type(recv_type)
    send = _as_ptr(send_buf)
    recv = _as_ptr(recv_buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclAlltoAll(
        send,
        ctypes.c_uint64(send_count),
        send_type,
        recv,
        ctypes.c_uint64(recv_count),
        recv_type,
        hccl_comm,
        stream,
    )
    check(code, "HcclAlltoAll")
    return int(code)


def all_to_all_v(
    hccl_comm,
    send_buf,
    send_counts,
    sdispls,
    send_type: int,
    recv_buf,
    recv_counts,
    rdispls,
    recv_type: int,
    stream=None,
) -> int:
    """AlltoAllV with per-peer counts/displacements (host uint64 arrays)."""
    send_type = _require_data_type(send_type)
    recv_type = _require_data_type(recv_type)
    if None in (send_counts, sdispls, recv_counts, rdispls):
        raise ValueError("send/recv counts and displs must not be None")
    send = _as_ptr(send_buf)
    scounts = _as_ptr(send_counts)
    sdisp = _as_ptr(sdispls)
    recv = _as_ptr(recv_buf)
    rcounts = _as_ptr(recv_counts)
    rdisp = _as_ptr(rdispls)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclAlltoAllV(
        send,
        scounts,
        sdisp,
        send_type,
        recv,
        rcounts,
        rdisp,
        recv_type,
        hccl_comm,
        stream,
    )
    check(code, "HcclAlltoAllV")
    return int(code)


def all_to_all_vc(
    hccl_comm,
    send_buf,
    send_count_matrix,
    send_type: int,
    recv_buf,
    recv_type: int,
    stream=None,
) -> int:
    """AlltoAllVC with host send_count_matrix (uint64, nRanks*nRanks)."""
    send_type = _require_data_type(send_type)
    recv_type = _require_data_type(recv_type)
    if send_count_matrix is None:
        raise ValueError("send_count_matrix must not be None")
    send = _as_ptr(send_buf)
    matrix = _as_ptr(send_count_matrix)
    recv = _as_ptr(recv_buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclAlltoAllVC(
        send,
        matrix,
        send_type,
        recv,
        recv_type,
        hccl_comm,
        stream,
    )
    check(code, "HcclAlltoAllVC")
    return int(code)


def send(hccl_comm, send_buf, count: int, data_type: int, dest_rank: int, stream=None) -> int:
    """Point-to-point send. `count` is element count; `dest_rank` is peer rank."""
    count = require_u64("count", count)
    data_type = _require_data_type(data_type)
    dest_rank = require_u32("dest_rank", dest_rank)
    send = _as_ptr(send_buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclSend(
        send,
        ctypes.c_uint64(count),
        data_type,
        ctypes.c_uint32(dest_rank),
        hccl_comm,
        stream,
    )
    check(code, "HcclSend")
    return int(code)


def recv(hccl_comm, recv_buf, count: int, data_type: int, src_rank: int, stream=None) -> int:
    """Point-to-point recv. `count` is element count; `src_rank` is peer rank."""
    count = require_u64("count", count)
    data_type = _require_data_type(data_type)
    src_rank = require_u32("src_rank", src_rank)
    recv = _as_ptr(recv_buf)
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    code = lib.HcclRecv(
        recv,
        ctypes.c_uint64(count),
        data_type,
        ctypes.c_uint32(src_rank),
        hccl_comm,
        stream,
    )
    check(code, "HcclRecv")
    return int(code)


def batch_send_recv(
    hccl_comm,
    items: Sequence[HcclSendRecvItem],
    stream=None,
) -> int:
    """Batch Send/Recv from a sequence of HcclSendRecvItem.

    A prebuilt ``(HcclSendRecvItem * n)`` ctypes array is passed through so
    hot paths (perf bench) can reuse the same buffer as the raw C call.
    """
    if items is None:
        raise TypeError("items must be a sequence of HcclSendRecvItem")
    hccl_comm = _as_comm(hccl_comm)
    stream = _as_stream(stream)
    lib = _bind()
    if isinstance(items, ctypes.Array) and getattr(items, "_type_", None) is HcclSendRecvItem:
        n = require_nonneg("items length", len(items))
        arr = items
    else:
        try:
            n = len(items)
        except TypeError as exc:
            raise TypeError("items must be a sequence of HcclSendRecvItem") from exc
        n = require_nonneg("items length", n)
        for i, item in enumerate(items):
            if not isinstance(item, HcclSendRecvItem):
                raise TypeError(
                    f"items[{i}] must be HcclSendRecvItem, got {type(item).__name__}"
                )
        arr = (HcclSendRecvItem * n)(*items)
    code = lib.HcclBatchSendRecv(arr, ctypes.c_uint32(n), hccl_comm, stream)
    check(code, "HcclBatchSendRecv")
    return int(code)


def make_send_recv_item(
    send_recv_type: int,
    buf,
    count: int,
    data_type: int,
    remote_rank: int,
) -> HcclSendRecvItem:
    """Build one HcclSendRecvItem for batch_send_recv."""
    item = HcclSendRecvItem()
    item.sendRecvType = _require_send_recv_type(send_recv_type)
    item.buf = _as_ptr(buf)
    item.count = require_u64("count", count)
    item.dataType = _require_data_type(data_type)
    item.remoteRank = require_u32("remote_rank", remote_rank)
    return item

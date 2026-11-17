# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal pointer/handle helpers for hccl4py.ops."""

import ctypes
from typing import Union

from .types import HcclComm, HcclDataType, HcclReduceOp, HcclSendRecvType, aclrtStream
from ._validate import is_null_handle


def _as_comm(hccl_comm: Union[int, HcclComm, object]) -> HcclComm:
    if hasattr(hccl_comm, "handle"):
        handle = hccl_comm.handle
        if is_null_handle(handle):
            raise ValueError("Comm handle is null/destroyed")
        return handle if isinstance(handle, HcclComm) else HcclComm(handle)
    if is_null_handle(hccl_comm):
        raise ValueError("null/zero communicator handle")
    return hccl_comm if isinstance(hccl_comm, HcclComm) else HcclComm(hccl_comm)


def _default_stream() -> aclrtStream:
    """Resolve ACL context default stream (HCCL rejects a literal NULL stream)."""
    from ._lib import get_ascendcl

    acl = get_ascendcl()
    fn = getattr(acl, "aclrtCtxGetCurrentDefaultStream", None)
    if fn is None:
        raise RuntimeError(
            "stream=None requires aclrtCtxGetCurrentDefaultStream; pass an explicit stream"
        )
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    fn.restype = ctypes.c_int
    out = ctypes.c_void_p()
    rc = fn(ctypes.byref(out))
    if rc != 0 or not out.value:
        raise RuntimeError(
            "aclrtCtxGetCurrentDefaultStream failed rc=%s value=%s" % (rc, out.value)
        )
    return aclrtStream(int(out.value))


def _as_stream(stream: Union[int, aclrtStream, None]) -> aclrtStream:
    if stream is None:
        return _default_stream()
    if isinstance(stream, bool):
        raise TypeError("stream must be int/aclrtStream/None, got bool")
    if isinstance(stream, ctypes.c_void_p):
        return aclrtStream(int(stream.value or 0))
    if isinstance(stream, int):
        return aclrtStream(stream)
    raise TypeError(
        "stream must be int/aclrtStream/None, got %s" % type(stream).__name__
    )


def _as_ptr(buf) -> ctypes.c_void_p:
    if buf is None:
        return ctypes.c_void_p()
    if isinstance(buf, bool):
        raise TypeError("buf must be int/c_void_p/None, got bool")
    if isinstance(buf, int):
        return ctypes.c_void_p(buf)
    if isinstance(buf, ctypes.c_void_p):
        return ctypes.c_void_p(int(buf.value or 0))
    return ctypes.cast(buf, ctypes.c_void_p)



_DATA_TYPES = frozenset(int(v) for v in HcclDataType if v != HcclDataType.RESERVED)
_REDUCE_OPS = frozenset(int(v) for v in HcclReduceOp if v != HcclReduceOp.RESERVED)
_SEND_RECV_TYPES = frozenset(int(v) for v in HcclSendRecvType if v != HcclSendRecvType.RESERVED)


def _require_data_type(data_type) -> int:
    if isinstance(data_type, bool) or not isinstance(data_type, int):
        raise TypeError(
            f"data_type must be int/HcclDataType, got {type(data_type).__name__}"
        )
    value = int(data_type)
    if value not in _DATA_TYPES:
        raise ValueError(f"invalid data_type: {value}")
    return value


def _require_reduce_op(reduce_op) -> int:
    if isinstance(reduce_op, bool) or not isinstance(reduce_op, int):
        raise TypeError(
            f"reduce_op must be int/HcclReduceOp, got {type(reduce_op).__name__}"
        )
    value = int(reduce_op)
    if value not in _REDUCE_OPS:
        raise ValueError(f"invalid reduce_op: {value}")
    return value


def _require_send_recv_type(send_recv_type) -> int:
    if isinstance(send_recv_type, bool) or not isinstance(send_recv_type, int):
        raise TypeError(
            f"send_recv_type must be int/HcclSendRecvType, got {type(send_recv_type).__name__}"
        )
    value = int(send_recv_type)
    if value not in _SEND_RECV_TYPES:
        raise ValueError(f"invalid send_recv_type: {value}")
    return value



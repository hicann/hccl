# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ctypes argtypes/restype tables for hccl4py ops (kept separate to keep ops.py lean)."""

import ctypes

from .types import HcclComm, HcclSendRecvItem, aclrtStream

_OP_SYMBOLS = (
    "HcclAllReduce",
    "HcclReduce",
    "HcclBroadcast",
    "HcclAllGather",
    "HcclAllGatherV",
    "HcclScatter",
    "HcclReduceScatter",
    "HcclReduceScatterV",
    "HcclAlltoAll",
    "HcclAlltoAllV",
    "HcclAlltoAllVC",
    "HcclSend",
    "HcclRecv",
    "HcclBatchSendRecv",
)


def _require_op_symbols(lib) -> None:
    missing = [name for name in _OP_SYMBOLS if not hasattr(lib, name)]
    if not missing:
        return
    raise RuntimeError(
        "libhccl.so missing HCCL op symbols %s; hccl4py requires CANN >= 9.2.0 "
        "(source set_env.sh or set HCCL4PY_HCCL_LIB)" % (missing,)
    )


def bind_unlocked(lib):
    _require_op_symbols(lib)
    # AllReduce
    lib.HcclAllReduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_int,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclAllReduce.restype = ctypes.c_int
    lib.HcclReduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint32,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclReduce.restype = ctypes.c_int
    lib.HcclBroadcast.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_uint32,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclBroadcast.restype = ctypes.c_int
    lib.HcclAllGather.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclAllGather.restype = ctypes.c_int
    lib.HcclAllGatherV.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclAllGatherV.restype = ctypes.c_int
    lib.HcclScatter.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_uint32,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclScatter.restype = ctypes.c_int
    lib.HcclReduceScatter.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_int,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclReduceScatter.restype = ctypes.c_int
    lib.HcclReduceScatterV.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_int,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclReduceScatterV.restype = ctypes.c_int
    lib.HcclAlltoAll.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclAlltoAll.restype = ctypes.c_int
    lib.HcclAlltoAllV.argtypes = [
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
    ]
    lib.HcclAlltoAllV.restype = ctypes.c_int
    lib.HcclAlltoAllVC.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclAlltoAllVC.restype = ctypes.c_int
    lib.HcclSend.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_uint32,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclSend.restype = ctypes.c_int
    lib.HcclRecv.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_uint32,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclRecv.restype = ctypes.c_int
    lib.HcclBatchSendRecv.argtypes = [
        ctypes.POINTER(HcclSendRecvItem),
        ctypes.c_uint32,
        HcclComm,
        aclrtStream,
    ]
    lib.HcclBatchSendRecv.restype = ctypes.c_int

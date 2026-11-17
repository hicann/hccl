# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Re-export domain types from ``hcomm4py`` (single source of truth)."""

from ._hcomm_dep import ensure_hcomm4py

ensure_hcomm4py()

from hcomm4py.types import *  # noqa: E402,F401,F403
from hcomm4py.types import (  # noqa: E402
    HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET,
    HCCL_COMM_CONFIG_BYTES,
    HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT,
    HCCL_COMM_CONFIG_MAGIC_WORD,
    HCCL_COMM_CONFIG_SIZE_V10,
    HCCL_COMM_CONFIG_SIZE_V11,
    HCCL_COMM_CONFIG_SIZE_V12,
    HCCL_COMM_CONFIG_VERSION_DEFAULT,
    HCCL_COMM_DEFAULT_OP_EXPANSION_MODE,
    HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET,
    HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET,
    HCCL_COMM_QOS_CONFIG_NOT_SET,
    HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET,
    HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET,
    HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET,
    HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET,
    HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE,
    HCCL_OP_EXPANSION_MODE_CCU_SCHED,
    HCCL_ROOT_INFO_BYTES,
    HcclComm,
    HcclCommConfig,
    HcclDataType,
    HcclError,
    HcclReduceOp,
    HcclResult,
    HcclRootInfo,
    HcclSendRecvItem,
    HcclSendRecvType,
    _OFF,
    _i32,
    _size,
    _u32,
    _u64,
    _u8,
    aclrtStream,
    check,
    fill_comm_config,
)

__all__ = [
    "HcclResult",
    "HcclReduceOp",
    "HcclDataType",
    "HcclSendRecvType",
    "HcclRootInfo",
    "HcclCommConfig",
    "HcclSendRecvItem",
    "HcclError",
    "HcclComm",
    "aclrtStream",
    "check",
    "fill_comm_config",
    "HCCL_ROOT_INFO_BYTES",
    "HCCL_COMM_CONFIG_BYTES",
    "HCCL_COMM_CONFIG_MAGIC_WORD",
    "HCCL_COMM_CONFIG_SIZE_V10",
    "HCCL_COMM_CONFIG_SIZE_V11",
    "HCCL_COMM_CONFIG_SIZE_V12",
    "HCCL_COMM_CONFIG_VERSION_DEFAULT",
    "HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT",
    "HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET",
    "HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET",
    "HCCL_COMM_DEFAULT_OP_EXPANSION_MODE",
    "HCCL_OP_EXPANSION_MODE_CCU_SCHED",
    "HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET",
    "HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET",
    "HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET",
    "HCCL_COMM_QOS_CONFIG_NOT_SET",
    "HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET",
    "HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET",
    "HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE",
]

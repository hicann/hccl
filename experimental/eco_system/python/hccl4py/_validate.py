# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Re-export validators from ``hcomm4py`` (single source of truth)."""

from __future__ import annotations

from ._hcomm_dep import ensure_hcomm4py

ensure_hcomm4py()

from hcomm4py._validate import (  # noqa: E402
    handle_value,
    is_null_handle,
    require_int,
    require_live_handle,
    require_nonneg,
    require_positive,
    require_positive_u32,
    require_u32,
    require_u64,
)

__all__ = [
    "require_int",
    "require_nonneg",
    "require_positive",
    "require_u32",
    "require_u64",
    "require_positive_u32",
    "handle_value",
    "is_null_handle",
    "require_live_handle",
]

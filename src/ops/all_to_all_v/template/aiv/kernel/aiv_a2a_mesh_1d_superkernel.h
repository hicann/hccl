/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base_v2.h"
#include "aiv_all_to_all_mesh_1D.h"

extern "C" __aicore__ void sk_alltoall_mesh_1d(SUPERKERNEL_LITE_ARGS_DEF)
{
    AIV_INFO_HINT;
    SUPERKERNEL_LITE_ARGS_EXTRACT;
    return sk_a2a_mesh_1d(SUPERKERNEL_ARGS_CALL);
}

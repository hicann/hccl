/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_TO_ALL_OP_H
#define AIV_ALL_TO_ALL_OP_H

#include "aiv_communication_base_v2.h"
#include "aiv_all_to_all_mesh_1D.h"
using namespace AscendC;

#define AIV_ALL_TO_ALL_KERNEL_DECL(type) extern "C" __aicore__ void aiv_alltoall_##type##_inner(KERNEL_ARGS_DEF);

#define AIV_ALL_TO_ALL_KERNEL_DEF(type)                                     \
    extern "C" __aicore__ void aiv_alltoall_##type##_inner(KERNEL_ARGS_DEF) \
    {                                                                       \
        AIV_INFO_HINT;                                                      \
        AivAlltoAllV2Mesh1D<type>(KERNEL_ARGS_CALL);                        \
    }

#if defined(BUILD_SK_FUNC) && defined(SK_FUNC_ID)
#define AIV_ALL_TO_ALL_KERNEL_BATCH_DEF(type) \
    AIV_ALL_TO_ALL_KERNEL_DECL(type);         \
    SK_BIND_FUNC_DEF(aiv_alltoall_##type, SK_FUNC_ID)
#else
#define AIV_ALL_TO_ALL_KERNEL_BATCH_DEF(type) \
    AIV_ALL_TO_ALL_KERNEL_DEF(type);          \
    GLOBAL_FUNC_DEF(aiv_alltoall_##type);     \
    SuperKernelBind(aiv_alltoall_##type)
#endif

// 定义各算子各数据类型Kernel入口
AIV_COPY_DATA_TYPE_DEF(AIV_ALL_TO_ALL_KERNEL_BATCH_DEF);
#endif // AIV_ALL_TO_ALL_OP_H

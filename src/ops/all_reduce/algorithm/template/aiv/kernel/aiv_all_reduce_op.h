/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_REDUCE_OP_H
#define AIV_ALL_REDUCE_OP_H

#include "aiv_communication_base_v2.h"
#include "aiv_all_reduce_mesh_1d_twoshot.h"
#include "aiv_all_reduce_mesh_1d_oneshot.h"

using namespace AscendC;

#define AIV_ALL_REDUCE_ONESHOT_KERNEL_DECL(type) \
    extern "C" __aicore__ void aiv_allreduce_##type##_inner(KERNEL_ARGS_DEF);

#define AIV_ALL_REDUCE_ONESHOT_KERNEL_DEF(type)                              \
    extern "C" __aicore__ void aiv_allreduce_##type##_inner(KERNEL_ARGS_DEF) \
    {                                                                        \
        AIV_INFO_HINT;                                                       \
        return AivAllReduceV2Mesh1DOneShot<type>(KERNEL_ARGS_CALL);          \
    }

#define AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_DECL(type) \
    extern "C" __aicore__ void aiv_allreduce_mesh1d_twoshot_##type##_inner(KERNEL_ARGS_DEF);

#define AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_DEF(type)                                       \
    extern "C" __aicore__ void aiv_allreduce_mesh1d_twoshot_##type##_inner(KERNEL_ARGS_DEF) \
    {                                                                                       \
        AIV_INFO_HINT;                                                                      \
        return AivAllReduceV2Mesh1DTwoShot<type>(KERNEL_ARGS_CALL);                         \
    }

#if defined(BUILD_SK_FUNC) && defined(SK_FUNC_ID)
#define AIV_ALL_REDUCE_ONESHOT_KERNEL_BATCH_DEF(type) \
    AIV_ALL_REDUCE_ONESHOT_KERNEL_DECL(type);         \
    SK_BIND_FUNC_DEF(aiv_allreduce_##type, SK_FUNC_ID)
#else
#define AIV_ALL_REDUCE_ONESHOT_KERNEL_BATCH_DEF(type) \
    AIV_ALL_REDUCE_ONESHOT_KERNEL_DEF(type);          \
    GLOBAL_FUNC_DEF(aiv_allreduce_##type);            \
    SuperKernelBind(aiv_allreduce_##type)
#endif

#if defined(BUILD_SK_FUNC) && defined(SK_FUNC_ID)
#define AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_BATCH_DEF(type) \
    AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_DECL(type);         \
    SK_BIND_FUNC_DEF(aiv_allreduce_mesh1d_twoshot_##type, SK_FUNC_ID)
#else
#define AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_BATCH_DEF(type) \
    AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_DEF(type);          \
    GLOBAL_FUNC_DEF(aiv_allreduce_mesh1d_twoshot_##type);   \
    SuperKernelBind(aiv_allreduce_mesh1d_twoshot_##type)
#endif

// 定义各算子各数据类型Kernel入口
AIV_ATOMIC_DATA_TYPE_DEF(AIV_ALL_REDUCE_ONESHOT_KERNEL_BATCH_DEF);
AIV_ATOMIC_DATA_TYPE_DEF(AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_BATCH_DEF);

#endif // AIV_ALL_REDUCE_OP_H

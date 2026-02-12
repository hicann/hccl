/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_AIV_COMMUNICATION_V2_H
#define ALL_REDUCE_AIV_COMMUNICATION_V2_H

#include "aiv_communication_base_v2.h"
#include "aiv_all_reduce_mesh_1d_twoshot.h"
#include "aiv_all_reduce_mesh_1d_oneshot.h"

using namespace AscendC;

#define AIV_ALL_REDUCE_ONESHOT_KERNEL_BATCH_DEF(type) \
extern "C" __global__ __aicore__ void aiv_allreduce_##type(EXTERN_KERNEL_ARGS_DEF_V2) { \
    if (isOpBase) { \
        return AivAllReduceV2Mesh1DOneShot<type>(EXTERN_KERNEL_ARGS_CALL); \
    } \
} \
EXPORT_AIV_META_INFO(aiv_allreduce_##type)

#define AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_BATCH_DEF(type) \
extern "C" __global__ __aicore__ void aiv_allreduce_mesh1d_twoshot_##type(EXTERN_KERNEL_ARGS_DEF_V2) { \
    if (isOpBase) { \
        return AivAllReduceV2Mesh1DTwoShot<type>(EXTERN_KERNEL_ARGS_CALL); \
    } \
} \
EXPORT_AIV_META_INFO(aiv_allreduce_mesh1d_twoshot_##type)

// 定义各算子各数据类型Kernel入口
AIV_ATOMIC_DATA_TYPE_DEF(AIV_ALL_REDUCE_ONESHOT_KERNEL_BATCH_DEF);
AIV_ATOMIC_DATA_TYPE_DEF(AIV_ALLREDUCE_MESH1D_TWOSHOT_KERNEL_BATCH_DEF);

#endif  /* AIV_COMMUNICATION_V2_H */
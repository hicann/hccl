/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_AIV_COMMUNICATION_V2_H
#define ALL_GATHER_AIV_COMMUNICATION_V2_H

#include "aiv_communication_base_v2.h"
#include "aiv_broadcast_mesh_1d.h"

using namespace AscendC;

#define AIV_BROADCAST_KERNEL_BATCH_DEF(type) \
extern "C" __global__ __aicore__ void aiv_broadcast_##type(EXTERN_KERNEL_ARGS_DEF_V2) { \
    if (isOpBase) { \
        return AivBroadcastV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL); \
    } \
} \
EXPORT_AIV_META_INFO(aiv_broadcast_##type)

// 定义各算子各数据类型Kernel入口
AIV_COPY_DATA_TYPE_DEF(AIV_BROADCAST_KERNEL_BATCH_DEF);

#endif  /* AIV_COMMUNICATION_V2_H */
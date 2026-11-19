/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_TO_ALL_V_OP_H
#define AIV_ALL_TO_ALL_V_OP_H

#include "aiv_communication_base_v2.h"
#include "aiv_all_to_all_v_mesh_1D.h"
// 960(A6/dav-920r1)暂不注册URMA kernel，其编译环境的include路径不含version/目录
#if !defined(__DAV_920R1__)
#include "version/hcomm_version.h"
#include "version/asc_devkit_version.h"
#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
#include "aiv_urma_all_to_all_v_mesh_1d.h"
#endif
#endif
using namespace AscendC;

#define AIV_ALL_TO_ALL_V_KERNEL_BATCH_DEF(type)                                           \
    extern "C" __global__ __aicore__ void aiv_alltoallv_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                     \
        AIV_INFO_HINT;                                                                    \
        AivAlltoAllVV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                              \
    }                                                                                     \
    EXPORT_AIV_META_INFO(aiv_alltoallv_##type)

// 定义各算子各数据类型Kernel入口
AIV_COPY_DATA_TYPE_DEF(AIV_ALL_TO_ALL_V_KERNEL_BATCH_DEF);

// URMA AllToAllV kernel入口（编入同一个.o文件）
// 960(A6/dav-920r1)只保留空kernel入口，不包含URMA实现（其编译环境无version/和hcomm相关头文件）
#if defined(__DAV_920R1__) || ((ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000))
#if defined(__DAV_920R1__)
#define AIV_URMA_ALL_TO_ALL_V_KERNEL_BATCH_DEF(type)                                                              \
    extern "C" __global__ __aicore__ void aiv_urma_alltoallv_##type(EXTERN_KERNEL_ARGS_DEF_V2) { AIV_INFO_HINT; } \
    EXPORT_AIV_META_INFO(aiv_urma_alltoallv_##type)
#else
#define AIV_URMA_ALL_TO_ALL_V_KERNEL_BATCH_DEF(type)                                           \
    extern "C" __global__ __aicore__ void aiv_urma_alltoallv_##type(EXTERN_KERNEL_ARGS_DEF_V2) \
    {                                                                                          \
        AIV_INFO_HINT;                                                                         \
        AivUrmaAllToAllVV2Mesh1D<type>(EXTERN_KERNEL_ARGS_CALL);                               \
    }                                                                                          \
    EXPORT_AIV_META_INFO(aiv_urma_alltoallv_##type)
#endif

AIV_COPY_DATA_TYPE_DEF(AIV_URMA_ALL_TO_ALL_V_KERNEL_BATCH_DEF);
#endif // defined(__DAV_920R1__) || ((ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000))
#endif // AIV_ALL_TO_ALL_V_OP_H

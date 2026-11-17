/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_GATHER_OP_H
#define AIV_ALL_GATHER_OP_H

#include "aiv_communication_base_v2.h"
#include "aiv_all_gather_mesh_1d.h"
// 960(A6/dav-920r1)暂不注册URMA kernel，其编译环境的include路径不含version/目录
#if !defined(__DAV_920R1__)
#include "version/hcomm_version.h"
#include "version/asc_devkit_version.h"
#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
#include "aiv_urma_all_gather_mesh_1d.h"
#endif
#endif
#include "aiv_all_gather_mesh_1d_write.h"

using namespace AscendC;

#if defined(HCCL_AIV_ACLGRAPH_SK)
#define AIV_ALLGATHER_KERNEL_DECL(type) extern "C" __aicore__ void aiv_all_gather_##type##_inner(KERNEL_ARGS_DEF)

// isWrite为host下发的运行期参数：true走写语义实现，false保持原有读语义实现。
#define AIV_ALLGATHER_KERNEL_DEF(type)                                        \
    extern "C" __aicore__ void aiv_all_gather_##type##_inner(KERNEL_ARGS_DEF) \
    {                                                                         \
        AIV_INFO_HINT;                                                        \
        if (isWrite) {                                                        \
            AivAllGatherV2Mesh1DWrite<type>(KERNEL_ARGS_CALL);                \
            return;                                                           \
        }                                                                     \
        AivAllGatherV2Mesh1D<type>(KERNEL_ARGS_CALL);                         \
    }

#if defined(BUILD_SK_FUNC) && defined(SK_FUNC_ID)
#define AIV_ALLGATHER_KERNEL_BATCH_DEF(type) \
    AIV_ALLGATHER_KERNEL_DECL(type);         \
    SK_BIND_FUNC_DEF(aiv_all_gather_##type, SK_FUNC_ID)
#else
#define AIV_ALLGATHER_KERNEL_BATCH_DEF(type) \
    AIV_ALLGATHER_KERNEL_DEF(type);          \
    GLOBAL_FUNC_DEF(aiv_all_gather_##type);  \
    SuperKernelBind(aiv_all_gather_##type)
#endif
#else // 未启用 aclgraph SK（960 等）：保持原 __global__ 入口导出
#define AIV_ALLGATHER_KERNEL_BATCH_DEF(type)                                     \
    extern "C" __global__ __aicore__ void aiv_all_gather_##type(KERNEL_ARGS_DEF) \
    {                                                                            \
        AIV_INFO_HINT;                                                           \
        if (isWrite) {                                                           \
            AivAllGatherV2Mesh1DWrite<type>(KERNEL_ARGS_CALL);                   \
            return;                                                              \
        }                                                                        \
        AivAllGatherV2Mesh1D<type>(KERNEL_ARGS_CALL);                            \
    }                                                                            \
    EXPORT_AIV_META_INFO(aiv_all_gather_##type)
#endif

// 定义各算子各数据类型Kernel入口
AIV_COPY_DATA_TYPE_DEF(AIV_ALLGATHER_KERNEL_BATCH_DEF);

// URMA AllGather kernel入口（编入同一个.o文件）
// 960(A6/dav-920r1)只保留空kernel入口，不包含URMA实现（其编译环境无version/和hcomm相关头文件）
#if defined(__DAV_920R1__) || ((ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000))
#if defined(HCCL_AIV_ACLGRAPH_SK)
#define AIV_URMA_ALL_GATHER_KERNEL_DECL(type) \
    extern "C" __aicore__ void aiv_urma_all_gather_##type##_inner(KERNEL_ARGS_DEF)

#if defined(__DAV_920R1__)
#define AIV_URMA_ALL_GATHER_KERNEL_DEF(type) \
    extern "C" __aicore__ void aiv_urma_all_gather_##type##_inner(KERNEL_ARGS_DEF) { AIV_INFO_HINT; }
#else
#define AIV_URMA_ALL_GATHER_KERNEL_DEF(type)                                       \
    extern "C" __aicore__ void aiv_urma_all_gather_##type##_inner(KERNEL_ARGS_DEF) \
    {                                                                              \
        AIV_INFO_HINT;                                                             \
        return AivUrmaAllGatherV2Mesh1D<type>(KERNEL_ARGS_CALL);                   \
    }
#endif

#if defined(BUILD_SK_FUNC) && defined(SK_FUNC_ID)
#define AIV_URMA_ALL_GATHER_KERNEL_BATCH_DEF(type) \
    AIV_URMA_ALL_GATHER_KERNEL_DECL(type);         \
    SK_BIND_FUNC_DEF(aiv_urma_all_gather_##type, SK_FUNC_ID)
#else
#define AIV_URMA_ALL_GATHER_KERNEL_BATCH_DEF(type) \
    AIV_URMA_ALL_GATHER_KERNEL_DEF(type);          \
    GLOBAL_FUNC_DEF(aiv_urma_all_gather_##type);   \
    SuperKernelBind(aiv_urma_all_gather_##type)
#endif
#else
#if defined(__DAV_920R1__)
#define AIV_URMA_ALL_GATHER_KERNEL_BATCH_DEF(type)                                                       \
    extern "C" __global__ __aicore__ void aiv_urma_all_gather_##type(KERNEL_ARGS_DEF) { AIV_INFO_HINT; } \
    EXPORT_AIV_META_INFO(aiv_urma_all_gather_##type)
#else
#define AIV_URMA_ALL_GATHER_KERNEL_BATCH_DEF(type)                                    \
    extern "C" __global__ __aicore__ void aiv_urma_all_gather_##type(KERNEL_ARGS_DEF) \
    {                                                                                 \
        AIV_INFO_HINT;                                                                \
        return AivUrmaAllGatherV2Mesh1D<type>(KERNEL_ARGS_CALL);                      \
    }                                                                                 \
    EXPORT_AIV_META_INFO(aiv_urma_all_gather_##type)
#endif
#endif

AIV_COPY_DATA_TYPE_DEF(AIV_URMA_ALL_GATHER_KERNEL_BATCH_DEF);
#endif // defined(__DAV_920R1__) || ((ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000))

#endif // AIV_ALL_GATHER_OP_H

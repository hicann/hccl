/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALGO_DIMS_H
#define HCCL_ALGO_DIMS_H

#include <stdint.h>
#include "hccl/hccl_types.h"

/* ===== Engine 维度（5 种）===== */
typedef enum {
    HCCL_ENGINE_AICPU = 0,
    HCCL_ENGINE_CCU_MS,
    HCCL_ENGINE_CCU_SCHED,
    HCCL_ENGINE_AIV,
    HCCL_ENGINE_DPU,
    HCCL_ENGINE_COUNT
} hcclEngineType_t;

/* ===== Executor 维度（5 种）===== */
typedef enum {
    HCCL_EXEC_SEQUENCE = 0,
    HCCL_EXEC_SOLE,
    HCCL_EXEC_PARALLEL,
    HCCL_EXEC_PIPILINE,
    HCCL_EXEC_CONCUR,
    HCCL_EXEC_COUNT
} hcclExecutorType_t;

/* ===== Template 维度（6 种）===== */
typedef enum {
    HCCL_TPL_MESH = 0,
    HCCL_TPL_NHR,
    HCCL_TPL_MESH_TWO_SHOT,
    HCCL_TPL_MESH_ONE_SHOT,
    HCCL_TPL_MESH_CHUNK,
    HCCL_TPL_MESH_2DIE,
    HCCL_TPL_COUNT
} hcclTemplateType_t;

/* ===== OpType 查找函数（基于 HcclCMDType，定义于 hccl_types.h）===== */

static inline const char* HcclOpTypeToPascal(HcclCMDType opType)
{
    switch (opType) {
        case HCCL_CMD_ALLREDUCE:
            return "AllReduce";
        case HCCL_CMD_ALLGATHER:
            return "AllGather";
        case HCCL_CMD_BROADCAST:
            return "Broadcast";
        case HCCL_CMD_REDUCE:
            return "Reduce";
        case HCCL_CMD_REDUCE_SCATTER:
            return "ReduceScatter";
        case HCCL_CMD_SCATTER:
            return "Scatter";
        case HCCL_CMD_ALLTOALL:
            return "AllToAll";
        case HCCL_CMD_ALLTOALLV:
            return "AllToAllV";
        case HCCL_CMD_ALLTOALLVC:
            return "AllToAllVC";
        case HCCL_CMD_ALLGATHER_V:
            return "AllGatherV";
        case HCCL_CMD_REDUCE_SCATTER_V:
            return "ReduceScatterV";
        case HCCL_CMD_SEND:
            return "Send";
        case HCCL_CMD_RECEIVE:
            return "Recv";
        case HCCL_CMD_BARRIER:
            return "Barrier";
        case HCCL_CMD_BATCH_SEND_RECV:
            return "BatchSendRecv";
        default:
            return nullptr;
    }
}

#endif /* HCCL_ALGO_DIMS_H */

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_HCCL_TYPES_H
#define BSR_HOST_TEST_HCCL_TYPES_H

// CANN 公共类型的最小替身：生产 hccl_aiv_utils.h 依赖 hccl_types.h / acl_rt.h，本文件只补齐这些头里
// 被 host 侧代码真正引用到的符号，其余一律改由生产头提供（例如 ExtraArgsBsr / AivBsrItemArgs 的
// sizeof 与偏移断言必须在 UT 里真跑一遍）。枚举值按 CANN 实际顺序排列，DATATYPE_SIZE_TABLE 依赖该顺序。

#include <stdint.h>

typedef void* HcclComm;
typedef uint64_t ThreadHandle;

// 与生产 alg_param.h 的取值一致（AivOpArgs::numBlocks 的默认值来源）；被测路径总会覆盖它。
constexpr uint32_t MAX_NUM_BLOCKS = 56;

enum HcclResult {
    HCCL_SUCCESS = 0,
    HCCL_E_PARA = 1,
    HCCL_E_PTR = 2,
    HCCL_E_INTERNAL = 3,
    HCCL_E_NOT_SUPPORT = 4,
    HCCL_E_TIMEOUT = 5,
    HCCL_E_AGAIN = 6
};

enum HcclDataType {
    HCCL_DATA_TYPE_INT8 = 0,
    HCCL_DATA_TYPE_INT16 = 1,
    HCCL_DATA_TYPE_INT32 = 2,
    HCCL_DATA_TYPE_FP16 = 3,
    HCCL_DATA_TYPE_FP32 = 4,
    HCCL_DATA_TYPE_FP64 = 5,
    HCCL_DATA_TYPE_BFP16 = 6,
    HCCL_DATA_TYPE_INT64 = 7,
    HCCL_DATA_TYPE_UINT8 = 8,
    HCCL_DATA_TYPE_UINT16 = 9,
    HCCL_DATA_TYPE_UINT32 = 10,
    HCCL_DATA_TYPE_UINT64 = 11,
    HCCL_DATA_TYPE_RESERVED = 12
};

enum HcclReduceOp { HCCL_REDUCE_SUM = 0, HCCL_REDUCE_PROD = 1, HCCL_REDUCE_MAX = 2, HCCL_REDUCE_MIN = 3 };

enum HcclCMDType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_ALLREDUCE = 1,
    HCCL_CMD_ALLGATHER = 2,
    HCCL_CMD_REDUCE_SCATTER = 3,
    HCCL_CMD_ALLTOALL = 4,
    HCCL_CMD_ALLTOALLV = 5,
    HCCL_CMD_BATCH_SEND_RECV = 6,
    HCCL_CMD_MAX = 7
};

enum class HcclSendRecvType { HCCL_SEND = 0, HCCL_RECV = 1 };

struct HcclSendRecvItem {
    void* buf = nullptr;
    uint64_t count = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    uint32_t remoteRank = 0;
    HcclSendRecvType sendRecvType = HcclSendRecvType::HCCL_SEND;
};

enum class HcclMemType { HCCL_MEM_TYPE_DEVICE = 0, HCCL_MEM_TYPE_HOST = 1 };

struct HcclMem {
    HcclMemType type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
    void* addr = nullptr;
    uint64_t size = 0;
};

enum class HcclDevType : uint8_t { DEV_TYPE_910B = 0, DEV_TYPE_910_93 = 1, DEV_TYPE_COUNT = 2 };

// 生产侧供 ClearAivTagCb 回调用，被测路径不注册该回调，这里只需类型可解析。
enum class HcclCommStatePhase { HCCL_COMM_STATE_PHASE_INIT = 0, HCCL_COMM_STATE_PHASE_DESTROY = 1 };

#endif // BSR_HOST_TEST_HCCL_TYPES_H

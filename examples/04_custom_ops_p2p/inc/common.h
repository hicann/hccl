/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUSTOM_COMMON_H
#define CUSTOM_COMMON_H

#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hcomm_primitives.h"
#include "acl/acl_rt.h"
#include "log.h"

namespace ops_hccl_p2p {

#define UNLIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 0)))

#define ACLCHECK(cmd)                                                                                           \
    do {                                                                                                        \
        aclError ret = cmd;                                                                                     \
        if (ret != ACL_SUCCESS) {                                                                               \
            HCCL_ERROR("acl interface return err %s:%d, retcode: %d.\n", __FILE__, __LINE__, ret);              \
            if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {                                                        \
                HCCL_ERROR("memory allocation error, check whether the current memory space is sufficient.\n"); \
            }                                                                                                   \
            return HCCL_E_RUNTIME;                                                                              \
        }                                                                                                       \
    } while (0)

constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr uint32_t CUSTOM_TIMEOUT = 1800;

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;

constexpr uint32_t AICPU_CONTROL_NOTIFY_NUM = 2;

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct AlgResourceCtx {
    ThreadHandle threadHandle;
    CommBuffer localBuffer;
    CommBuffer remoteBuffer;
    ChannelHandle channelHandle;
    uint32_t notifyIds[AICPU_CONTROL_NOTIFY_NUM]; // aicpu 模式下device侧控制notify
};

struct OpParam {
    char tag[TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    void* inputPtr = nullptr;
    void* outputPtr = nullptr;
    uint64_t count = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    AlgResourceCtx* resCtx = nullptr;
};

constexpr uint32_t SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};
}
#endif

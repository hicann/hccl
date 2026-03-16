/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_COMMON_H
#define OPS_HCCL_CUSTOM_COMMON_H

#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hcomm_primitives.h"
#include "acl/acl_rt.h"
#include "log.h"
#include "extra_args.h"

namespace ops_hccl_allgather {

constexpr uint32_t CUSTOM_TIMEOUT = 1800;

constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH;

constexpr uint64_t AIV_TAG_BUFF_LEN = 2 * 1024 * 1024; // 2MB

struct OpParam {
    char tag[TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    
    uint64_t buffIn = 0; // cclMem address
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint64_t xRankSize = 0;
    uint64_t yRankSize = 0;
    uint64_t zRankSize = 0;
    uint64_t len = 0;
    uint32_t dataType = 0;
    uint32_t reduceOp = 0;
    uint32_t root = 0;
    uint32_t tagId = 0; // Maps to 'tag' in kernel
    
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    uint64_t repeatNum = 0;
    uint64_t inputRepeatStride = 0;
    uint64_t outputRepeatStride = 0;
    
    bool isOpBase = false;
    
    uint64_t headCountMem = 0;
    uint64_t tailCountMem = 0;
    uint64_t addOneMem = 0;
    uint32_t counterMemSize = 0;
    bool isEnableCounter = false;
    
    ExtraArgs extraArgs;
};

}

#endif // OPS_HCCL_CUSTOM_COMMON_H

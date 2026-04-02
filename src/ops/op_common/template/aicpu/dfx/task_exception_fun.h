/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_EXCEPTION_FUN_H
#define TASK_EXCEPTION_FUN_H

#include <string>
#include "hccl/base.h"
#include "alg_param.h"
#include "hccl_diag.h"

namespace ops_hccl {

struct ScatterOpInfo {
    char algTag[ALG_TAG_LENGTH]; // 保存资源的key值，和算法绑定
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    void* inputPtr = nullptr;
    void* outputPtr = nullptr;
    u32 root = INVALID_VALUE_RANKID;
    u64 count;
    HcclDataType dataType;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
};

HcclResult CreateScatter(OpParam *param, ScatterOpInfo *opInfo);

void GetScatterOpInfo(const void *opInfo, char *outPut, size_t size);

HcclResult ConvertToHcclDfxOpInfo(OpParam *param, HcclDfxOpInfo *dfxOpInfo);
}
#endif
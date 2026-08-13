/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_AICPU_TASK_CACHE_KEY_H
#define OPS_HCCL_AICPU_TASK_CACHE_KEY_H

#include <cstdint>
#include <string>

#include "hccl_types.h"
#include "alg_param.h"

namespace ops_hccl {

class AicpuTaskCacheKey {
public:
    AicpuTaskCacheKey() = delete;

    // 拼接cacheTag
    static HcclResult GetAicpuTaskCacheTag(const OpParam& param, uint64_t inputSize, std::string& cacheTag);
};

} // namespace ops_hccl

#endif

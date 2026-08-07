/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AICPU_TASK_CACHE_POLICY_H
#define HCCL_AICPU_TASK_CACHE_POLICY_H

#include "alg_param.h"

namespace ops_hccl {

class AicpuTaskCachePolicy {
public:
    static HcclResult IsAicpuTaskCacheEnable(
        const OpParam &param, const AlgResourceCtxSerializable &resCtx, bool &isCacheEnable);

private:
    static bool IsTopoSupported(const AlgResourceCtxSerializable &resCtx);
    static HcclResult IsInplaceForCache(const OpParam &param, const uint32_t rankSize, bool &isInplace);
    static bool IsOpTypeSupported(const OpParam &param);
};

} // namespace ops_hccl
#endif

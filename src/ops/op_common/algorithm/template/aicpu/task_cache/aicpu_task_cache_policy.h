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
    static HcclResult
    IsAicpuTaskCacheEnable(const OpParam& param, const AlgResourceCtxSerializable& resCtx, bool& isCacheEnable);

private:
    static bool IsTopoSupported(const AlgResourceCtxSerializable& resCtx);
    // isInplace: input/output 地址区间是否重叠(Broadcast 的 input==output 豁免场景返回 false)
    // isEmptyInOut: inputSize 与 outputSize 同时为 0(A3 场景), 地址类型无法区分, 维持屏蔽
    static HcclResult
    IsInplaceForCache(const OpParam& param, const uint32_t rankSize, bool& isInplace, bool& isEmptyInOut);
    static bool IsOpTypeSupported(const OpParam& param);
    static HcclResult
    IsAlltoallvCacheSupported(const OpParam& param, const AlgResourceCtxSerializable& resCtx, bool& isSupported);
};

} // namespace ops_hccl
#endif

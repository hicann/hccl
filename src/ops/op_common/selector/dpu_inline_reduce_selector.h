/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_INLINE_REDUCE_SELECTOR_H
#define DPU_INLINE_REDUCE_SELECTOR_H

#include "alg_param.h"

namespace ops_hccl {

enum class DpuInlineReduceDecisionReason : u8 {
    SUPPORTED = 0,
    CAPABILITY_NOT_SUPPORTED,
    DEVICE_TYPE_NOT_SUPPORTED,
    DATA_TYPE_NOT_SUPPORTED,
    REDUCE_OP_NOT_SUPPORTED,
    DETERMINISTIC_STRICT,
    PRIMITIVE_NOT_SUPPORTED
};

struct DpuInlineReduceDecision {
    bool supported = false;
    DpuInlineReduceDecisionReason reason = DpuInlineReduceDecisionReason::CAPABILITY_NOT_SUPPORTED;
};

// Checks the aggregated communication-domain capability and HCOMM write-reduce primitive availability.
DpuInlineReduceDecision DpuInlineReduceCustomCheck(const TopoInfoWithNetLayerDetails* topoInfo);

// Logs the common decision for the SelectorEngine topology custom check.
bool CheckDpuInlineReduceTopo(const TopoInfoWithNetLayerDetails* topoInfo);

const char* GetDpuInlineReduceDecisionReason(DpuInlineReduceDecisionReason reason);

// Checks the complete condition set shared by legacy selection and SelectorEngine fallback filtering.
bool CanUseDpuInlineReduce(const OpParam& opParam, const TopoInfoWithNetLayerDetails* topoInfo);

} // namespace ops_hccl

#endif

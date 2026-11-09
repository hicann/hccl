/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_inline_reduce_selector.h"

#include <mutex>

#include "alg_attrs.h"
#include "hccl_common.h"
#include "hcomm_primitives_dl.h"
#include "log.h"
#include "order_preserved_common.h"

namespace ops_hccl {
namespace {

    bool IsReduceOpSupported(HcclReduceOp reduceOp)
    {
        return reduceOp == HCCL_REDUCE_SUM || reduceOp == HCCL_REDUCE_MAX || reduceOp == HCCL_REDUCE_MIN;
    }

    void LogDpuInlineReduceDecision(const DpuInlineReduceDecision& decision)
    {
        HCCL_INFO(
            "[DpuInlineReduceSelector] %s: reason=%s.", decision.supported ? "matched" : "filtered",
            GetDpuInlineReduceDecisionReason(decision.reason));
    }

} // namespace

const char* GetDpuInlineReduceDecisionReason(DpuInlineReduceDecisionReason reason)
{
    switch (reason) {
        case DpuInlineReduceDecisionReason::SUPPORTED:
            return "SUPPORTED";
        case DpuInlineReduceDecisionReason::CAPABILITY_NOT_SUPPORTED:
            return "CAPABILITY_NOT_SUPPORTED";
        case DpuInlineReduceDecisionReason::DEVICE_TYPE_NOT_SUPPORTED:
            return "DEVICE_TYPE_NOT_SUPPORTED";
        case DpuInlineReduceDecisionReason::DATA_TYPE_NOT_SUPPORTED:
            return "DATA_TYPE_NOT_SUPPORTED";
        case DpuInlineReduceDecisionReason::REDUCE_OP_NOT_SUPPORTED:
            return "REDUCE_OP_NOT_SUPPORTED";
        case DpuInlineReduceDecisionReason::DETERMINISTIC_STRICT:
            return "DETERMINISTIC_STRICT";
        case DpuInlineReduceDecisionReason::PRIMITIVE_NOT_SUPPORTED:
            return "PRIMITIVE_NOT_SUPPORTED";
        default:
            return "UNKNOWN";
    }
}

DpuInlineReduceDecision DpuInlineReduceCustomCheck(const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (!topoInfo->allDpuSupportInlineReduce) {
        return {false, DpuInlineReduceDecisionReason::CAPABILITY_NOT_SUPPORTED};
    }
    if (!HcommIsSupportHcommWriteReduceWithNotifyOnThread()) {
        static std::once_flag warningFlag;
        std::call_once(warningFlag, []() {
            HCCL_WARNING(
                "[DpuInlineReduceSelector] filtered: reason=%s, write-reduce primitive symbol is unavailable.",
                GetDpuInlineReduceDecisionReason(DpuInlineReduceDecisionReason::PRIMITIVE_NOT_SUPPORTED));
        });
        return {false, DpuInlineReduceDecisionReason::PRIMITIVE_NOT_SUPPORTED};
    }
    return {true, DpuInlineReduceDecisionReason::SUPPORTED};
}

bool CheckDpuInlineReduceTopo(const TopoInfoWithNetLayerDetails* topoInfo)
{
    const DpuInlineReduceDecision decision = DpuInlineReduceCustomCheck(topoInfo);
    LogDpuInlineReduceDecision(decision);
    return decision.supported;
}

bool CanUseDpuInlineReduce(const OpParam& opParam, const TopoInfoWithNetLayerDetails* topoInfo)
{
    DpuInlineReduceDecision decision;
    if (!EqualToOrGreaterThan950(topoInfo->deviceType)) {
        decision = {false, DpuInlineReduceDecisionReason::DEVICE_TYPE_NOT_SUPPORTED};
    } else if (UNSUPPORTED_64BIT.count(opParam.DataDes.dataType) > 0) {
        decision = {false, DpuInlineReduceDecisionReason::DATA_TYPE_NOT_SUPPORTED};
    } else if (!IsReduceOpSupported(opParam.reduceType)) {
        decision = {false, DpuInlineReduceDecisionReason::REDUCE_OP_NOT_SUPPORTED};
    } else if (IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize)) {
        decision = {false, DpuInlineReduceDecisionReason::DETERMINISTIC_STRICT};
    } else {
        decision = DpuInlineReduceCustomCheck(topoInfo);
    }
    LogDpuInlineReduceDecision(decision);
    return decision.supported;
}

} // namespace ops_hccl

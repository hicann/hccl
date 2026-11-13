/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BARRIER_RHD_UTILS_H
#define BARRIER_RHD_UTILS_H

#include "template_utils.h"

namespace ops_hccl {

struct BarrierRhdStep {
    RhdPhase phase;
    u32 step;
    // 各阶段的对端统一使用用户 rank。
    u32 peerRank;
    u32 virtualPeer = INVALID_VALUE_RANKID;
    u32 algPeer = INVALID_VALUE_RANKID;
};

// 一次生成本 rank 的完整层内同步计划，覆盖前处理、主体和后处理。
HcclResult
BuildBarrierRhdSteps(u32 myRank, const std::vector<std::vector<u32>>& subCommRanks, std::vector<BarrierRhdStep>& steps);
HcclResult CalcBarrierRhdResources(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subCommRanks, bool hostDpu, AlgResourceRequest& resourceRequest);
HcclResult CheckBarrierRhdChannels(
    const std::vector<BarrierRhdStep>& steps, const std::map<u32, std::vector<ChannelInfo>>& channels,
    EndpointLocType expectedLocation);

} // namespace ops_hccl

#endif // BARRIER_RHD_UTILS_H

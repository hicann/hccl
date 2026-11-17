/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_temp_all_gather_mesh_1D_Z_axis_detour.h"

namespace ops_hccl {

AivTempAllGatherMesh1DZAxisDetour::AivTempAllGatherMesh1DZAxisDetour(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : AivTempAllGatherMesh1D(param, rankId, subCommRanks)
{}

AivTempAllGatherMesh1DZAxisDetour::~AivTempAllGatherMesh1DZAxisDetour() {}

HcclResult AivTempAllGatherMesh1DZAxisDetour::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    HCCL_INFO("[AivTempAllGatherMesh1DZAxisDetour][CalcRes] start");
    CHK_PRT_RET(
        topoInfo == nullptr, HCCL_ERROR("[AivTempAllGatherMesh1DZAxisDetour][CalcRes] topoInfo is nullptr"),
        HCCL_E_PARA);

    u32 threadNum = 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels));
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels));

    std::vector<HcclChannelDesc> mergedChannels;
    mergedChannels.insert(mergedChannels.end(), level0Channels.begin(), level0Channels.end());
    mergedChannels.insert(mergedChannels.end(), level1Channels.begin(), level1Channels.end());
    resourceRequest.channels.push_back(mergedChannels);
    resRequest_ = resourceRequest;

    HCCL_INFO(
        "[AivTempAllGatherMesh1DZAxisDetour][CalcRes] level0Channels[%zu] level1Channels[%zu] merged[%zu]",
        level0Channels.size(), level1Channels.size(), mergedChannels.size());
    return HCCL_SUCCESS;
}

std::vector<CostModelParam> AivTempAllGatherMesh1DZAxisDetour::CalcCostCoeff(CalcCostCoeffParam param)
{
    // URMA 特性灰度门禁：cost model 暂未建模，返回极大值使其在候选中排最后，
    // 仅当其他算法均不匹配时才被选中（与 executor 侧同款约定）。
    CostModelParam urmaCost{1e10f, 0.0f, 0.0f, 0.0f};
    return {urmaCost};
}

} // namespace ops_hccl

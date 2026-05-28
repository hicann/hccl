/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_hd.h"

namespace ops_hccl {

namespace {
u32 GetHDStepNum(u32 rankSize)
{
    u32 stepNum = 0;
    while ((1U << stepNum) < rankSize) {
        ++stepNum;
    }
    return stepNum;
}
}  // namespace

InsTempReduceScatterHD::InsTempReduceScatterHD(const OpParam &param, const u32 rankId,
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempReduceScatterNHR(param, rankId, subCommRanks)
{
}

InsTempReduceScatterHD::~InsTempReduceScatterHD() {}

HcclResult InsTempReduceScatterHD::GetStepInfoList(std::vector<AicpuNHRStepInfo> &stepInfoList)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    const u32 nSteps = GetHDStepNum(templateRankSize_);
    CHK_PRT_RET(templateRankSize_ != (1U << nSteps),
        HCCL_ERROR("[ReduceScatterHD][GetStepInfoList] rankSize[%u] is not power of 2", templateRankSize_),
        HCCL_E_PARA);

    stepInfoList.clear();
    stepInfoList.resize(nSteps);
    for (u32 step = 0; step < nSteps; ++step) {
        const u32 stride = 1U << (nSteps - step - 1);
        const u32 remoteRank = myAlgRank ^ stride;
        CHK_PRT_RET(remoteRank >= templateRankSize_,
            HCCL_ERROR("[ReduceScatterHD][GetStepInfoList] remoteRank[%u] out of range, rankSize[%u]", remoteRank,
                templateRankSize_),
            HCCL_E_PARA);

        const u32 txSliceIdx = remoteRank / stride * stride;
        const u32 rxSliceIdx = myAlgRank / stride * stride;

        AicpuNHRStepInfo &currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.myRank = myAlgRank;
        currStepInfo.nSlices = stride;
        currStepInfo.toRank = remoteRank;
        currStepInfo.fromRank = remoteRank;
        currStepInfo.txSliceIdxs.reserve(stride);
        currStepInfo.rxSliceIdxs.reserve(stride);
        for (u32 idx = 0; idx < stride; ++idx) {
            currStepInfo.txSliceIdxs.push_back(txSliceIdx + idx);
            currStepInfo.rxSliceIdxs.push_back(rxSliceIdx + idx);
            HCCL_DEBUG("[ReduceScatterHD][GetStepInfoList] idx[%u] txSliceIdx[%u] rxSliceIdx[%u]", idx,
                txSliceIdx + idx, rxSliceIdx + idx);
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl

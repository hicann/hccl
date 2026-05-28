/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_hd.h"

namespace ops_hccl {

InsTempAllGatherHD::InsTempAllGatherHD(const OpParam &param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
    : InsTempAllGatherNHR(param, rankId, subCommRanks)
{
}

InsTempAllGatherHD::~InsTempAllGatherHD() {}

HcclResult InsTempAllGatherHD::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo)
{
    (void)nSteps;
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = myAlgRank;

    CHK_PRT_RET(templateRankSize_ != (1U << nSteps),
        HCCL_ERROR("[AllGatherHD][GetStepInfo] rankSize[%u] is not power of 2", templateRankSize_),
        HCCL_E_PARA);

    const u32 stride = 1U << step;
    CHK_PRT_RET(stride >= templateRankSize_,
        HCCL_ERROR("[AllGatherHD][GetStepInfo] invalid step[%u], rankSize[%u]", step, templateRankSize_),
        HCCL_E_PARA);
    const u32 remoteRank = myAlgRank ^ stride;
    CHK_PRT_RET(remoteRank >= templateRankSize_,
        HCCL_ERROR("[AllGatherHD][GetStepInfo] remoteRank[%u] out of range, rankSize[%u]", remoteRank,
            templateRankSize_),
        HCCL_E_PARA);

    const u32 txSliceIdx = myAlgRank / stride * stride;
    const u32 rxSliceIdx = remoteRank / stride * stride;

    stepInfo.nSlices = stride;
    stepInfo.toRank = remoteRank;
    stepInfo.fromRank = remoteRank;

    for (u32 idx = 0; idx < stepInfo.nSlices; ++idx) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx + idx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx + idx);
        HCCL_DEBUG("[AllGatherHD][GetStepInfo] idx[%u] txSliceIdx[%u] rxSliceIdx[%u]", idx, txSliceIdx + idx,
            rxSliceIdx + idx);
    }
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl

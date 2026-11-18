/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_TO_ALL_HIER_STAGE_BASE_H_
#define HCCL_CCU_TEMP_ALL_TO_ALL_HIER_STAGE_BASE_H_

#include "utils.h"
#include "ccu_alg_template_base.h"
#include "dev_type.h"
#include "log.h"

namespace ops_hccl {

class CcuTempAlltoAllHierStageBase : public CcuAlgTemplateBase {
public:
    // FastLaunch 回放侧默认构造
    CcuTempAlltoAllHierStageBase() = default;
    explicit CcuTempAlltoAllHierStageBase(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex)
        : CcuAlgTemplateBase(param, rankId, subCommRanks),
          stageIndex_(stageIndex)
    {
        // 取组 0 大小（覆盖基类求和语义）
        templateRankSize_ = subCommRanks.empty() ? 0 : static_cast<u32>(subCommRanks[0].size());
        // CalcRankIdxInSubComm 不判空，空组须前置拦截
        myAlgRank_
            = subCommRanks.empty() ? INVALID_VALUE_RANKID : CalcRankIdxInSubComm(rankId, subCommRanks, myAlgRank_);
    }

    ~CcuTempAlltoAllHierStageBase() override = default;

    void SetStageRole(u32 stageIndex, u32 totalStages)
    {
        stageIndex_ = stageIndex;
        totalStages_ = totalStages;
        isFirstStage_ = (stageIndex == totalStages - 1);
        isLastStage_ = (stageIndex == 0);
        isMiddleStage_ = !isFirstStage_ && !isLastStage_;
    }

    bool IsFirstStage() const { return isFirstStage_; }
    bool IsLastStage() const { return isLastStage_; }
    bool IsMiddleStage() const { return isMiddleStage_; }
    u32 GetStageIndex() const { return stageIndex_; }
    u32 GetTemplateRankSize() const { return templateRankSize_; }
    u32 GetMyAlgRank() const { return myAlgRank_; }

    virtual std::string GetAlgoType() const = 0;

protected:
    HcclResult ValidateStageTopo(const TopoInfoWithNetLayerDetails* topoInfo) const
    {
        CHK_PTR_NULL(topoInfo);
        CHK_PRT_RET(
            subCommRanks_.size() != 1 || templateRankSize_ == 0 || myAlgRank_ == INVALID_VALUE_RANKID
                || topoInfo->userRankSize % templateRankSize_ != 0,
            HCCL_ERROR(
                "[CcuTempAlltoAllHierStageBase][%s] invalid stage topo contract: algoType[%s], subCommRanks "
                "size[%zu], groupSize[%zu], rankId[%u], myAlgRank[%u], templateRankSize[%u], userRankSize[%u].",
                __func__, GetAlgoType().c_str(), subCommRanks_.size(),
                subCommRanks_.empty() ? 0 : subCommRanks_[0].size(), myRank_, myAlgRank_, templateRankSize_,
                topoInfo->userRankSize),
            HcclResult::HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    u32 stageIndex_{0};
    u32 totalStages_{0};
    bool isFirstStage_{false};
    bool isLastStage_{false};
    bool isMiddleStage_{false};
    u32 myAlgRank_{INVALID_VALUE_RANKID};
};

inline HcclResult CalcUniformSliceParams(
    const OpParam& param, const TemplateDataParams& tempAlgParams, u64& totalCount, u64& dataTypeSize, u64& otherDim,
    u64& cclCount)
{
    CHK_PRT_RET(
        param.all2AllVDataDes.sendCounts == nullptr,
        HCCL_ERROR("[CcuTempAlltoAllHierStageBase][%s] sendCounts is nullptr.", __func__), HcclResult::HCCL_E_PARA);

    totalCount = *(reinterpret_cast<const u64*>(param.all2AllVDataDes.sendCounts));

    CHK_PRT_RET(
        static_cast<u32>(tempAlgParams.dataType) >= HCCL_DATA_TYPE_RESERVED,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierStageBase][%s] invalid dataType[%u].", __func__,
            static_cast<u32>(tempAlgParams.dataType)),
        HcclResult::HCCL_E_PARA);
    dataTypeSize = HCCL_SIZE_TABLE[tempAlgParams.dataType];
    otherDim = tempAlgParams.outputSliceStride;
    CHK_PRT_RET(
        dataTypeSize == 0 || otherDim == 0,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierStageBase][%s] invalid dataTypeSize[%llu] or otherDim[%llu]", __func__, dataTypeSize,
            otherDim),
        HcclResult::HCCL_E_INTERNAL);
    cclCount = tempAlgParams.inputSliceStride / dataTypeSize / otherDim;

    CHK_PRT_RET(
        cclCount == 0 || cclCount * dataTypeSize * otherDim != tempAlgParams.inputSliceStride,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierStageBase][%s] invalid inputSliceStride[%llu], cclCount[%llu], "
            "dataTypeSize[%llu], otherDim[%llu].",
            __func__, tempAlgParams.inputSliceStride, cclCount, dataTypeSize, otherDim),
        HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_ALL_TO_ALL_HIER_STAGE_BASE_H_

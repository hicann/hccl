/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_TO_ALL_HIER_STAGE_BASE_H
#define INS_TEMP_ALL_TO_ALL_HIER_STAGE_BASE_H

#include "alg_v2_template_base.h"

namespace ops_hccl {

constexpr u32 ALLTOALLV_DIRECT_FULLMESH_CONCURRENT_SIZE = 16;

class InsTempAlltoAllHierStageBase : public InsAlgTemplateBase {
public:
    explicit InsTempAlltoAllHierStageBase(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex)
        : InsAlgTemplateBase(param, rankId, subCommRanks),
          stageIndex_(stageIndex)
    {}

    ~InsTempAlltoAllHierStageBase() override = default;

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

    virtual std::string GetAlgoType() const = 0;

    using InsAlgTemplateBase::GetNotifyIdxMainToSub;
    using InsAlgTemplateBase::GetNotifyIdxSubToMain;

protected:
    u32 stageIndex_{0};
    u32 totalStages_{0};
    bool isFirstStage_{false};
    bool isLastStage_{false};
    bool isMiddleStage_{false};
};

} // namespace ops_hccl

#endif // !INS_TEMP_ALL_TO_ALL_HIER_STAGE_BASE_H

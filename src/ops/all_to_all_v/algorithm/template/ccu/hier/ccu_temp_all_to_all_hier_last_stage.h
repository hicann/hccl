/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_TO_ALL_HIER_LAST_STAGE_H_
#define HCCL_CCU_TEMP_ALL_TO_ALL_HIER_LAST_STAGE_H_

#include "utils.h"
#include "ccu_temp_all_to_all_hier_stage_base.h"

namespace ops_hccl {

/*
 * 二级序列 AllToAll 末 Stage（框内/板内组）：HCCL_BUFFER → OUTPUT。板内组全交换，
 * cclBuff 槽位列直写对端 OUTPUT，本地份经 GroupCopy 落本地 OUTPUT。
 */
class CcuTempAlltoAllHierLastStage : public CcuTempAlltoAllHierStageBase {
public:
    CcuTempAlltoAllHierLastStage() = default;
    CcuTempAlltoAllHierLastStage(
        const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex);
    ~CcuTempAlltoAllHierLastStage() override = default;

    std::string Describe() const override
    {
        return "Hierarchical AllToAll last stage (intra-board), stageIndex=" + std::to_string(stageIndex_)
               + ", rankSize=" + std::to_string(templateRankSize_);
    }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    std::string GetAlgoType() const override { return "LastMeshStage"; }

private:
    uint32_t srcSliceNum_ = 0; // 每对端承载的槽位数（板数 d1）
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_ALL_TO_ALL_HIER_LAST_STAGE_H_

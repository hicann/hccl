/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_INS_TEMP_ALLGATHER_NHR_NDA_INTER_H
#define BROADCAST_INS_TEMP_ALLGATHER_NHR_NDA_INTER_H

#include "alg_v2_template_base.h"

namespace ops_hccl {
class InsTempAllGatherNHRNDAInter : public InsAlgTemplateBase {
public:
    InsTempAllGatherNHRNDAInter() = default;
    InsTempAllGatherNHRNDAInter(const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempAllGatherNHRNDAInter() override = default;

    std::string Describe() const override { return "Broadcast Sequence InsTempAllGatherNHRNDAInter in CCL memory"; }
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override { return 1; }
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override {}
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override {}

private:
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const;
    HcclResult BuildSlices(
        const TemplateDataParams& tempAlgParams, const std::vector<u32>& sliceIdxs, void* buffer,
        std::vector<DataSlice>& slices) const;
    HcclResult RunNHR(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource) const;
};
} // namespace ops_hccl

#endif // BROADCAST_INS_TEMP_ALLGATHER_NHR_NDA_INTER_H

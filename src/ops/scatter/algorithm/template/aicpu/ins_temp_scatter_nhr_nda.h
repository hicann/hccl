/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_SCATTER_NHR_NDA_H
#define INS_TEMP_SCATTER_NHR_NDA_H

#include "alg_v2_template_base.h"

namespace ops_hccl {
// Scatter Sequence NHR NDA: same-slot ranks across boxes exchange rank-indexed slices
// over Host RoCE channels (allRankSliceSize/allRankDispls layout, repeatNum == 1).
class InsTempScatterSequenceNHRNDA : public InsAlgTemplateBase {
public:
    InsTempScatterSequenceNHRNDA() = default;
    InsTempScatterSequenceNHRNDA(const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempScatterSequenceNHRNDA() override = default;

    std::string Describe() const override { return "Scatter Sequence NHR NDA"; }
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 GetThreadNum() const override { return 1; }
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override { notifyIdxMainToSub.clear(); }
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override { notifyIdxSubToMain.clear(); }
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;

protected:
    HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rootAlgRank, AicpuNHRStepInfo& stepInfo) const;
    u32 GetRankFromMap(u32 rankIdx) const;
    HcclResult RunNHR(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const ThreadHandle& thread) const;
};
} // namespace ops_hccl

#endif // INS_TEMP_SCATTER_NHR_NDA_H

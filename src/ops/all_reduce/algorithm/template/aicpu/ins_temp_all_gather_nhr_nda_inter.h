/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_NHR_NDA_INTER_H
#define INS_TEMP_ALL_GATHER_NHR_NDA_INTER_H

#include "alg_v2_template_base.h"

namespace ops_hccl {

class InsTempAllGatherNhrNdaInter : public InsAlgTemplateBase {
public:
    InsTempAllGatherNhrNdaInter() = default;
    InsTempAllGatherNhrNdaInter(const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempAllGatherNhrNdaInter() override = default;

    std::string Describe() const override
    {
        std::string info = "Template of inter-node AllGather NHR NDA with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType) override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;

private:
    HcclResult GetStepInfo(u32 step, u32 stepNum, AicpuNHRStepInfo& stepInfo) const;
    u32 GetRankFromMap(u32 rankIdx) const;
    HcclResult LocalDataCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    HcclResult RunNHR(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const ThreadHandle& thread) const;
    HcclResult PostLocalCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) override {}
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override {}
};

} // namespace ops_hccl

#endif

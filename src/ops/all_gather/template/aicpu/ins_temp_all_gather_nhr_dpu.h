/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_NHR_DPU_H
#define INS_TEMP_ALL_GATHER_NHR_DPU_H

#pragma once

#include "alg_v2_template_base.h"

namespace ops_hccl {
class InsTempAllGatherNHRDPU : public InsAlgTemplateBase {
public:
    InsTempAllGatherNHRDPU() {}
    InsTempAllGatherNHRDPU(const OpParam& param, const u32 rankId,
                           const std::vector<std::vector<u32>> &subCommRanks);

    ~InsTempAllGatherNHRDPU() override {}

    std::string Describe() const override
    {
        std::string info = "Template of AllGather NHR with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType) override;
    HcclResult KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
                         const TemplateResource& templateResource) override;
    HcclResult DPUKernelRun(const TemplateDataParams& tempAlgParams,
                            const std::map<u32, std::vector<ChannelInfo>>& channels,
                            const u32 myRank,
                            const std::vector<std::vector<uint32_t>>& subCommRanks) override;

private:
    HcclResult GetStepInfo(uint32_t step, uint32_t nSteps, AicpuNHRStepInfo &stepInfo);
    u32 GetRankFromMap(const uint32_t rankIdx);
    HcclResult LocalDataCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    HcclResult RunNHR(const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels);
    HcclResult PostLocalCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override {}
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override {}
};
}

#endif
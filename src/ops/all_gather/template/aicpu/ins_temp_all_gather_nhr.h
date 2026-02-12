/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_NHR_H
#define INS_TEMP_ALL_GATHER_NHR_H

#include <cstring>
#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

class InsTempAllGatherNHR : public InsAlgTemplateBase {
public:
    explicit InsTempAllGatherNHR(const OpParam &param, const u32 rankId,
                                 const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAllGatherNHR() override;

    std::string Describe() const override
    {
        std::string info = "Template of all gather nhr with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         const TemplateResource &templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfo *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest &resourceReques) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() override;
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainnToSub) override {};
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override{};
private:
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo);

    HcclResult LocalDataCopy(const std::vector<ThreadHandle> &threads);
    HcclResult PostLocalCopy(const std::vector<ThreadHandle> &threads);
    HcclResult RunAllGatherNHR(const std::vector<ThreadHandle> &threads,
                               const std::map<u32, std::vector<ChannelInfo>> &channels);
    u32 GetRankFromMap(u32 algRankId);
    TemplateDataParams tempAlgParams_;
};

}  // namespace Hccl

#endif  // INS_TEMP_ALL_GATHER_NHR_H
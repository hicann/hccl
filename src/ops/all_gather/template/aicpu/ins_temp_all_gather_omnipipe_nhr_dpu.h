/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_OMNIPIPE_NHR_DPU_H
#define INS_TEMP_ALL_GATHER_OMNIPIPE_NHR_DPU_H

#pragma once

#include "ins_temp_all_gather_nhr_dpu.h"

namespace ops_hccl {
class InsTempAllGatherOmniPipeNHRDPU : public InsTempAllGatherNHRDPU {
public:
    InsTempAllGatherOmniPipeNHRDPU() {}
    InsTempAllGatherOmniPipeNHRDPU(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);

    ~InsTempAllGatherOmniPipeNHRDPU() override;

    std::string Describe() const override
    {
        std::string info = "Template of AllGather omniPipe NHR dpu with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;

    u64 GetThreadNum() const override;

private:
    HcclResult RunNHR(const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels)
        const override;
};
} // namespace ops_hccl

#endif

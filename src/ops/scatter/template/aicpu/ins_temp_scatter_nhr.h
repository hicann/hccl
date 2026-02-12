/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_SCATTER_NHR_H
#define INS_TEMP_SCATTER_NHR_H

#include <cstring>
#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
class InsTempScatterNHR : public InsAlgTemplateBase {
public:
    explicit InsTempScatterNHR(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempScatterNHR() override;

    std::string Describe() const override
    {
        std::string info = "Template of scatter NHR with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams &tempAlgParams,
                         const TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                        AlgResourceRequest& resourceRequest) override;
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo);
    u64 GetThreadNum() override;
    HcclResult GetRes(AlgResourceRequest &resourceReques) override;
    void SetRoot(u32 root);

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainnToSub) override {};
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override{};

private:
    HcclResult PreCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);
    HcclResult PostCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);
    HcclResult RunNHR(const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
        const TemplateDataParams &tempAlgParams);
    HcclResult BatchSend(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels,
        const ThreadHandle &thread, const TemplateDataParams &templateDataParams, u32 repeat) const;
    HcclResult BatchRecv(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels,
        const ThreadHandle &thread, const TemplateDataParams &templateDataParams, u32 repeat) const;
    HcclResult BatchSR(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels,
        const ThreadHandle &thread, const TemplateDataParams &templateDataParams, u32 repeat) const;
    u64 processSize_{0};
    u64 count_{0};
};

}  // namespace ops_hccl

#endif  // OPEN_HCCL_INS_TEMP_SCATTER_NHR_H
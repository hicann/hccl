/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_NHR_H
#define INS_TEMP_REDUCE_SCATTER_NHR_H

#include <cstring>
#include "alg_v2_template_base.h"
#include "executor_v2_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempReduceScatterNHR : public InsAlgTemplateBase {
public:
    explicit InsTempReduceScatterNHR(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                     const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempReduceScatterNHR() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter NHR with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                       AlgResourceRequest& resourceRequest)  override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& tempAlgParams,
                         const TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub);
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain);
private:
    HcclResult GetStepInfoList(std::vector<AicpuNHRStepInfo> &stepInfoList);
    HcclResult LocalDataCopy(const std::vector<ThreadHandle> &threads);
    HcclResult RunNHR(const std::vector<ThreadHandle> &threads);
    HcclResult PostLocalCopy(const std::vector<ThreadHandle> &threads);
    TemplateDataParams tempAlgParams_;
    std::map<u32, std::vector<ChannelInfo>> channels_;
};

} // namespace Hccl

#endif // !HCCLV2_INS_TEMP_REDUCE_SCATTER_NHR
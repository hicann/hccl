/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_NHR_H
#define INS_TEMP_REDUCE_NHR_H

#include <cstring>
#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

class InsTempReduceNHR : public InsAlgTemplateBase {
public:
    explicit InsTempReduceNHR(const OpParam &param, const u32 rankId,  // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>> &subCommRanks);

    ~InsTempReduceNHR() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter Mesh with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    // 现在的Kernel就是之前的GenExtIns
    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
        const TemplateResource &templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam &param, const TopoInfo *topoInfo, AlgResourceRequest &resourceRequest) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    HcclResult PostCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub);
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain);

    u64 GetThreadNum();

private:
    HcclResult CalcSlice(const u64 dataSize);
    HcclResult PreCopy(const TemplateDataParams &tempAlgParams);
    HcclResult RunReduceScatter(const std::map<u32, std::vector<ChannelInfo>> &channels);
    HcclResult RunGather(const std::map<u32, std::vector<ChannelInfo>> &channels);
    HcclResult PostCopy(const TemplateDataParams &tempAlgParams);

    HcclResult GetStepInfoList(std::vector<AicpuNHRStepInfo> &stepInfoList);
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo);
    std::pair<std::vector<DataSlice>, std::vector<DataSlice>> getTxRxSlices(
        const AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels);
    u32 getMyAlgRank() const;

    ThreadHandle thread_;
    BuffInfo buffInfo_;
    u64 processSize_{0};
    u64 count_{0};
    u32 myIdx_ = UINT32_MAX;  // 本rank在通信域内的索引

    BufferType reduceInBuffType_ = BufferType::INPUT;
    BufferType reduceOutBuffType_ = BufferType::OUTPUT;
    u64 reduceInBuffBaseOff_ = 0;
    u64 reduceOutBuffBaseOff_ = 0;
    RankSliceInfo sliceInfoVec_;
};

}  // namespace ops_hccl

#endif  // INS_TEMP_REDUCE_NHR_H
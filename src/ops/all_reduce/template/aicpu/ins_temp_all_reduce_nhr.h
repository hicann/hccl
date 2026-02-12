/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_REDUCE_NHR
#define INS_TEMP_ALL_REDUCE_NHR

#include <cstring>
#include "alg_v2_template_base.h"
#include "executor_base.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

struct NHRStepInfo {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfo() : nSlices(0) {}
};

struct NHRSliceInfo {
    u64 offset{0};
    u64 size{0};
    u64 count{0};

    NHRSliceInfo(const u64 offset, const u64 size, const u64 count) 
    : offset(offset), size(size), count(count) {}
};

class InsTempAllReduceNHR : public InsAlgTemplateBase {
public:
    explicit InsTempAllReduceNHR(const OpParam& param, const u32 rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAllReduceNHR() override;

    std::string Describe() const override
    {
        std::string info = "Template of all reduce NHR with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) override;
    u64 GetThreadNum() override;

    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;
    
    HcclResult KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
        const TemplateResource& templateResource) override;

private:
    HcclResult SplitData();

    HcclResult PreCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);
    HcclResult RunReduceScatter(const TemplateDataParams &tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads);
    HcclResult RunAllGather(const TemplateDataParams &tempAlgParams,
        const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads);
    HcclResult PostCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads);

    HcclResult GetReduceScatterStepInfoList(std::vector<NHRStepInfo> &stepInfoList);
    HcclResult GetAllGatherStepInfoList(std::vector<NHRStepInfo> &stepInfoList);
    u32 GetNHRStepNum();

    u32 dataTypeSize_{0};
    u64 count_{0};
    u64 processSize_{0};
    
    u32 myRankIdx_{0};
    std::vector<NHRSliceInfo> sliceInfoList_;
    std::vector<u32> rankList_;
    
    void* nhrInBuff_;
    void* nhrOutBuff_;
    u64 nhrInBuffBaseOff_ = 0;
    u64 nhrOutBuffBaseOff_ = 0;
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALL_REDUCE_NHR
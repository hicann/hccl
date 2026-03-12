/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCL_CCU_TEMP_ALLREDUCE_NHR_1D_MEM2MEM_H
#define HCCL_CCU_TEMP_ALLREDUCE_NHR_1D_MEM2MEM_H
#include "ccu_alg_template_base.h"
#include "utils.h"
#include "ins_temp_all_reduce_nhr.h"

namespace ops_hccl {

class CcuTempAllReduceNHRMem2Mem1D : public CcuAlgTemplateBase {
public:
    explicit CcuTempAllReduceNHRMem2Mem1D(const OpParam& param, 
                                                const u32 rankId,
                                                const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAllReduceNHRMem2Mem1D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of AllReduce ccu nhr 1D mem2mem with tempRankSize [%u].", subCommRanks_[0].size());
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;
    u64 GetThreadNum() const override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
private:
    HcclResult CalcSlice(const u64 dataSize, RankSliceInfo &sliceInfoVec) const;
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    HcclResult GetReduceScatterStepInfo(u32 step, NHRStepInfo &stepInfo) const;
    HcclResult GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo) const;
    HcclResult ProcessNHRStepInfo(HcclComm comm,
                                  std::vector<NHRStepInfo>& stepInfoVector,
                                  std::map<u32, u32>& rank2ChannelIdx, u32 enableDieNum,
                                  std::vector<std::vector<HcclChannelDesc>>& channelsPerDie);
    HcclResult SplitDataFor2Dies(uint64_t dataCount, uint64_t &die0Size, uint64_t &die1Size) const;
    HcclResult GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum);
    uint32_t mySubCommRank_ = 0;
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
};

} // namespace ops_hccl

#endif
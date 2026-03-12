/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_REDUCE_NHR_1D_MEM2MEM_H
#define HCCL_CCU_TEMP_REDUCE_NHR_1D_MEM2MEM_H

#include "ccu_alg_template_base.h"
#include "utils.h"
#include "ins_temp_all_reduce_nhr.h"

namespace ops_hccl {

class CcuTempReduceNHR1DMem2Mem : public CcuAlgTemplateBase {
public:
    explicit  CcuTempReduceNHR1DMem2Mem(const OpParam& param,
                                        const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempReduceNHR1DMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Reduce ccu nhr 1D mem2mem with tempRankSize [%u].", subCommRanks_[0].size());
    }

    void SetRoot(u32 root);

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 GetThreadNum() const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;

private:
    uint32_t mySubCommRank_ = 0;
    uint32_t rootId_        = 0;
    // rank对应的channel，size为1-2
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;

    HcclResult GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum);
    HcclResult GetReduceScatterStepInfo(u32 step, NHRStepInfo &stepInfo) const;
    HcclResult GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    HcclResult ProcessNHRStepInfo(HcclComm comm,
                                  std::vector<NHRStepInfo>& stepInfoVector, std::map<u32, u32>& rank2ChannelIdx,
                                  u32 enableDieNum, std::vector<std::vector<HcclChannelDesc>>& channelsPerDie);
    HcclResult SplitDataFor2Dies(const OpParam& param, const TemplateDataParams& templateDataParams, uint64_t& die0Size,
                                 uint64_t& die1Size) const;
    HcclResult CalcSliceInfoAllReduce(const u64 dataSize, RankSliceInfo &sliceInfoVec) const;    
};

}// namespace ops_hccl

#endif// HCCL_CCU_TEMP_REDUCE_NHR_1D_MEM2MEM_H
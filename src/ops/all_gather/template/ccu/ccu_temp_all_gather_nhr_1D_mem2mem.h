/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_CCU_TEMP_ALL_GATHER_NHR_1D
#define HCCL_CCU_TEMP_ALL_GATHER_NHR_1D
 
#include "utils.h"
#include "ccu_alg_template_base.h"

using NHRStepInfo = struct NHRStepInfoDef {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfoDef() : nSlices(0)
    {
    }
};

namespace ops_hccl {
class CcuTempAllGatherNHR1DMem2Mem : public CcuAlgTemplateBase {
public:
    explicit CcuTempAllGatherNHR1DMem2Mem(const OpParam& param, 
                                              const u32 rankId, // 传通信域的rankId，userRank
                                              const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAllGatherNHR1DMem2Mem() override;
 
    std::string Describe() const override
    {
        return StringFormat("Template of AllGather ccu nhr 1D mem2mem with tempRankSize [%u].",
                            subCommRanks_[0].size());
    }
 
    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
 
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
 
private:
    uint32_t mySubCommRank_ = 0;
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
    HcclResult GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum);
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    HcclResult ProcessNHRStepInfo(HcclComm comm,
                                  std::vector<NHRStepInfo>& stepInfoVector, std::map<u32, u32>& rank2ChannelIdx,
                                  u32 enableDieNum, std::vector<std::vector<HcclChannelDesc>>& channelsPerDie);
    HcclResult SplitDataFor2Dies(const OpParam& param, const TemplateDataParams& templateDataParams, uint64_t& die0Size,
                                 uint64_t& die1Size) const;
};
 
} // namespace ops_hccl
 
#endif // HCCL_CCU_TEMP_ALL_GATHER_NHR_1D
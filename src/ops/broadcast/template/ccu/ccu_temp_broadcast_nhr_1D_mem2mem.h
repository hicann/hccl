/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_BROADCAST_NHR_1D_MEM2MEM_H
#define HCCL_CCU_TEMP_BROADCAST_NHR_1D_MEM2MEM_H

#include "ccu_alg_template_base.h"
#include "ccu_kernel_broadcast_nhr1d_mem2mem.h"
#include "utils.h"

namespace ops_hccl {

class CcuTempBroadcastNHR1DMem2Mem : public CcuAlgTemplateBase {
public:
    explicit  CcuTempBroadcastNHR1DMem2Mem(const OpParam& param,
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempBroadcastNHR1DMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Broadcast ccu nhr 1D Mem2Mem with tempRankSize [%u].",
                            subCommRanks_[0].size());
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    void SetRoot(u32 root);
    u64 GetThreadNum() const override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
private:
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    HcclResult GetScatterStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    HcclResult GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    HcclResult ProcessNHRStepInfo(HcclComm comm, std::vector<NHRStepInfo>& stepInfoVector,
                                    std::map<u32, u32>& rank2ChannelIdx, u32 enableDieNum,
                                    std::vector<std::vector<HcclChannelDesc>>& channelsPerDie);
    HcclResult GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum);
    HcclResult SplitDataFor2Dies(const OpParam& param,
                                const TemplateDataParams& templateDataParams,
                                uint64_t& die0Size, uint64_t& die1Size) const;
    u32 mySubCommRank_ = 0;
    u32 subCommRootId_  = 0;
    u32 root_ = 0;
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
};
}// namespace ops_hccl

#endif// HCCL_CCU_TEMP_BROADCAST_NHR_1D_MEM2MEM_H
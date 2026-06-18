/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_SCATTER_NHR
#define HCCL_CCU_TEMP_SCATTER_NHR

#include "ccu_alg_template_base.h"
#include "ccu_kernel_scatter_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct KernalRunTempArgs {
    u32 kernelNum;
    uint64_t die0Size;
    uint64_t die1Size;
    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t scratchAddr;
    uint64_t token;
    uint64_t sliceSize;
    uint64_t repeatNum;
    uint64_t inputSliceStride;
    uint64_t outputSliceStride;
    uint64_t inputRepeatStride;
    uint64_t outputRepeatStride;
    uint64_t isOutputScratch;
    uint64_t isInputOutputEqual;
    uint64_t die0TailSize;
    uint64_t die1TailSize;
    uint64_t isSliceSizeZero;
};


class CcuTempScatterNHR1DMem2Mem : public CcuAlgTemplateBase {
public:
    CcuTempScatterNHR1DMem2Mem() = default;
    explicit CcuTempScatterNHR1DMem2Mem(const OpParam& param, 
                                              const u32 rankId, // 传通信域的rankId，userRank
                                              const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempScatterNHR1DMem2Mem() override;

    std::string Describe() const override
    {
        return StringFormat("Template of Scatter ccu nhr 1D mem2mem with tempRankSize [%u].",
                            subCommRanks_[0].size());
    }

    HcclResult KernelRun(const OpParam& param,
                        const TemplateDataParams& templateDataParams,
                        TemplateResource& templateResource) override;

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    
    u64 GetThreadNum() const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    void SetRoot(u32 root);
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;
private:
    uint32_t mySubCommRank_ = 0;
    uint32_t subCommRootId_ = 0;
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
    HcclResult CalcChannelDescs(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                std::vector<HcclChannelDesc> &channelDescs);
    HcclResult BuildKernelInfos(const OpParam &param, u32 enableDieNum,
                                const std::vector<NHRStepInfo> &stepInfoVector,
                                const std::map<u32, u32> &rank2ChannelIdx,
                                const std::vector<std::vector<HcclChannelDesc>> &channelsPerDie,
                                AlgResourceRequest &resourceRequest);
    HcclResult GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum);
    HcclResult GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo);
    HcclResult ProcessNHRStepInfo(HcclComm comm,
                                  std::vector<NHRStepInfo>& stepInfoVector, std::map<u32, u32>& rank2ChannelIdx,
                                  u32 enableDieNum, std::vector<std::vector<HcclChannelDesc>>& channelsPerDie);
    HcclResult SplitDataFor2Dies(const OpParam& param, const TemplateDataParams& templateDataParams, uint64_t& die0Size,
                                 uint64_t& die1Size) const;
    void FillKernelRunTempArgs(const TemplateDataParams &templateDataParams, KernalRunTempArgs &tempArgs) const;
    HcclResult FillKernelRunArgs(const KernalRunTempArgs &tempArgs, const TemplateDataParams &templateDataParams, const TemplateResource& templateResource) const;
    void SaveSubmitInfo(const KernalRunTempArgs &tempArgs, TemplateResource& templateResource) const;
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_SCATTER_NHR
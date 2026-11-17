/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_HD_OP_H
#define INS_TEMP_ALL_GATHER_HD_OP_H

#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

// HD(Halving-Doubling) 递归倍增单步信息：步 v 与对端 P = R XOR 2^v 交换互补的 aligned 窗口半区，
// 发送半窗为块 {W .. W+2^v-1}，接收半窗为块 {W XOR 2^v .. W XOR 2^v+2^v-1}，两半窗互补不重叠。
struct AicpuHDStepInfo {
    u32 step = 0;       // HD 步序 v，取值 0..log2(N)-1
    u32 myRank = 0;     // 算法内 rank R
    u32 peerRank = 0;   // 对端算法 rank P = R XOR 2^v
    u32 windowBase = 0; // 发送半窗基块号 W = R & ~(2^v - 1)；接收半窗基块号为 W XOR 2^v
    u32 nSlices = 0;    // 半窗块数 2^v
};

class InsTempAllGatherHDOp : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::HD};
    InsTempAllGatherHDOp() = default;
    explicit InsTempAllGatherHDOp(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempAllGatherHDOp() override;

    std::string Describe() const override
    {
        std::string info = "Template of all gather halving-doubling with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;

    HcclResult SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

protected:
    virtual HcclResult GetStepInfoHD(u32 step, AicpuHDStepInfo& stepInfo);
    u32 GetRankFromMap(const u32 algRankIdx) const;
    TemplateDataParams tempAlgParams_;
    bool isDmaRead_{false};

private:
    HcclResult CheckKernelRunParams(const TemplateResource& templateResource);
    HcclResult PrepareKernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    HcclResult SyncInterThreads(const std::vector<ThreadHandle>& threads, bool isPreSync);
    HcclResult RunChannelMainLoop(const TemplateResource& templateResource);
    HcclResult PrepareDataSplitForMultiChannel(const TemplateResource& templateResource);
    HcclResult LocalDataCopy(const std::vector<ThreadHandle>& threads, const u32& channelIdx);
    HcclResult PostLocalCopy(const ThreadHandle& thread, const u32& channelIdx);
    struct HDCalcInfo {
        u32 txBlk; // 发送半窗块号 W+i
        u32 rxBlk; // 接收半窗块号 (W XOR 2^v)+i
        u64 txPartialOffset;
        u64 rxPartialOffset;
        u64 txScratchOff;
        u64 rxScratchOff;
        u64 txOutOff;
        u64 rxOutOff;
        u64 txSliceSize;
        u64 rxSliceSize;
    };
    HDCalcInfo CalcSliceInfoHD(const AicpuHDStepInfo& stepInfo, u32 rpt, u32 i, u32 channelIdx) const;
    HcclResult BuildStepSlicesHD(
        const ChannelInfo& channelSend, const ChannelInfo& channelRecv, const AicpuHDStepInfo& stepInfo,
        const u32& channelIdx, std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices,
        std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices);
    HcclResult RunStepHD(
        const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32& channelIdx, u32 step);
    HcclResult RunAllGatherHD(
        const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32& channelIdx);
    u32 maxChannelsPerRank_ = 1;
    std::vector<u64> dataSplit_;
    std::vector<u64> dataOffset_;
    std::vector<u64> dataSplitTail_;
    std::vector<u64> dataOffsetTail_;
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_GATHER_HD_OP_H

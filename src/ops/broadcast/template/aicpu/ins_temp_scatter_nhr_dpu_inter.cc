/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "ins_temp_scatter_nhr_dpu_inter.h"
#include "dpu_alg_nhr_opt_wrapper.h"


namespace ops_hccl {
InsTempScatterNHRDPUInter::InsTempScatterNHRDPUInter(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}
 
InsTempScatterNHRDPUInter::~InsTempScatterNHRDPUInter()
{}
 
u64 InsTempScatterNHRDPUInter::GetThreadNum() const
{
    u64 threadNum = 1;
    return threadNum;
}
 
void InsTempScatterNHRDPUInter::SetRoot(u32 root)
{
    HCCL_INFO("[InsTempScatterNHRDPUInter][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root_);
    root_ = root;
}
 
HcclResult InsTempScatterNHRDPUInter::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    u32 threadNum = 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
 
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
 
u64 InsTempScatterNHRDPUInter::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return templateRankSize_;
}
 
HcclResult InsTempScatterNHRDPUInter::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo)
{
#ifndef AICPU_COMPILE
    u32 rankSize = templateRankSize_;
    u32 myAlgRank;
    u32 rootAlgRank;
    HCCL_INFO("[InsTempScatterNHRDPUInter][GetStepInfo][GetAlgRank]* root_ before getAlgRank [%u] ", root_);
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    HCCL_INFO("[InsTempScatterNHRDPUInter][GetStepInfo][GetAlgRank]* myRank_[%u], myAlgRank[%u]", myRank_, myAlgRank);
    CHK_RET(GetAlgRank(root_, subCommRanks_[0], rootAlgRank));
    HCCL_INFO("[InsTempScatterNHRDPUInter][GetStepInfo][GetAlgRank]* root_[%u], rootAlgRank[%u]", root_, rootAlgRank);
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.step = step;
    stepInfo.myRank = myRank_;
 
    u32 deltaRoot = (rootAlgRank + rankSize - myAlgRank) % rankSize;
    u32 deltaRankPair = 1 << step;
 
    // 数据份数和数据编号增量
    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);
 
    // 判断是否是2的幂
    u32 nRanks = 0;  // 本步需要进行收/发的rank数
    bool isPerfect = (rankSize & (rankSize - 1)) == 0;
    if (!isPerfect && step == nSteps - 1) {
        nRanks = rankSize - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }
 
    if (deltaRoot < nRanks) {  // 需要发
        HCCL_INFO("[InsTempScatterNHRDPUInter][GetStepInfo] Need to Send: deltaRoot[%u], nRanks[%d]", deltaRoot, nRanks);
        u32 sendTo = (myAlgRank + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }
        HCCL_INFO("[InsTempScatterNHRDPUInter][GetStepInfo] rankSize[%u], myAlgRank[%d], sendTo Idx[%u]", subCommRanks_[0].size(), myAlgRank, sendTo);
        stepInfo.toRank = subCommRanks_[0].at(sendTo);
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) {  // 需要收
        HCCL_INFO("[InsTempScatterNHRDPUInter][GetStepInfo] Need to Recv: deltaRoot[%u], nRanks[%d], deltaRankPair[%d]", deltaRoot, nRanks, deltaRankPair);
        u32 recvFrom = (myAlgRank + deltaRankPair) % rankSize;
        u32 rxSliceIdx = myAlgRank;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = rxSliceIdx;
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }
        HCCL_INFO("[InsTempScatterNHRDPUInter][GetStepInfo] rankSize[%u], myAlgRank[%d], recvFrom Idx[%u]", subCommRanks_[0].size(), myAlgRank, recvFrom);
        stepInfo.fromRank = subCommRanks_[0].at(recvFrom);
        stepInfo.nSlices = nSlices;
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempScatterNHRDPUInter::KernelRun(const OpParam& param, const TemplateDataParams &tempAlgParams,
    TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempScatterNHRDPUInter] Run Start");
    
    threadNum_ =  subCommRanks_.size();
    count_ = tempAlgParams.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    SetRoot(tempAlgParams.root);

    HCCL_INFO("[InsTempScatterNHRDPUInter] queNum_ =  [%d], threads size = [%d]", threadNum_, templateResource.threads.size());
    
    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempScatterNHRDPUInter] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }
    
    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempScatterNHRDPUInter] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
 
    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("[InsTempScatterNHRDPUInter] HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempScatterNHRDPUInter";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    dpuRunInfo.subCommRanks = subCommRanks_;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();

    u32 sendMsgId = 0;
    if (HcommSendRequest(reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
        static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId) != 0) {
        HCCL_ERROR("[InsTempScatterNHRDPUInter] HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempScatterNHRDPUInter] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);

    // 等待DPU数据传输，然后回写结果回来
    void *recvData = nullptr;
    u32 recvMsgId = 0;
    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempScatterNHRDPUInter] HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempScatterNHRDPUInter] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("[InsTempScatterNHRDPUInter] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempScatterNHRDPUInter] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
 
    HCCL_INFO("[InsTempScatterNHRDPUInter] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHRDPUInter::DPUKernelRun(const TemplateDataParams& tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunNHR(channels, tempAlgParams));
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHRDPUInter::LocalDataCopy(const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    uint32_t algRankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], algRankIdx));

    u64 sliceSize = tempAlgParams.allRankSliceSize.at(algRankIdx);
    u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(algRankIdx);
    u64 sliceOffset = tempAlgParams.allRankDispls.at(algRankIdx);

    if (sliceSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    for (uint64_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams.buffInfo.inBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBaseoff = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
 
        const u64 inOff = inBaseOff + sliceOffset;
        const u64 scOff = scratchBaseoff + sliceOffset;
 
        DataSlice srcSlices(tempAlgParams.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, scOff, sliceSize, sliceCount);
        HCCL_INFO("[InsTempScatterNHRDPUInter][LocalCopy] LocalDataCopy RankID [%d] algRankIdx [%d] "
            "srcOff[%d] dstOff[%d] sliceOffset[%d] sliceSize[%d].", myRank_, algRankIdx, inOff, scOff, sliceOffset,
            sliceSize);
        LocalCopy(templateResource.threads[0], srcSlices, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHRDPUInter::PostLocalCopy(const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    if (tempAlgParams.buffInfo.outputPtr == nullptr) {
        return HcclResult::HCCL_SUCCESS;
    }
 
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
        for (auto rank : subCommRanks_[0]) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            
            u64 sliceSize = tempAlgParams.allRankSliceSize.at(algRank);
            u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(algRank);
            u64 sliceOffset = tempAlgParams.allRankDispls.at(algRank);

            u64 scratchOffset = scratchBase + sliceOffset;
            u64 outOffset = outBaseOff + sliceOffset;
            DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, outOffset, sliceSize, sliceCount);
            HCCL_INFO("[InsTempScatterNHRDPUInter][LocalCopy] LocalDataCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                       "srcOff[%d] dstOff[%d] sliceOffset[%d] sliceSize[%d].",
                       myRank_, rank, algRank, scratchOffset, outOffset, sliceOffset, sliceSize);
            LocalCopy(templateResource.threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempScatterNHRDPUInter::RunNHR(const std::map<u32, std::vector<ChannelInfo>> &channels,
    const TemplateDataParams &tempAlgParam)
{
#ifndef AICPU_COMPILE
    // nhr主体部分
    SetRoot(tempAlgParam.root);
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    HCCL_INFO("[RunNHR] root_ at RunNHR [%u] ", root_);
    for (u32 r = 0; r < tempAlgParam.repeatNum; r++) {
        for (u32 step = 0; step < nSteps; step++) {
            AicpuNHRStepInfo stepInfo;
            GetStepInfo(step, nSteps, stepInfo);
            // 统一 BatchTransferNHR：内部自动处理 只发/只收/同对端/不同对端 四种场景
            CHK_RET(BatchTransferNHR(stepInfo, channels, tempAlgParam, r, myRank_, templateRankSize_));
        }
    }
#endif
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempScatterNHRDPUInter", InsTempScatterNHRDPUInter);

}  // namespace ops_hccl
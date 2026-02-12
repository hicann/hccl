/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_nhr.h"

namespace ops_hccl {
InsTempScatterNHR::InsTempScatterNHR(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks)
                                        : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempScatterNHR::~InsTempScatterNHR()
{}

u64 InsTempScatterNHR::GetThreadNum()
{
    u64 threadNum = 1;
    return threadNum;
}

void InsTempScatterNHR::SetRoot(u32 root)
{   
    HCCL_INFO("[InsTempScatterNHR][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root);
    root_ = root;
}

HcclResult InsTempScatterNHR::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                        AlgResourceRequest& resourceRequest)
{
    GetRes(resourceRequest);

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::GetRes(AlgResourceRequest &resourceRequest)
{
    // NHR算法主需要一条主流
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;  // 没有从流
    return HCCL_SUCCESS;
}

u64 InsTempScatterNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return templateRankSize_;
}

HcclResult InsTempScatterNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo)
{
    u32 rankSize = templateRankSize_;
    u32 myAlgRank;
    u32 rootAlgRank;
    HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] root_ before getAlgRank [%u] ", root_);
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);
    HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] myRank_[%u], myAlgRank[%u]", myRank_, myAlgRank);
    GetAlgRank(root_, subCommRanks_[0], rootAlgRank);
    HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] root_[%u], rootAlgRank[%u]", root_, rootAlgRank);
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
        HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] Need to Send: deltaRoot[%u], nRanks[%d]", deltaRoot, nRanks);
        u32 sendTo = (myAlgRank + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }
        HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] rankSize[%u], myAlgRank[%d], sendTo Idx[%u]", subCommRanks_[0].size(), myAlgRank, sendTo);
        stepInfo.toRank = subCommRanks_[0].at(sendTo);
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) {  // 需要收
        HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] Need to Recv: deltaRoot[%u], nRanks[%d], deltaRankPair[%d]", deltaRoot, nRanks, deltaRankPair);
        u32 recvFrom = (myAlgRank + deltaRankPair) % rankSize;
        u32 rxSliceIdx = myAlgRank;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = rxSliceIdx;
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }
        HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] rankSize[%u], myAlgRank[%d], recvFrom Idx[%u]", subCommRanks_[0].size(), myAlgRank, recvFrom);
        stepInfo.fromRank = subCommRanks_[0].at(recvFrom);
        stepInfo.nSlices = nSlices;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::KernelRun(const OpParam& param, const TemplateDataParams &tempAlgParams,
                     const TemplateResource& templateResource)
{
    threadNum_ =  subCommRanks_.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    HCCL_INFO("[InsTempScatterNHR] queNum_ =  [%d], threads size = [%d]", threadNum_, templateResource.threads.size());
    HCCL_INFO("[InsTempScatterNHR] Run Start");
    CHK_PRT_RET(threadNum_ != templateResource.threads.size(),
        HCCL_ERROR("[InsTempScatterNHR] Rank [%d], requiredThread Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    CHK_RET(PreCopy(tempAlgParams, templateResource.threads));
    CHK_RET(RunNHR(templateResource.channels, templateResource.threads, tempAlgParams));
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));
    HCCL_INFO("[InsTempScatterNHR] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::PreCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    if (u32(myRank_) != root_ || tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        return HCCL_SUCCESS;
    }

    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        for (u32 algRank = 0; algRank < templateRankSize_; algRank++) {
            u64 srcOffset = r * tempAlgParams.inputRepeatStride + tempAlgParams.inputSliceStride * algRank +
                            tempAlgParams.buffInfo.inBuffBaseOff;
            u64 dstOffset = r * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff +
                            algRank * tempAlgParams.sliceSize;
            DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, processSize_, count_);
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, processSize_, count_);
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    u32 myAlgRank;
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);
    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        u64 srcOffset = r * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff +
                        myAlgRank * tempAlgParams.sliceSize;
        u64 dstOffset = tempAlgParams.buffInfo.outBuffBaseOff + r * tempAlgParams.sliceSize;
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, processSize_, count_);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, processSize_, count_);
        if (tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER && srcOffset == dstOffset) {
            continue;
        }
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::RunNHR(const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParam)
{
    // nhr主体部分
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    HCCL_DEBUG("[RunNHR] root_ at RunNHR [%u] ", root_);
    for (u32 r = 0; r < tempAlgParam.repeatNum; r++) {
        for (u32 step = 0; step < nSteps; step++) {
            AicpuNHRStepInfo stepInfo;
            GetStepInfo(step, nSteps, stepInfo);
            // 只有Tx,使用send指令
            if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() == 0) {
                CHK_RET(BatchSend(stepInfo, channels, threads.at(0), tempAlgParam, r));
            }
            // 只有Rx，使用recv指令
            else if (stepInfo.txSliceIdxs.size() == 0 && stepInfo.rxSliceIdxs.size() > 0) {
                CHK_RET(BatchRecv(stepInfo, channels, threads.at(0), tempAlgParam, r));
            }
            // 既有Tx又有Rx，使用SendRecv指令
            else if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() > 0) {
                CHK_RET(BatchSR(stepInfo, channels, threads.at(0), tempAlgParam, r));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::BatchSend(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels,
    const ThreadHandle &thread, const TemplateDataParams &tempAlgParam, u32 repeat) const
{
    const ChannelInfo &linkSend = channels.at(stepInfo.toRank)[0];
    HCCL_INFO("[InsTempScatterNHR][BatchSend] myRank[%d], toRank[%d]", myRank_, stepInfo.toRank);
    void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs.at(i);
        u64 srcDstOffset = repeat * templateRankSize_ * tempAlgParam.sliceSize + tempAlgParam.buffInfo.hcclBuffBaseOff +
                           txId * tempAlgParam.sliceSize;
        DataSlice srcSlice = DataSlice(tempAlgParam.buffInfo.hcclBuff.addr, srcDstOffset, processSize_, count_);
        DataSlice dstSlice = DataSlice(remoteCclBuffAddr, srcDstOffset, processSize_, count_);
        srcSlices.push_back(srcSlice);
        dstSlices.push_back(dstSlice);
    }
    SlicesList txSlicesList({srcSlices}, {dstSlices});
    DataInfo sendData(linkSend, txSlicesList);
    CHK_PRT_RET(static_cast<HcclResult>(SendWrite(sendData, thread)),
        HCCL_ERROR("[InsTempScatterNHR] BatchSend failed"),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::BatchRecv(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels,
    const ThreadHandle &thread, const TemplateDataParams &tempAlgParam, u32 repeat) const
{
    const ChannelInfo &linkRecv = channels.at(stepInfo.fromRank)[0];
    std::vector<DataSlice> srcDstSlices;
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxId = stepInfo.rxSliceIdxs.at(i);
        u64 srcDstOffset = repeat * templateRankSize_ * tempAlgParam.sliceSize + tempAlgParam.buffInfo.hcclBuffBaseOff +
                           rxId * tempAlgParam.sliceSize;
        DataSlice srcDstSlice(tempAlgParam.buffInfo.hcclBuff.addr, srcDstOffset, processSize_, count_);
        srcDstSlices.push_back(srcDstSlice);
    }
    SlicesList rxSlicesList(srcDstSlices, srcDstSlices); // RecvWrite函数下远端地址不起作用
    DataInfo recvData(linkRecv, rxSlicesList);
    CHK_PRT_RET(static_cast<HcclResult>(RecvWrite(recvData, thread)),
        HCCL_ERROR("[InsTempScatterNHR] BatchRecv failed"),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::BatchSR(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels,
    const ThreadHandle &thread, const TemplateDataParams &tempAlgParam, u32 repeat) const
{
    const ChannelInfo &linkSend = channels.at(stepInfo.toRank)[0];
    const ChannelInfo &linkRecv = channels.at(stepInfo.fromRank)[0];
    void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs.at(i);
        u64 srcDstOffset = repeat * templateRankSize_ * tempAlgParam.sliceSize + tempAlgParam.buffInfo.hcclBuffBaseOff +
                           txId * tempAlgParam.sliceSize;
        DataSlice srcSlice(tempAlgParam.buffInfo.hcclBuff.addr, srcDstOffset, processSize_, count_);
        DataSlice dstSlice(remoteCclBuffAddr, srcDstOffset, processSize_, count_);
        txSrcSlices.push_back(srcSlice);
        txDstSlices.push_back(dstSlice);
    }
    std::vector<DataSlice> rxSrcDstSlices;
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxId = stepInfo.rxSliceIdxs.at(i);
        u64 srcDstOffset = repeat * templateRankSize_ * tempAlgParam.sliceSize + tempAlgParam.buffInfo.hcclBuffBaseOff +
                           rxId * tempAlgParam.sliceSize;
        DataSlice srcDstSlice(tempAlgParam.buffInfo.hcclBuff.addr, srcDstOffset, processSize_, count_);
        rxSrcDstSlices.push_back(srcDstSlice);
    }
    SendRecvInfo sendRecvInfo{
        {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcDstSlices, rxSrcDstSlices}}};

    CHK_PRT_RET(SendRecvWrite(sendRecvInfo, thread),
        HCCL_ERROR("[InsTempScatterNHR] RunNHR BatchSendRecv failed"),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl
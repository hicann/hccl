/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_to_all_hier_mesh_stage.h"
#include "alltoall_stage_template_registry.h"
#include "channel.h"
#include "executor_base.h"

namespace ops_hccl {

InsTempAlltoAllHierMeshStage::InsTempAlltoAllHierMeshStage(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex)
    : InsTempAlltoAllHierStageBase(param, rankId, subCommRanks, stageIndex)
{}

HcclResult InsTempAlltoAllHierMeshStage::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    u32 threadNum = templateRankSize_;
    resourceRequest.slaveThreadNum = (threadNum > 1) ? (threadNum - 1) : 0;
    for (u32 index = 0; index < resourceRequest.slaveThreadNum; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;

    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);

    HCCL_INFO(
        "[InsTempAlltoAllHierMeshStage][CalcRes] stageIndex[%u], rankSize[%u], slaveThreadNum[%u], "
        "channelNum[%u]",
        stageIndex_, templateRankSize_, resourceRequest.slaveThreadNum, static_cast<u32>(channels.size()));
    return HCCL_SUCCESS;
}

u64 InsTempAlltoAllHierMeshStage::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

void InsTempAlltoAllHierMeshStage::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = (templateRankSize_ > 1) ? (templateRankSize_ - 1) : 0;
    for (u32 i = 0; i < slaveThreadNum; i++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAlltoAllHierMeshStage::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 slaveThreadNum = (templateRankSize_ > 1) ? (templateRankSize_ - 1) : 0;
    for (u32 i = 0; i < slaveThreadNum; i++) {
        notifyIdxSubToMain.push_back(i);
    }
}

u32 InsTempAlltoAllHierMeshStage::FindMyAlgRank() const
{
    if (subCommRanks_.empty() || subCommRanks_[0].empty()) {
        return INVALID_ALG_RANK;
    }
    for (u32 i = 0; i < subCommRanks_[0].size(); i++) {
        if (subCommRanks_[0][i] == myRank_) {
            return i;
        }
    }
    return INVALID_ALG_RANK;
}

void InsTempAlltoAllHierMeshStage::InitStageParams(const TemplateDataParams& tempAlgParams, StageParams& params)
{
    dataType_ = tempAlgParams.dataType;
    dataTypeSize_ = HCCL_SIZE_TABLE[dataType_];
    params.otherDimRankSize = static_cast<u32>(tempAlgParams.outputSliceStride);
    cclBufferCountPerRank_ = (params.otherDimRankSize > 0) ?
                                 (tempAlgParams.inputSliceStride / dataTypeSize_ / params.otherDimRankSize) :
                                 0;
    params.stageRankSize = static_cast<u32>(subCommRanks_[0].size());
    params.srcGroupSize = params.otherDimRankSize;
    params.targetGroupSize = isFirstStage_ ? params.otherDimRankSize : params.stageRankSize;
    params.subSlotCount = isFirstStage_ ? templateRankSize_ : params.otherDimRankSize;
}

HcclResult InsTempAlltoAllHierMeshStage::ValidateAndInit(
    const TemplateDataParams& tempAlgParams, u32& myAlgRank, StageParams& params)
{
    if (subCommRanks_.empty() || subCommRanks_[0].empty()) {
        HCCL_ERROR("[InsTempAlltoAllHierMeshStage][KernelRun] subCommRanks is empty, stageIndex[%u]", stageIndex_);
        return HCCL_E_PARA;
    }
    myAlgRank = FindMyAlgRank();
    if (myAlgRank == INVALID_ALG_RANK) {
        HCCL_ERROR("[InsTempAlltoAllHierMeshStage][KernelRun] myRank[%u] not found in subCommRanks", myRank_);
        return HCCL_E_INTERNAL;
    }
    InitStageParams(tempAlgParams, params);
    if (cclBufferCountPerRank_ == 0) {
        HCCL_ERROR(
            "[InsTempAlltoAllHierMeshStage][KernelRun] cclBufferCountPerRank_ is 0, "
            "inputSliceStride[%llu], dataTypeSize[%llu], otherDimRankSize[%u]",
            tempAlgParams.inputSliceStride, dataTypeSize_, params.otherDimRankSize);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[InsTempAlltoAllHierMeshStage][KernelRun] stageIndex[%u], myRank[%u], myAlgRank[%u], "
        "stageRankSize[%u], isFirst[%d], isLast[%d], isMiddle[%d], "
        "cclBufferCountPerRank_[%llu], srcGroupSize[%u], targetGroupSize[%u], subSlotCount[%u]",
        stageIndex_, myRank_, myAlgRank, params.stageRankSize, isFirstStage_, isLastStage_, isMiddleStage_,
        cclBufferCountPerRank_, params.srcGroupSize, params.targetGroupSize, params.subSlotCount);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllHierMeshStage::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    CHK_PRT_RET(
        param.all2AllVDataDes.sendCounts == nullptr,
        HCCL_ERROR("[InsTempAlltoAllHierMeshStage][KernelRun] sendCounts is nullptr, stageIndex[%u]", stageIndex_),
        HCCL_E_PARA);
    totalCount_ = *(reinterpret_cast<const u64*>(param.all2AllVDataDes.sendCounts));

    u32 myAlgRank = 0;
    StageParams params;
    CHK_RET(ValidateAndInit(tempAlgParams, myAlgRank, params));

    CHK_RET(PreSyncThreads(templateResource));

    if (isFirstStage_) {
        CHK_RET(LocalCopyForMyGroup(
            tempAlgParams, templateResource, myAlgRank, params.targetGroupSize, params.subSlotCount));
    }
    if (isLastStage_) {
        CHK_RET(
            LocalCopyForMyRank(tempAlgParams, templateResource, myAlgRank, params.srcGroupSize, params.subSlotCount));
    }
    CHK_RET(RunSendRecv(param, tempAlgParams, templateResource, myAlgRank, params));

    CHK_RET(PostSyncThreads(templateResource));

    HCCL_INFO("[InsTempAlltoAllHierMeshStage][KernelRun] stageIndex[%u] done", stageIndex_);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllHierMeshStage::PreSyncThreads(const TemplateResource& templateResource)
{
    u32 threadNum = static_cast<u32>(templateResource.threads.size());
    if (threadNum <= 1) {
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    notifyIdxMainToSub_.resize(subThreads.size(), 0);
    return PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_);
}

HcclResult InsTempAlltoAllHierMeshStage::PostSyncThreads(const TemplateResource& templateResource)
{
    u32 threadNum = static_cast<u32>(templateResource.threads.size());
    if (threadNum <= 1) {
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
    notifyIdxSubToMain_.clear();
    for (u32 i = 0; i < subThreads.size(); i++) {
        notifyIdxSubToMain_.push_back(i);
    }
    return PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_);
}

HcclResult InsTempAlltoAllHierMeshStage::LocalCopyForMyGroup(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource, u32 myAlgRank,
    u32 targetGroupSize, u32 subSlotCount)
{
    u64 sliceSize = tempAlgParams.sliceSize;
    if (sliceSize == 0) {
        return HCCL_SUCCESS;
    }

    for (u32 targetIdx = 0; targetIdx < targetGroupSize; targetIdx++) {
        u64 srcOffset = (myAlgRank * targetGroupSize + targetIdx) * totalCount_ * dataTypeSize_
                        + tempAlgParams.buffInfo.inBuffBaseOff;
        u64 dstOffset = targetIdx * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                        + myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;

        DataSlice srcSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, sliceSize, tempAlgParams.count);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, sliceSize, tempAlgParams.count);

        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    }

    HCCL_DEBUG(
        "[InsTempAlltoAllHierMeshStage][LocalCopyForMyGroup] myAlgRank[%u], targetGroupSize[%u], "
        "sliceSize[%llu]",
        myAlgRank, targetGroupSize, sliceSize);
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllHierMeshStage::LocalCopyForMyRank(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource, u32 myAlgRank, u32 srcGroupSize,
    u32 subSlotCount)
{
    u64 sliceSize = tempAlgParams.sliceSize;
    if (sliceSize == 0) {
        return HCCL_SUCCESS;
    }

    for (u32 srcIdx = 0; srcIdx < srcGroupSize; srcIdx++) {
        u64 srcOffset = myAlgRank * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                        + srcIdx * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;
        u64 dstOffset = (srcIdx * templateRankSize_ + myAlgRank) * totalCount_ * dataTypeSize_
                        + tempAlgParams.buffInfo.outBuffBaseOff;

        DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, sliceSize, tempAlgParams.count);
        DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, sliceSize, tempAlgParams.count);

        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    }

    HCCL_DEBUG(
        "[InsTempAlltoAllHierMeshStage][LocalCopyForMyRank] myAlgRank[%u], srcGroupSize[%u], "
        "sliceSize[%llu]",
        myAlgRank, srcGroupSize, sliceSize);
    return HCCL_SUCCESS;
}

void InsTempAlltoAllHierMeshStage::BuildSendSlices(
    const TemplateDataParams& tempAlgParams, u32 myAlgRank, u32 peerAlgRank, void* remoteCclBuffAddr,
    u32 targetGroupSize, u32 subSlotCount, std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices)
{
    u64 sliceSize = tempAlgParams.sliceSize;
    u32 loopCount = isLastStage_ ? subSlotCount : targetGroupSize;
    for (u32 i = 0; i < loopCount; i++) {
        u64 srcOffset = 0;
        u64 dstOffset = 0;
        if (isFirstStage_) {
            srcOffset = (peerAlgRank * targetGroupSize + i) * totalCount_ * dataTypeSize_
                        + tempAlgParams.buffInfo.inBuffBaseOff;
            dstOffset = i * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                        + myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;
            txSrcSlices.emplace_back(tempAlgParams.buffInfo.inputPtr, srcOffset, sliceSize, tempAlgParams.count);
        } else if (isLastStage_) {
            srcOffset = peerAlgRank * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                        + i * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;
            dstOffset = myAlgRank * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                        + i * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;
            txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, sliceSize, tempAlgParams.count);
        } else {
            srcOffset = i * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                        + myAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;
            dstOffset = i * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                        + peerAlgRank * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;
            txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, sliceSize, tempAlgParams.count);
        }
        txDstSlices.emplace_back(remoteCclBuffAddr, dstOffset, sliceSize, tempAlgParams.count);
    }
}

void InsTempAlltoAllHierMeshStage::BuildRecvSlices(
    const TemplateDataParams& tempAlgParams, u32 myAlgRank, u32 peerAlgRank, void* remoteCclBuffAddr,
    const StageParams& params, std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices)
{
    u64 sliceSize = tempAlgParams.sliceSize;
    u32 subSlotCount = params.subSlotCount;
    u32 targetGroupSize = params.targetGroupSize;
    u32 srcGroupSize = params.srcGroupSize;

    if (isFirstStage_) {
        for (u32 i = 0; i < targetGroupSize; i++) {
            u64 srcOffset = (myAlgRank * targetGroupSize + i) * totalCount_ * dataTypeSize_
                            + tempAlgParams.buffInfo.inBuffBaseOff;
            u64 dstOffset = i * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                            + peerAlgRank * cclBufferCountPerRank_ * dataTypeSize_
                            + tempAlgParams.buffInfo.hcclBuffBaseOff;
            rxSrcSlices.emplace_back(remoteCclBuffAddr, srcOffset, sliceSize, tempAlgParams.count);
            rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, sliceSize, tempAlgParams.count);
        }
    } else if (isLastStage_) {
        for (u32 i = 0; i < srcGroupSize; i++) {
            u64 srcOffset = myAlgRank * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                            + i * cclBufferCountPerRank_ * dataTypeSize_ + tempAlgParams.buffInfo.hcclBuffBaseOff;
            u64 dstOffset = (i * templateRankSize_ + peerAlgRank) * totalCount_ * dataTypeSize_
                            + tempAlgParams.buffInfo.outBuffBaseOff;
            rxSrcSlices.emplace_back(remoteCclBuffAddr, srcOffset, sliceSize, tempAlgParams.count);
            rxDstSlices.emplace_back(tempAlgParams.buffInfo.outputPtr, dstOffset, sliceSize, tempAlgParams.count);
        }
    } else {
        for (u32 i = 0; i < targetGroupSize; i++) {
            u64 srcOffset = i * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                            + peerAlgRank * cclBufferCountPerRank_ * dataTypeSize_
                            + tempAlgParams.buffInfo.hcclBuffBaseOff;
            u64 dstOffset = i * subSlotCount * cclBufferCountPerRank_ * dataTypeSize_
                            + myAlgRank * cclBufferCountPerRank_ * dataTypeSize_
                            + tempAlgParams.buffInfo.hcclBuffBaseOff;
            rxSrcSlices.emplace_back(remoteCclBuffAddr, srcOffset, sliceSize, tempAlgParams.count);
            rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, sliceSize, tempAlgParams.count);
        }
    }
}

HcclResult InsTempAlltoAllHierMeshStage::ExecuteSendRecv(
    const ChannelInfo& linkSend, const ChannelInfo& linkRecv, const ThreadHandle& thread,
    const std::vector<DataSlice>& txSrcSlices, const std::vector<DataSlice>& txDstSlices,
    const std::vector<DataSlice>& rxSrcSlices, const std::vector<DataSlice>& rxDstSlices, u32 remoteRank)
{
    DataInfo sendInfo{linkSend, {txSrcSlices, txDstSlices}};
    DataInfo recvInfo{linkRecv, {rxSrcSlices, rxDstSlices}};
    SendRecvInfo sendRecvInfo{{linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};

    if (!txSrcSlices.empty() && !rxSrcSlices.empty()) {
        if (isLastStage_) {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllHierMeshStage] SendRecvRead failed, remoteRank[%u]", remoteRank),
                HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvWrite(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllHierMeshStage] SendRecvWrite failed, remoteRank[%u]", remoteRank),
                HCCL_E_INTERNAL);
        }
    } else if (!txSrcSlices.empty()) {
        CHK_PRT_RET(
            SendWrite(sendInfo, thread),
            HCCL_ERROR("[InsTempAlltoAllHierMeshStage] SendWrite failed, remoteRank[%u]", remoteRank), HCCL_E_INTERNAL);
    } else if (!rxSrcSlices.empty()) {
        if (isLastStage_) {
            CHK_PRT_RET(
                RecvRead(recvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllHierMeshStage] RecvRead failed, remoteRank[%u]", remoteRank),
                HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                RecvWrite(recvInfo, thread),
                HCCL_ERROR("[InsTempAlltoAllHierMeshStage] RecvWrite failed, remoteRank[%u]", remoteRank),
                HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAlltoAllHierMeshStage::RunSendRecv(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource, u32 myAlgRank,
    const StageParams& params)
{
    (void)param;
    u32 threadNum = static_cast<u32>(templateResource.threads.size());

    for (u32 peerAlgRank = 0; peerAlgRank < templateRankSize_; peerAlgRank++) {
        if (peerAlgRank == myAlgRank) {
            continue;
        }

        u32 remoteRank = subCommRanks_[0][peerAlgRank];
        auto chanIt = templateResource.channels.find(remoteRank);
        if (chanIt == templateResource.channels.end() || chanIt->second.empty()) {
            HCCL_ERROR("[InsTempAlltoAllHierMeshStage] channel not found for remoteRank[%u]", remoteRank);
            return HCCL_E_INTERNAL;
        }
        const ChannelInfo& linkSend = chanIt->second[0];
        const ChannelInfo& linkRecv = chanIt->second[0];
        void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;

        HCCL_DEBUG(
            "[InsTempAlltoAllHierMeshStage][RunSendRecv] stageIndex[%u], myAlgRank[%u], peerAlgRank[%u], "
            "remoteRank[%u], isValid[%d], protocol[%u], locationType[%u], "
            "remoteCclBuffAddr[%p], remoteCclMemSize[%llu], sliceSize[%llu], count[%llu]",
            stageIndex_, myAlgRank, peerAlgRank, remoteRank, static_cast<int>(linkSend.isValid),
            static_cast<u32>(linkSend.protocol), static_cast<u32>(linkSend.locationType), remoteCclBuffAddr,
            linkSend.remoteCclMem.size, tempAlgParams.sliceSize, tempAlgParams.count);

        u32 queIdx = (peerAlgRank < myAlgRank) ? peerAlgRank : (peerAlgRank - 1);
        if (queIdx >= threadNum) {
            HCCL_WARNING(
                "[InsTempAlltoAllHierMeshStage] queIdx[%u] >= threadNum[%u], fallback to main thread, "
                "peerAlgRank[%u], myAlgRank[%u], stageIndex[%u]",
                queIdx, threadNum, peerAlgRank, myAlgRank, stageIndex_);
            queIdx = 0;
        }
        const ThreadHandle& thread = templateResource.threads[queIdx];

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        BuildSendSlices(
            tempAlgParams, myAlgRank, peerAlgRank, remoteCclBuffAddr, params.targetGroupSize, params.subSlotCount,
            txSrcSlices, txDstSlices);
        BuildRecvSlices(tempAlgParams, myAlgRank, peerAlgRank, remoteCclBuffAddr, params, rxSrcSlices, rxDstSlices);

        HCCL_DEBUG(
            "[InsTempAlltoAllHierMeshStage][RunSendRecv] after build slices, txSrcSlices[%zu], "
            "rxDstSlices[%zu], txSliceSize[%llu], rxSliceSize[%llu]",
            txSrcSlices.size(), rxDstSlices.size(), txSrcSlices.empty() ? 0 : txSrcSlices[0].size_,
            rxDstSlices.empty() ? 0 : rxDstSlices[0].size_);

        CHK_RET(ExecuteSendRecv(
            linkSend, linkRecv, thread, txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices, remoteRank));
    }

    HCCL_DEBUG("[InsTempAlltoAllHierMeshStage][RunSendRecv] done, myAlgRank[%u]", myAlgRank);
    return HCCL_SUCCESS;
}

REGISTER_A2A_STAGE_TEMPLATE(MeshStage, InsTempAlltoAllHierMeshStage);

} // namespace ops_hccl

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_omnipipe_mesh1d.h"

namespace ops_hccl {
InsTempScatterOmniPipeMesh1D::InsTempScatterOmniPipeMesh1D(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempScatterOmniPipeMesh1D::~InsTempScatterOmniPipeMesh1D() {}

void InsTempScatterOmniPipeMesh1D::SetRoot(u32 root)
{
    HCCL_DEBUG("[InsTempScatterOmniPipeMesh1D][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    root_ = root;
}

void InsTempScatterOmniPipeMesh1D::SetDoTask(bool doTask)
{
    HCCL_DEBUG("[InsTempScatterOmniPipeMesh1D][SetDoTask] myRank_ [%u], set doTask_ [%u] ", myRank_, doTask);
    doTask_.store(doTask, std::memory_order_relaxed);
}

HcclResult InsTempScatterOmniPipeMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    GetRes(resourceRequest);

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    HCCL_DEBUG("InsTempScatterOmniPipeMesh1D--CalcRes],level0Channels.size()=[%u]", level0Channels.size());
    resourceRequest.channels.push_back(level0Channels);
    HCCL_DEBUG("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

u64 InsTempScatterOmniPipeMesh1D::GetThreadNum() const { return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1; }

HcclResult InsTempScatterOmniPipeMesh1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

// 语义改为返回当前template的类型，mesh返回1，nhr返回0
u64 InsTempScatterOmniPipeMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) { return 1; }

// 这个也不用，计算scratch、对齐、loop信息封装在公共接口了
u64 InsTempScatterOmniPipeMesh1D::CalcScratchSlice(u64 dataSize) const
{
    // mesh直接乘rankSize
    u64 scratchMultiple = templateRankSize_ * dataSize;
    return scratchMultiple;
}

void InsTempScatterOmniPipeMesh1D::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempScatterOmniPipeMesh1D::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempScatterOmniPipeMesh1D::DoLocalCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    HCCL_DEBUG("[InsTempScatterOmniPipeMesh1D][DoLocalCopy] DoLocalCopy myRank_ = [%u]", myRank_);
    if (tempAlgParams.sliceSize == 0) {
        HCCL_DEBUG("Rank [%d], get slicesize zero. skip localcopy", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    void* srcAddr = tempAlgParams.buffInfo.inputPtr;
    void* dstAddr = tempAlgParams.buffInfo.outputPtr;
    auto srcSlice
        = DataSlice(srcAddr, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.sliceSize, tempAlgParams.count);
    auto dstSlice
        = DataSlice(dstAddr, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.sliceSize, tempAlgParams.count);
    HCCL_DEBUG(
        "myRank[%u], srcSlice:%s, dstSlice:%s", myRank_, srcSlice.Describe().c_str(), dstSlice.Describe().c_str());
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    if (templateRankSize_ == 1) {
        HCCL_DEBUG("templateRankSize_ ==1");
        return HcclResult::HCCL_SUCCESS;
    }
    if (!doTask_.load(std::memory_order_relaxed)) {
        HCCL_DEBUG("[InsTempScatterOmniPipeMesh1D] Rank [%d], doTask_ is false, skip KernelRun.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    threadNum_ = templateResource.threads.size();
    dataType_ = param.DataDes.dataType;
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunScatter(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_DEBUG("[%s]Run End", __func__);
    return HcclResult::HCCL_SUCCESS;
}

// root单向分发：root把每个rank对应的数据发给该rank，非root只从root接收属于自己的数据
HcclResult InsTempScatterOmniPipeMesh1D::RunScatter(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParam)
{
    u32 myAlgRank = 0;
    // 这里获取子通信域的subrank给myAlgRank
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    HCCL_DEBUG(
        "[InsTempScatterOmniPipeMesh1D][RunScatter] myRank[%u], myAlgRank[%u], root[%u], channels.size=%u "
        "templateRankSize_=%u",
        myRank_, myAlgRank, root_, channels.size(), templateRankSize_);

    auto stepSliceInfo = tempAlgParam.stepSliceInfo;
    void* localCclBuffAddr = tempAlgParam.buffInfo.hcclBuff.addr;
    void* srcPtr = tempAlgParam.stepSliceInfo.buffInfo.inputPtr;
    u64 srcBaseOff = tempAlgParam.stepSliceInfo.buffInfo.inBuffBaseOff;
    u32 rowIdx = (myRank_ - xyTotalRankSize_) / templateRankSize_;
    HCCL_DEBUG(
        "[Mesh1D][RunScatter] inOff=%lu outOff=%lu rows=%zu ", srcBaseOff, tempAlgParam.buffInfo.outBuffBaseOff,
        stepSliceInfo.stepSliceSize.size());

    if (u32(myRank_) == root_) {
        CHK_RET(RunRootScatter(channels, threads, tempAlgParam, myAlgRank));
    } else {
        CHK_RET(RunNonRootScatter(channels, threads, tempAlgParam, myAlgRank));
    }
    return HcclResult::HCCL_SUCCESS;
}

// root分支：遍历子通信域所有非己rank，按CCU映射只发该rank对应的那组piece（消除冗余发送）
HcclResult InsTempScatterOmniPipeMesh1D::RunRootScatter(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParam, u32 myAlgRank)
{
    HCCL_DEBUG("[InsTempScatterOmniPipeMesh1D][RunRootScatter] myRank[%u], myAlgRank[%u]", myRank_, myAlgRank);
    const auto& stepSliceInfo = tempAlgParam.stepSliceInfo;
    u32 rowIdx = (myRank_ - xyTotalRankSize_) / templateRankSize_;
    u64 totalPieceNum = stepSliceInfo.stepSliceSize[rowIdx].size();
    u64 peerNum = templateRankSize_ - 1;
    CHK_PRT_RET(
        peerNum == 0 || totalPieceNum % peerNum != 0,
        HCCL_ERROR(
            "[InsTempScatterOmniPipeMesh1D][RunScatter] totalPieceNum[%llu] not divisible by peerNum[%llu]",
            totalPieceNum, peerNum),
        HCCL_E_INTERNAL);
    u32 threadIdx = 0;
    u64 repeatNum = totalPieceNum / peerNum;
    u32 originIndex = 0; // 非root rank序号（跳过root），对齐CCU BuildSliceStrideVec
    for (u32 algRank = 0; algRank < templateRankSize_; algRank++) {
        if (algRank == myAlgRank) {
            continue;
        }
        CHK_RET(SendRootDataToRank(channels, threads, tempAlgParam, algRank, repeatNum, originIndex, threadIdx));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeMesh1D::SendRootDataToRank(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParam, u32 algRank, u64 repeatNum, u32& originIndex, u32& threadIdx)
{
    u32 remoteRank = subCommRanks_[0][algRank];
    HCCL_DEBUG(
        "[InsTempScatterOmniPipeMesh1D][SendRootDataToRank] remoteRank[%u], originIndex[%u]", remoteRank, originIndex);
    CHK_PRT_RET(
        channels.find(remoteRank) == channels.end() || channels.at(remoteRank).empty(),
        HCCL_ERROR("[InsTempScatterOmniPipeMesh1D][RunScatter] remoteRank[%u] not found in channels", remoteRank),
        HCCL_E_INTERNAL);
    const ChannelInfo& linkSend = channels.at(remoteRank)[0];
    const auto& stepSliceInfo = tempAlgParam.stepSliceInfo;
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    CHK_RET(BuildRootTxBatchSlices(
        stepSliceInfo, (myRank_ - xyTotalRankSize_) / templateRankSize_, remoteRank, stepSliceInfo.buffInfo.inputPtr,
        stepSliceInfo.buffInfo.inBuffBaseOff, tempAlgParam.buffInfo.outBuffBaseOff, linkSend.remoteCclMem.addr,
        repeatNum, originIndex, txSrcSlices, txDstSlices));
    originIndex++;
    if (txSrcSlices.empty()) {
        HCCL_DEBUG("[InsTempScatterOmniPipeMesh1D][RunScatter] algRank[%u] all slices empty, skip send", algRank);
        return HcclResult::HCCL_SUCCESS;
    }
    if (threadIdx >= threads.size()) {
        HCCL_ERROR("[RunScatter] threadIdx[%u] >= threads.size[%u]", threadIdx, threads.size());
        return HcclResult::HCCL_E_INTERNAL;
    }
    SlicesList txSlicesList({txSrcSlices}, {txDstSlices});
    DataInfo sendData(linkSend, txSlicesList);
    CHK_PRT_RET(
        static_cast<HcclResult>(SendBatchWrite(sendData, threads.at(threadIdx))),
        HCCL_ERROR("[InsTempScatterOmniPipeMesh1D][RunScatter] Send to rank[%u] failed", remoteRank),
        HcclResult::HCCL_E_INTERNAL);
    HCCL_DEBUG("[InsTempScatterOmniPipeMesh1D][RunScatter] Send to rank[%u] success", remoteRank);
    threadIdx++;
    return HcclResult::HCCL_SUCCESS;
}

// root分支：为某个非己rank构建tx批数据切片（for rpt循环体），按CCU映射只发该rank对应的那组piece
HcclResult InsTempScatterOmniPipeMesh1D::BuildRootTxBatchSlices(
    const StepSliceInfo& stepSliceInfo, u32 rowIdx, u32 remoteRank, void* srcPtr, u64 srcBaseOff, u64 outBuffBaseOff,
    void* remoteCclBuffAddr, u64 repeatNum, u32 originIndex, std::vector<DataSlice>& txSrcSlices,
    std::vector<DataSlice>& txDstSlices)
{
    HCCL_DEBUG(
        "[InsTempScatterOmniPipeMesh1D][BuildRootTxBatchSlices] rowIdx[%u], remoteRank[%u], repeatNum[%lu], "
        "originIndex[%u]",
        rowIdx, remoteRank, repeatNum, originIndex);
    for (u64 rpt = 0; rpt < repeatNum; rpt++) {
        u64 pieceIdx = repeatNum * originIndex + rpt;
        u64 srcOffset = srcBaseOff + stepSliceInfo.inputOmniPipeSliceStride[rowIdx][pieceIdx];
        u64 dstOffset = outBuffBaseOff + stepSliceInfo.outputOmniPipeSliceStride[rowIdx][pieceIdx];
        u64 sliceSize = stepSliceInfo.stepSliceSize[rowIdx][pieceIdx];
        u64 count = stepSliceInfo.stepCount[rowIdx][pieceIdx];
        if (sliceSize == 0) {
            continue;
        }
        txSrcSlices.emplace_back(srcPtr, srcOffset, sliceSize, count);
        txDstSlices.emplace_back(remoteCclBuffAddr, dstOffset, sliceSize, count);
        HCCL_DEBUG(
            "[Mesh1D][RunScatter] send to rank[%u], pieceIdx[%lu], srcOff[%lu], dstOff[%lu], sz[%lu]", remoteRank,
            pieceIdx, srcOffset, dstOffset, sliceSize);
    }
    return HcclResult::HCCL_SUCCESS;
}

// 非root分支：按CCU映射只收属于自己的那组piece（消除冗余接收）
HcclResult InsTempScatterOmniPipeMesh1D::RunNonRootScatter(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParam, u32 myAlgRank)
{
    HCCL_DEBUG(
        "[InsTempScatterOmniPipeMesh1D][RunNonRootScatter] myRank[%u], myAlgRank[%u] root_[%u]", myRank_, myAlgRank,
        root_);
    auto stepSliceInfo = tempAlgParam.stepSliceInfo;
    void* localCclBuffAddr = tempAlgParam.buffInfo.hcclBuff.addr;
    if (channels.count(root_) == 0 || channels.at(root_).empty()) {
        HCCL_DEBUG("[InsTempScatterOmniPipeMesh1D][RunScatter] root[%u] not in channels, skip", root_);
        return HcclResult::HCCL_SUCCESS;
    }
    const ChannelInfo& linkRecv = channels.at(root_)[0];
    void* remoteCclBuffAddr = linkRecv.remoteCclMem.addr;

    u32 rootAlgRank;
    CHK_RET(GetAlgRank(root_, subCommRanks_[0], rootAlgRank));
    u64 totalPieceNum = stepSliceInfo.stepSliceSize[(myRank_ - xyTotalRankSize_) / templateRankSize_].size();
    u64 peerNum = templateRankSize_ - 1;
    CHK_PRT_RET(
        peerNum == 0 || totalPieceNum % peerNum != 0,
        HCCL_ERROR(
            "[InsTempScatterOmniPipeMesh1D][RunScatter] totalPieceNum[%llu] not divisible by peerNum[%llu]",
            totalPieceNum, peerNum),
        HCCL_E_INTERNAL);
    u32 myOriginIndex = CalcNonRootOriginIndex(myAlgRank, rootAlgRank);
    u64 repeatNum = totalPieceNum / peerNum;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    CHK_RET(BuildNonRootRxBatchSlices(
        stepSliceInfo, (myRank_ - xyTotalRankSize_) / templateRankSize_, root_, localCclBuffAddr, remoteCclBuffAddr,
        tempAlgParam.buffInfo.inBuffBaseOff, tempAlgParam.buffInfo.outBuffBaseOff, repeatNum, myOriginIndex,
        rxSrcSlices, rxDstSlices));
    // 无有效数据，跳过接收（与root端对称，不会死锁）
    if (rxSrcSlices.empty()) {
        HCCL_DEBUG("[InsTempScatterOmniPipeMesh1D][RunScatter] myRank[%u] all slices empty, skip recv", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    SlicesList rxSlicesList({rxSrcSlices}, {rxDstSlices});
    DataInfo recvData(linkRecv, rxSlicesList);
    CHK_PRT_RET(
        static_cast<HcclResult>(RecvWrite(recvData, threads.at(0))),
        HCCL_ERROR("[InsTempScatterOmniPipeMesh1D][RunScatter] Recv from root[%u] failed", root_),
        HcclResult::HCCL_E_INTERNAL);
    HCCL_DEBUG("[InsTempScatterOmniPipeMesh1D][RunScatter] Recv from root[%u] success", root_);
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempScatterOmniPipeMesh1D::CalcNonRootOriginIndex(u32 myAlgRank, u32 rootAlgRank) const
{
    u32 originIndex = 0;
    for (u32 rank = 0; rank < myAlgRank; rank++) {
        if (rank != rootAlgRank) {
            originIndex++;
        }
    }
    return originIndex;
}

// 非root分支：构建rx批数据切片（for rpt循环体），与root发送端对称的piece索引
HcclResult InsTempScatterOmniPipeMesh1D::BuildNonRootRxBatchSlices(
    const StepSliceInfo& stepSliceInfo, u32 rowIdx, u32 rootRank, void* localCclBuffAddr, void* remoteCclBuffAddr,
    u64 inBuffBaseOff, u64 outBuffBaseOff, u64 repeatNum, u32 myOriginIndex, std::vector<DataSlice>& rxSrcSlices,
    std::vector<DataSlice>& rxDstSlices)
{
    HCCL_DEBUG(
        "[InsTempScatterOmniPipeMesh1D][BuildNonRootRxBatchSlices] rowIdx[%u], rootRank[%u], repeatNum[%lu], "
        "myOriginIndex[%u]",
        rowIdx, rootRank, repeatNum, myOriginIndex);
    for (u64 rpt = 0; rpt < repeatNum; rpt++) {
        u64 pieceIdx = repeatNum * myOriginIndex + rpt; // 与root发送端对称的piece索引
        u64 srcOffset = inBuffBaseOff + stepSliceInfo.inputOmniPipeSliceStride[rowIdx][pieceIdx];
        u64 dstOffset = outBuffBaseOff + stepSliceInfo.outputOmniPipeSliceStride[rowIdx][pieceIdx];
        u64 sliceSize = stepSliceInfo.stepSliceSize[rowIdx][pieceIdx];
        u64 count = stepSliceInfo.stepCount[rowIdx][pieceIdx];
        if (sliceSize == 0) {
            continue;
        }
        rxSrcSlices.emplace_back(remoteCclBuffAddr, srcOffset, sliceSize, count);
        rxDstSlices.emplace_back(localCclBuffAddr, dstOffset, sliceSize, count);
        HCCL_DEBUG(
            "[Mesh1D][RunScatter] recv from rank[%u], pieceIdx[%lu], srcOff[%lu], dstOff[%lu], sz[%lu]", rootRank,
            pieceIdx, srcOffset, dstOffset, sliceSize);
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl

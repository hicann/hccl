/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_1D_intra.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {
InsTempAllGatherMesh1dIntra::InsTempAllGatherMesh1dIntra(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}
InsTempAllGatherMesh1dIntra::~InsTempAllGatherMesh1dIntra() {}

HcclResult InsTempAllGatherMesh1dIntra::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1dIntra::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1; // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1dIntra::GetThreadNum() const { return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1; }

u64 InsTempAllGatherMesh1dIntra::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempAllGatherMesh1dIntra::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherMesh1dIntra] Run start");
    threadNum_ = templateResource.threads.size();
    HCCL_INFO("[InsTempAllGatherMesh1dIntra][KernelRun] threadNum:%u", threadNum_);
    tempAlgParams_ = tempAlgParams;

    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;

    HCCL_DEBUG("[InsTempAllGatherMesh1dIntra] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);
    CHK_RET(LocalDataCopy(templateResource.threads));
    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(RunAllGatherMesh(templateResource.threads, templateResource.channels));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    CHK_RET(PostLocalCopy(templateResource.threads));
    HCCL_INFO("[InsTempAllGatherMesh1dIntra] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1dIntra::RunAllGatherMesh(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1dIntra] RunAllGatherMesh RankIDs[%d].", myRank_);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    HCCL_INFO("[InsTempAllGatherMesh1dIntra] RunAllGatherMesh RankIDs[%d], myAlgRank[%d].", myRank_, myAlgRank);

    u64 myAlgSize = tempAlgParams_.allRankSliceSize.at(myAlgRank);
    u64 myAlgCount = tempAlgParams_.allRankProcessedDataCount.at(myAlgRank);
    u64 myAlgOffset = tempAlgParams_.allRankDispls.at(myAlgRank);

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
            u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];

            u32 connectedAlgRank = 0;
            CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
            HCCL_INFO(
                "[InsTempAllGatherMesh1dIntra] RunAllGatherMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
                myRank_, connectedRank, connectedAlgRank);

            // 异常检查
            CHK_PRT_RET(
                threadIdx >= threads.size() || !channels.count(connectedRank),
                HCCL_ERROR(
                    "[InsTempAllGatherMesh1dIntra][RankID]=%u threadIdx=%u, threads.size=%u, "
                    "connectedRank=%d, channels.size=%u",
                    myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                HcclResult::HCCL_E_INTERNAL);

            u64 connectedAlgSize = tempAlgParams_.allRankSliceSize.at(connectedAlgRank);
            u64 connectedAlgCount = tempAlgParams_.allRankProcessedDataCount.at(connectedAlgRank);
            u64 connectedAlgOffset = tempAlgParams_.allRankDispls.at(connectedAlgRank);

            // 既不发送也不接受
            if (myAlgSize == 0 && connectedAlgSize == 0) {
                continue;
            }

            ThreadHandle currQue = threads[threadIdx];
            // allgather 预留兼容offload模式
            const ChannelInfo& linkRemote = channels.at(connectedRank)[0];
            void* remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

            u64 txOutOffset = outBaseOff + myAlgOffset;
            u64 txScratchOffset = scratchBase + myAlgOffset;
            u64 txDstOffset = txScratchOffset;

            u64 rxOutOffset = connectedAlgOffset + outBaseOff;
            u64 rxScratchOffset = scratchBase + connectedAlgOffset;
            u64 rxDstOffset = rxScratchOffset;

            void* txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
            void* txDstPtr = remoteCclBuffAddr;
            void* rxSrcPtr = tempAlgParams_.buffInfo.outputPtr;
            void* rxDstPtr = remoteCclBuffAddr;
            // allgather write模式使用tx,rx地址不生效，仅使用对端link做Post/Wait
            // allgather read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
            std::vector<DataSlice> txSrcSlices{DataSlice(txSrcPtr, txOutOffset, myAlgSize, myAlgCount)}; // 本地(send)
            std::vector<DataSlice> txDstSlices{DataSlice(txDstPtr, txDstOffset, myAlgSize, myAlgCount)}; // 远程(send)
            // allgather read模式使用rx
            std::vector<DataSlice> rxDstSlices{
                DataSlice(rxDstPtr, rxDstOffset, connectedAlgSize, connectedAlgCount)}; // 本地(recv)
            std::vector<DataSlice> rxSrcSlices{
                DataSlice(rxSrcPtr, rxOutOffset, connectedAlgSize, connectedAlgCount)}; // 远程(recv)

            if (myAlgSize == 0) { // 发送数据片为0时，只接收数据
                SlicesList recvSliceList(rxSrcSlices, rxDstSlices);
                DataInfo recvInfo(linkRemote, recvSliceList);
                CHK_PRT_RET(
                    RecvWrite(recvInfo, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherMesh1dIntra][RunAllGatherMesh] Recv failed."),
                    HcclResult::HCCL_E_INTERNAL);
            } else if (connectedAlgSize == 0) { // 接收数据片为0时，只发送数据
                SlicesList sendSliceList(txSrcSlices, txDstSlices);
                DataInfo sendInfo(linkRemote, sendSliceList);
                CHK_PRT_RET(
                    SendWrite(sendInfo, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherMesh1dIntra][RunAllGatherMesh] Send failed."),
                    HcclResult::HCCL_E_INTERNAL);
            } else {
                SendRecvInfo sendRecvInfo{
                    {linkRemote, linkRemote}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
                CHK_PRT_RET(
                    SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherMesh1dIntra][RunAllGatherMesh] Send failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1dIntra::LocalDataCopy(const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] LocalDataCopy.");
    if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr) {
        return HcclResult::HCCL_SUCCESS;
    }
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = tempAlgParams_.allRankSliceSize.at(myAlgRank);
    u64 sliceCount = tempAlgParams_.allRankProcessedDataCount.at(myAlgRank);
    u64 sliceOffset = tempAlgParams_.allRankDispls.at(myAlgRank);

    if (sliceSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        // repeat 造成的偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        // 数据块rank编号造成的偏移
        const u64 inOff = inBaseOff;
        const u64 outOff = sliceOffset + outBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);

        HCCL_DEBUG(
            "[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] srcSlice: inBaseOff[%d] inOff[%d] "
            "sliceSize[%d] count[%d].",
            myRank_, myAlgRank, inBaseOff, inOff, sliceSize, sliceCount);
        HCCL_DEBUG(
            "[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] dstSlice: outBaseoff[%d] "
            "outOff[%d] sliceSize[%d] count[%d].",
            myRank_, myAlgRank, outBaseOff, outOff, sliceSize, sliceCount);

        LocalCopy(threads[0], srcSlice, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1dIntra::PostLocalCopy(const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1dIntra] PostLocalCopy.");

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            if (rank == myRank_) {
                continue;
            }
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));

            u64 sliceSize = tempAlgParams_.allRankSliceSize.at(algRank);
            u64 sliceCount = tempAlgParams_.allRankProcessedDataCount.at(algRank);
            u64 sliceOffset = tempAlgParams_.allRankDispls.at(algRank);

            if (sliceSize == 0) {
                continue;
            }

            u64 scratchOffset = sliceOffset + scratchBase;
            u64 outOffset = sliceOffset + outBaseOff;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, sliceSize, sliceCount);
            HCCL_DEBUG(
                "[InsTempAllGatherMesh1D][PostLocalCopy] LocalDataCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                "scratchBase[%d] outBaseOff[%d] scratchOffset[%d] outOffset[%d].",
                myRank_, rank, algRank, scratchBase, outBaseOff, scratchOffset, outOffset);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherMesh1dIntra::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = threadNum_;
    u32 slaveThreadNum = threadNum - 1;
    HCCL_INFO(
        "[InsTempAllGatherMesh1dIntra][GetNotifyIdxMainToSub]threadNum: %u, slaveThreadNum: %u", threadNum,
        slaveThreadNum);
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherMesh1dIntra::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = threadNum_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl

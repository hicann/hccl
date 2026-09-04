/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_mesh_1D_intra.h"

namespace ops_hccl {

InsTempScatterMesh1DIntra::InsTempScatterMesh1DIntra(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempScatterMesh1DIntra::~InsTempScatterMesh1DIntra() {}

void InsTempScatterMesh1DIntra::SetRoot(u32 root) { root_ = root; }

u64 InsTempScatterMesh1DIntra::GetThreadNum() const
{
    u64 threadNum = (subCommRanks_[0].size() > 1) ? (subCommRanks_[0].size() - 1) : 1;
    return threadNum;
}

HcclResult InsTempScatterMesh1DIntra::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // mesh 算法只做level 0 层级的
    u64 threadNum = (subCommRanks_[0].size() > 1) ? (subCommRanks_[0].size() - 1) : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    // mesh算法只做level 0的，因此这里算的channels也是level 0的
    // 多级的时候需要分别在template中计算，然后在executor中将channels组合
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

u64 InsTempScatterMesh1DIntra::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}

HcclResult InsTempScatterMesh1DIntra::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    count_ = tempAlgParams.count;
    HCCL_INFO("[InsTempScatterMesh1DIntra] Run Start");

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

    HCCL_INFO("[InsTempScatterMesh1DIntra] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1DIntra::LocalCopyforMyRank(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
    const std::vector<ThreadHandle>& threads)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    u64 sliceSize = tempAlgParams.allRankSliceSize.at(myAlgRank);
    u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(myAlgRank);
    u64 sliceOffset = tempAlgParams.allRankDispls.at(myAlgRank);

    // 将myRank的切片直接从inputPtr拷贝到outputPtr上
    DataSlice srcSlice = DataSlice(
        tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff + sliceOffset, sliceSize, sliceCount);

    DataSlice dstSlice = DataSlice(
        tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff + sliceOffset, sliceSize, sliceCount);
    if (sliceCount != 0) {
        HCCL_DEBUG(
            "[InsTempScatterMesh1DIntra][LocalCopyforMyRank] RankID [%d] dataAlgRank[%d] "
            "srcOff[%d] dstOff[%d] sliceOffset[%d] sliceSize[%d].",
            myRank_, myAlgRank, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff,
            sliceOffset, sliceSize);

        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempScatterMesh1DIntra::PostCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    // 通信结束之后，非root rank数据都在 cclBuffer 上，需要搬运到对应的输出位置。
    if (u32(myRank_) == root_ || tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        return HCCL_SUCCESS;
    }

    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempScatterMesh1DIntra][PostCopy] subCommRanks_[0] or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    // 如果是单算子模式, 并且是最后一步算子，需要将数据从 cclBuffer 拷贝到 userOut
    HCCL_INFO("[InsTempScatterMesh1DIntra][PostCopy], copy from cclBuffer to userOut");

    u64 sliceSize = tempAlgParams.allRankSliceSize.at(myAlgRank);
    u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(myAlgRank);
    u64 sliceOffset = tempAlgParams.allRankDispls.at(myAlgRank);

    DataSlice srcSlice = DataSlice(
        tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff + sliceOffset,
        sliceSize * tempAlgParams.repeatNum, sliceCount * tempAlgParams.repeatNum);
    DataSlice dstSlice = DataSlice(
        tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff + sliceOffset,
        sliceSize * tempAlgParams.repeatNum, sliceCount * tempAlgParams.repeatNum);
    HCCL_DEBUG(
        "[InsTempScatterMesh1DIntra][LocalCopy] LocalDataCopy RankID [%d] dataAlgRank[%d] "
        "srcOff[%d] dstOff[%d] sliceOffset[%d] sliceSize[%d].",
        myRank_, myAlgRank, tempAlgParams.buffInfo.hcclBuffBaseOff + sliceOffset,
        tempAlgParams.buffInfo.outBuffBaseOff + sliceOffset, sliceOffset, sliceSize);
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1DIntra::RunScatter(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParams)
{
    u32 myAlgRank = 0;
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter != subCommRanks_[0].end()) {
        myAlgRank = std::distance(subCommRanks_[0].begin(), iter);
    } else {
        HCCL_ERROR("[InsTempScatterMesh1DIntra][RunScatter] subCommRanks_[0] or myRank_ is error.");
        return HCCL_E_INTERNAL;
    }

    if (root_ == u32(myRank_)) { // 本卡也需要搬运一部分数据
        LocalCopyforMyRank(subCommRanks_[0], tempAlgParams, threads);
    }

    u64 recvSize = tempAlgParams.allRankSliceSize.at(myAlgRank);
    u64 recvCount = tempAlgParams.allRankProcessedDataCount.at(myAlgRank);
    u64 recvOffset = tempAlgParams.allRankDispls.at(myAlgRank);

    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        if (root_ == u32(myRank_)) {
            u32 count = 0;
            for (u32 algRank = 0; algRank < subCommRanks_[0].size(); algRank++) {
                if (myAlgRank == algRank) { // 本卡的数据搬运已经完成
                    continue;
                }
                u32 remoteRank = subCommRanks_[0].at(algRank);
                HCCL_DEBUG("[InsTempScatterMesh1DIntra][RunScatter] myRank[%d], toRank[%d]", myRank_, remoteRank);
                const ChannelInfo& linkSend = channels.at(remoteRank)[0];
                void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;

                u64 sendSize = tempAlgParams.allRankSliceSize.at(algRank); // 向对端发送的参数
                u64 sendCount = tempAlgParams.allRankProcessedDataCount.at(algRank);
                u64 sendOffset = tempAlgParams.allRankDispls.at(algRank);

                u64 srcOffset = tempAlgParams.buffInfo.inBuffBaseOff + r * tempAlgParams.inputRepeatStride + sendOffset;
                u64 dstOffset
                    = tempAlgParams.buffInfo.hcclBuffBaseOff + r * tempAlgParams.outputRepeatStride + sendOffset;

                DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, sendSize, sendCount);
                DataSlice dstSlice = DataSlice(remoteCclBuffAddr, dstOffset, sendSize, sendCount);

                if (sendSize != 0) {
                    SlicesList txSlicesList({srcSlice}, {dstSlice});
                    DataInfo sendData(linkSend, txSlicesList);
                    HCCL_DEBUG("[InsTempScatterMesh1DIntra][RunScatter] start SendWrite");
                    CHK_PRT_RET(
                        static_cast<HcclResult>(SendWrite(sendData, threads.at(count))),
                        HCCL_ERROR("[InsTempScatterMesh1DIntra][RunScatter] Send failed"), HcclResult::HCCL_E_INTERNAL);
                }
                count++;
            }
        } else {
            if (channels.size() == 0 || channels.count(root_) == 0) {
                continue;
            }
            const ChannelInfo& linkRecv = channels.at(root_)[0];
            u64 srcOffset = tempAlgParams.buffInfo.inBuffBaseOff + r * tempAlgParams.inputRepeatStride + recvOffset;
            u64 dstOffset = tempAlgParams.buffInfo.hcclBuffBaseOff + r * tempAlgParams.outputRepeatStride + recvOffset;
            if (recvSize != 0) {
                DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, recvSize, recvCount);
                DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, recvSize, recvCount);
                SlicesList rxSlicesList({srcSlice}, {dstSlice});
                DataInfo recvData(linkRecv, rxSlicesList);
                CHK_PRT_RET(
                    static_cast<HcclResult>(RecvWrite(recvData, threads.at(0))),
                    HCCL_ERROR("[InsTempScatterMesh1DIntra][RunScatter] Recv failed"), HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempScatterMesh1DIntra::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = threadNum_;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempScatterMesh1DIntra::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = threadNum_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl

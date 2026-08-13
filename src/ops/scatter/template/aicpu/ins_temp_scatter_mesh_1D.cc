/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_mesh_1D.h"

namespace ops_hccl {
InsTempScatterMesh1D::InsTempScatterMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempScatterMesh1D::~InsTempScatterMesh1D() {}

void InsTempScatterMesh1D::SetRoot(u32 root)
{
    HCCL_INFO("[InsTempScatterMesh1D][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root);
    root_ = root;
}

u64 InsTempScatterMesh1D::GetThreadNum() const
{
    u64 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    return threadNum;
}

void InsTempScatterMesh1D::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempScatterMesh1D::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempScatterMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // mesh 算法只做level 0 层级的
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    // mesh算法只做level 0的，因此这里算的channels也是level 0的
    // 多级的时候需要分别在template中计算，然后在exector中将channels组合
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) { return 1; }

HcclResult InsTempScatterMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    for (const auto& item : templateResource.channels) {
        u32 key = item.first;
        HCCL_DEBUG("[KernelRun] myRank_ = %u, channel key = %u", myRank_, key);
    }
    CHK_PRT_RET(
        templateResource.threads.empty(), HCCL_ERROR("[InsTempScatterMesh1D][KernelRun] threads is empty"),
        HCCL_E_INTERNAL);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    CHK_PTR_NULL(tempAlgParams.buffInfo.hcclBuff.addr);
    if (u32(myRank_) == root_ && tempAlgParams.buffInfo.inBuffType != BufferType::HCCL_BUFFER) {
        CHK_PTR_NULL(tempAlgParams.buffInfo.inputPtr);
    }
    CHK_PTR_NULL(tempAlgParams.buffInfo.outputPtr);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    // 尾块模式
    if (tempAlgParams.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
        processSize_ = tempAlgParams.tailSize;
        count_ = processSize_ / dataTypeSize;
    }
    HCCL_INFO("[InsTempScatterMesh1D] Run Start");
    CHK_RET(PreCopy(tempAlgParams, templateResource.threads));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunMesh(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));
    HCCL_INFO("[InsTempScatterMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempScatterMesh1D::PreCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    // 非根节点 或 in/outBuffType均为CCL BUFFER 则跳过前拷贝
    if (u32(myRank_) != root_
        || (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER
            && tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER)) {
        HCCL_INFO("[InsTempScatterMesh1D][PreCopy] skip precopy, myRank = %u, root = %u", myRank_, root_);
        return HCCL_SUCCESS;
    }

    // inplace场景跳过前拷贝
    if (tempAlgParams.buffInfo.inputPtr == tempAlgParams.buffInfo.outputPtr) {
        HCCL_INFO(
            "[InsTempScatterMesh1D][PreCopy] skip precopy due to inplace, myRank = %u, root = %u", myRank_, root_);
        return HCCL_SUCCESS;
    }

    u32 myAlgRank = 0;
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);

    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        u64 srcOffset = tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER ?
                            r * tempAlgParams.inputRepeatStride + tempAlgParams.inputSliceStride * myAlgRank
                                + tempAlgParams.buffInfo.hcclBuffBaseOff :
                            r * tempAlgParams.inputRepeatStride + tempAlgParams.inputSliceStride * myAlgRank
                                + tempAlgParams.buffInfo.inBuffBaseOff;
        u64 dstOffset = tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER ?
                            r * tempAlgParams.outputRepeatStride + tempAlgParams.outputSliceStride * myAlgRank
                                + tempAlgParams.buffInfo.hcclBuffBaseOff :
                            r * tempAlgParams.outputRepeatStride + tempAlgParams.outputSliceStride * myAlgRank
                                + tempAlgParams.buffInfo.outBuffBaseOff;
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, processSize_, count_);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, processSize_, count_);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempScatterMesh1D::PostCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    bool isInplaceRootRank
        = u32(myRank_) == root_ && tempAlgParams.buffInfo.inputPtr == tempAlgParams.buffInfo.outputPtr;
    // 通信结束之后，非root rank数据都在 cclBuffer 上，需要搬运到对应的输出位置。
    if (tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER || enableRemoteMemAccess_) {
        HCCL_INFO("[InsTempScatterMesh1D][PostCopy] skip postcopy, myRank = %u, root = %u", myRank_, root_);
        return HCCL_SUCCESS;
    }

    // root rank数据在非inplace场景不需要进行搬运
    if (u32(myRank_) == root_ && tempAlgParams.buffInfo.inputPtr != tempAlgParams.buffInfo.outputPtr) {
        HCCL_INFO("[InsTempScatterMesh1D][PostCopy] root rank skip postcopy, myRank = %u, root = %u", myRank_, root_);
        return HCCL_SUCCESS;
    }

    u32 myAlgRank = 0;
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);
    // root rank数据在inplace场景下需要进行搬运
    if (isInplaceRootRank) {
        HCCL_INFO("[InsTempScatterMesh1D][PostCopy], is inplace root rank, copy from userIn to userOut");
    } else { // 如果是单算子模式, 并且是最后一步算子，需要将数据从 cclBuffer 拷贝到 userOut
        HCCL_INFO("[InsTempScatterMesh1D][PostCopy], copy from cclBuffer to userOut");
    }

    void* srcPtr = isInplaceRootRank ? tempAlgParams.buffInfo.inputPtr : tempAlgParams.buffInfo.hcclBuff.addr;
    u64 srcOffset = isInplaceRootRank ?
                        tempAlgParams.buffInfo.inBuffBaseOff + tempAlgParams.inputSliceStride * myAlgRank :
                        tempAlgParams.buffInfo.hcclBuffBaseOff + tempAlgParams.outputSliceStride * myAlgRank;
    DataSlice srcSlice
        = DataSlice(srcPtr, srcOffset, processSize_ * tempAlgParams.repeatNum, count_ * tempAlgParams.repeatNum);
    DataSlice dstSlice = DataSlice(
        tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.buffInfo.outBuffBaseOff + tempAlgParams.outputSliceStride * myAlgRank,
        processSize_ * tempAlgParams.repeatNum, count_ * tempAlgParams.repeatNum);
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1D::RunMesh(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParams)
{
    u32 myAlgRank = 0;
    u32 curSliceSize = 0;
    u32 curCount = 0;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);
    HCCL_DEBUG(
        "[InsTempScatterMesh1D][RunMesh] myRank[%d], myAlgRank[%d], channels size[%d]", myRank_, myAlgRank,
        channels.size());
    if (root_ == u32(myRank_)) {
        u32 count = 0; // 用于标记当前使用的线程
        for (u32 algRank = 0; algRank < subCommRanks_[0].size(); algRank++) {
            curSliceSize = tempAlgParams.tailSize != 0 && algRank == templateRankSize_ - 1 ? tempAlgParams.tailSize :
                                                                                             processSize_;
            curCount = curSliceSize / dataTypeSize;
            if (myAlgRank == algRank) {
                continue;
            }
            HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] algRank[%d]", algRank);
            u32 remoteRank = subCommRanks_[0][algRank];
            HCCL_INFO("[InsTempScatterMesh1D][RunMesh] myRank[%d], toRank[%d]", myRank_, remoteRank);
            CHK_PRT_RET(
                channels.find(remoteRank) == channels.end() || channels.at(remoteRank).empty(),
                HCCL_ERROR("[InsTempScatterMesh1D][RunMesh] remoteRank[%d] not found in channels", remoteRank),
                HCCL_E_INTERNAL);
            const std::vector<ChannelInfo>& linkSends = channels.at(remoteRank);
            // 需要根据channel切分数据
            CHK_RET(CalcDataSplitByPortGroup(
                curCount, DATATYPE_SIZE_TABLE[dataType_], linkSends, elemCountOut_, sizeOut_, elemOffset_));
            // 循环每个channel发送数据
            for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
                const ChannelInfo& linkSend = linkSends[channelIdx];
                curSliceSize = sizeOut_[channelIdx];
                curCount = elemCountOut_[channelIdx];
                std::vector<DataSlice> txSrcSlices;
                std::vector<DataSlice> txDstSlices;
                for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
                    u64 srcOffset = tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER ?
                                        tempAlgParams.buffInfo.hcclBuffBaseOff + r * tempAlgParams.inputRepeatStride
                                            + algRank * tempAlgParams.inputSliceStride + elemOffset_[channelIdx] :
                                        r * tempAlgParams.inputRepeatStride + algRank * tempAlgParams.inputSliceStride
                                            + tempAlgParams.buffInfo.inBuffBaseOff + elemOffset_[channelIdx];
                    u64 dstOffset
                        = (!enableRemoteMemAccess_) ?
                              tempAlgParams.buffInfo.hcclBuffBaseOff + algRank * tempAlgParams.outputSliceStride
                                  + r * tempAlgParams.outputRepeatStride + elemOffset_[channelIdx] :
                              tempAlgParams.buffInfo.outBuffBaseOff + algRank * tempAlgParams.outputSliceStride
                                  + r * tempAlgParams.outputRepeatStride + elemOffset_[channelIdx];
                    void* txDstPtr
                        = (!enableRemoteMemAccess_) ? linkSend.remoteCclMem.addr : linkSend.remoteOutputGraphMode.addr;
                    HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] txDstPtr[%d]", txDstPtr);
                    HCCL_DEBUG(
                        "[InsTempScatterMesh1D][RunMesh] srcOffset[%d], tempAlgParams.buffInfo.inputPtr[%d]", srcOffset,
                        tempAlgParams.buffInfo.inputPtr);
                    HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] dstOffset[%d], txDstPtr[%d]", dstOffset, txDstPtr);
                    DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, curSliceSize, curCount);
                    HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] got srcSlice");
                    DataSlice dstSlice = DataSlice(txDstPtr, dstOffset, curSliceSize, curCount);
                    HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] got dstSlice");
                    txSrcSlices.push_back(srcSlice);
                    txDstSlices.push_back(dstSlice);
                }
                SlicesList txSlicesList({txSrcSlices}, {txDstSlices});
                HCCL_DEBUG(
                    "[InsTempScatterMesh1D][RunMesh] tempAlgParam.buffInfo.hcclBuff.addr[%d], "
                    "tempAlgParams.buffInfo.inputPtr[%d], tempAlgParams.buffInfo.outputPtr[%d], ",
                    tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.inputPtr,
                    tempAlgParams.buffInfo.outputPtr);
                HCCL_DEBUG(
                    "[InsTempScatterMesh1D][RunMesh] tempAlgParam.buffInfo.inBuffBaseOff[%d], "
                    "tempAlgParam.buffInfo.outBuffBaseOff[%d], tempAlgParam.buffInfo.hcclBuffBaseOff[%d]",
                    tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff,
                    tempAlgParams.buffInfo.hcclBuffBaseOff);

                DataInfo sendData(linkSend, txSlicesList);
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] start SendWrite");
                CHK_PRT_RET(
                    count >= threads.size(),
                    HCCL_ERROR(
                        "[InsTempScatterMesh1D][RunMesh] count[%d] >= threads.size()[%d]", count, threads.size()),
                    HCCL_E_INTERNAL);
                CHK_PRT_RET(
                    static_cast<HcclResult>(SendBatchWrite(sendData, threads.at(count))),
                    HCCL_ERROR("[InsTempScatterMesh1D] RunMesh Send failed"), HcclResult::HCCL_E_INTERNAL);
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] end SendWrite");
                count++;
            }
        }
    } else {
        if (channels.size() == 0 || channels.count(root_) == 0) {
            return HcclResult::HCCL_SUCCESS;
        }
        u32 count = 0; // 用于标记当前使用的线程
        CHK_PRT_RET(
            channels.find(root_) == channels.end() || channels.at(root_).empty(),
            HCCL_ERROR("[InsTempScatterMesh1D][RunMesh] root[%d] not found in channels", root_), HCCL_E_INTERNAL);
        const std::vector<ChannelInfo>& linkRecvs = channels.at(root_);
        curSliceSize
            = tempAlgParams.tailSize != 0 && myAlgRank == templateRankSize_ - 1 ? tempAlgParams.tailSize : processSize_;
        curCount = curSliceSize / dataTypeSize;
        // 需要根据channel切分数据
        CHK_RET(CalcDataSplitByPortGroup(
            curCount, DATATYPE_SIZE_TABLE[dataType_], linkRecvs, elemCountOut_, sizeOut_, elemOffset_));
        // 循环每个channel接收数据
        for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
            const ChannelInfo& linkRecv = linkRecvs[channelIdx];
            curSliceSize = sizeOut_[channelIdx];
            curCount = elemCountOut_[channelIdx];
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
                u64 srcOffset = tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER ?
                                    tempAlgParams.buffInfo.hcclBuffBaseOff + r * tempAlgParams.inputRepeatStride
                                        + myAlgRank * tempAlgParams.inputSliceStride + elemOffset_[channelIdx] :
                                    r * tempAlgParams.inputRepeatStride + myAlgRank * tempAlgParams.inputSliceStride
                                        + tempAlgParams.buffInfo.inBuffBaseOff + elemOffset_[channelIdx];
                u64 dstOffset = (!enableRemoteMemAccess_) ?
                                    tempAlgParams.buffInfo.hcclBuffBaseOff + myAlgRank * tempAlgParams.outputSliceStride
                                        + r * tempAlgParams.outputRepeatStride + elemOffset_[channelIdx] :
                                    tempAlgParams.buffInfo.outBuffBaseOff + myAlgRank * tempAlgParams.outputSliceStride
                                        + r * tempAlgParams.outputRepeatStride + elemOffset_[channelIdx];
                // write模式使用tx, rx地址不生效，仅使用对端link做Post/Wait
                void* rxDstPtr = (!enableRemoteMemAccess_) ? tempAlgParams.buffInfo.hcclBuff.addr :
                                                             tempAlgParams.buffInfo.outputPtr;
                DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, curSliceSize, curCount);
                DataSlice dstSlice = DataSlice(rxDstPtr, dstOffset, curSliceSize, curCount);
                rxSrcSlices.push_back(srcSlice);
                rxDstSlices.push_back(dstSlice);
            }
            SlicesList rxSlicesList({rxSrcSlices}, {rxDstSlices});
            DataInfo recvData(linkRecv, rxSlicesList);
            CHK_PRT_RET(
                static_cast<HcclResult>(RecvWrite(recvData, threads.at(count))),
                HCCL_ERROR("[InsTempScatterMesh1D] RunMesh Recv failed"), HcclResult::HCCL_E_INTERNAL);
            count++;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl

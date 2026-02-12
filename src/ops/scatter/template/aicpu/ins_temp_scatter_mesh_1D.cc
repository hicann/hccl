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
InsTempScatterMesh1D::InsTempScatterMesh1D(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                const std::vector<std::vector<u32>> &subCommRanks)
                                : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempScatterMesh1D::~InsTempScatterMesh1D()
{}

void InsTempScatterMesh1D::SetRoot(u32 root)
{
    HCCL_INFO("[InsTempScatterMesh1D][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root);
    root_ = root;
}

u64 InsTempScatterMesh1D::GetThreadNum()
{
    u64 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    return threadNum;
}

void InsTempScatterMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempScatterMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempScatterMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                        AlgResourceRequest& resourceRequest)
{
    // mesh 算法只做level 0 层级的
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
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

u64 InsTempScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return 1;
}

HcclResult InsTempScatterMesh1D::KernelRun(const OpParam& param, const TemplateDataParams &tempAlgParams,
                     const TemplateResource& templateResource)
{
    for (const auto& item : templateResource.channels) {
        u32 key = item.first;
        HCCL_DEBUG("[KernelRun] myRank_ = %u, channel key = %u", myRank_, key);
    }
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
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

HcclResult InsTempScatterMesh1D::PreCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    if (u32(myRank_) != root_) {
        return HCCL_SUCCESS;
    }

    u32 myAlgRank = 0;
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);

    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        u64 srcOffset = tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER
                            ? r * tempAlgParams.inputRepeatStride + tempAlgParams.inputSliceStride * myAlgRank +
                                  tempAlgParams.buffInfo.hcclBuffBaseOff
                            : r * tempAlgParams.inputRepeatStride + tempAlgParams.inputSliceStride * myAlgRank +
                                  tempAlgParams.buffInfo.inBuffBaseOff;
        u64 dstOffset = tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER
                            ? r * tempAlgParams.outputRepeatStride + tempAlgParams.buffInfo.hcclBuffBaseOff
                            : r * tempAlgParams.outputRepeatStride + tempAlgParams.buffInfo.outBuffBaseOff;
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, processSize_, count_);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, processSize_, count_);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1D::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    // 通信结束之后，非root rank数据都在 cclBuffer 上，需要搬运到对应的输出位置。
    if (u32(myRank_) == root_ || tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        return HCCL_SUCCESS;
    }

    // 如果是单算子模式, 并且是最后一步算子，需要将数据从 cclBuffer 拷贝到 userOut
    HCCL_INFO("[InsTempScatterMesh1D][PostCopy], copy from cclBuffer to userOut");
    DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
        tempAlgParams.buffInfo.hcclBuffBaseOff,
        processSize_ * tempAlgParams.repeatNum,
        count_ * tempAlgParams.repeatNum);
    DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.buffInfo.outBuffBaseOff,
        processSize_ * tempAlgParams.repeatNum,
        count_ * tempAlgParams.repeatNum);
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterMesh1D::RunMesh(const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads, 
                    const TemplateDataParams &tempAlgParams)
{
    u32 myAlgRank = 0;
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);
    HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] myRank[%d], myAlgRank[%d], channels size[%d]", myRank_, myAlgRank, channels.size());
    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        if (root_ == u32(myRank_)) {
            u32 count = 0;
            for (u32 algRank = 0; algRank < subCommRanks_[0].size(); algRank++) {
                if (myAlgRank == algRank) {
                    continue;
                }
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] algRank[%d]", algRank);
                u32 remoteRank = subCommRanks_[0][algRank];
                HCCL_INFO("[InsTempScatterMesh1D][RunMesh] myRank[%d], toRank[%d]", myRank_, remoteRank);
                const ChannelInfo &linkSend = channels.at(remoteRank)[0];
                u64 srcOffset = tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER
                                    ? tempAlgParams.buffInfo.hcclBuffBaseOff + r * tempAlgParams.inputRepeatStride +
                                          algRank * tempAlgParams.inputSliceStride
                                    : r * tempAlgParams.inputRepeatStride + algRank * tempAlgParams.inputSliceStride +
                                          tempAlgParams.buffInfo.inBuffBaseOff;
                u64 dstOffset =
                    tempAlgParams.buffInfo.hcclBuffBaseOff + r * tempAlgParams.outputRepeatStride;  // 暂不支持ZeroCopy，简化逻辑
                void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] srcOffset[%d], tempAlgParams.buffInfo.inputPtr[%d]", srcOffset, tempAlgParams.buffInfo.inputPtr);
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] dstOffset[%d], remoteCclBuffAddr[%d]", srcOffset, remoteCclBuffAddr);
                DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, processSize_, count_);
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] got srcSlice");
                DataSlice dstSlice = DataSlice(remoteCclBuffAddr, dstOffset, processSize_, count_);
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] got dstSlice");
                SlicesList txSlicesList({srcSlice}, {dstSlice});

                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] tempAlgParam.buffInfo.hcclBuff.addr[%d], tempAlgParams.buffInfo.inputPtr[%d], tempAlgParams.buffInfo.outputPtr[%d], ", tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.outputPtr);
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] remoteCclBuffAddr[%d]", remoteCclBuffAddr);
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] tempAlgParam.buffInfo.inBuffBaseOff[%d], tempAlgParam.buffInfo.outBuffBaseOff[%d], tempAlgParam.buffInfo.hcclBuffBaseOff[%d]", remoteCclBuffAddr, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.buffInfo.hcclBuffBaseOff);

                DataInfo sendData(linkSend, txSlicesList);
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] start SendWrite");
                CHK_PRT_RET(static_cast<HcclResult>(SendWrite(sendData, threads.at(count))),
                    HCCL_ERROR("[InsTempScatterMesh1D] RunMesh Send failed"),
                    HcclResult::HCCL_E_INTERNAL);
                HCCL_DEBUG("[InsTempScatterMesh1D][RunMesh] end SendWrite");
                count++;
            }
        } else {
            if(channels.size() == 0 || channels.count(root_) == 0){
                continue;
            }
            const ChannelInfo &linkRecv = channels.at(root_)[0];
            u64 srcOffset = tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER
                    ? tempAlgParams.buffInfo.hcclBuffBaseOff + r * tempAlgParams.inputRepeatStride +
                          myAlgRank * tempAlgParams.inputSliceStride
                    : r * tempAlgParams.inputRepeatStride + myAlgRank * tempAlgParams.inputSliceStride +
                          tempAlgParams.buffInfo.inBuffBaseOff;
            u64 dstOffset = tempAlgParams.buffInfo.hcclBuffBaseOff + r * tempAlgParams.outputRepeatStride;
            DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, processSize_, count_);
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, processSize_, count_);
            SlicesList rxSlicesList({srcSlice}, {dstSlice});
            DataInfo recvData(linkRecv, rxSlicesList);
            CHK_PRT_RET(static_cast<HcclResult>(RecvWrite(recvData, threads.at(0))),
                HCCL_ERROR("[InsTempScatterMesh1D] RunMesh Recv failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
}  // namespace ops_hccl
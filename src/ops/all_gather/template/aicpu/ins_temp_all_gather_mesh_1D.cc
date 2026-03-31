/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
namespace ops_hccl {
InsTempAllGatherMesh1D::InsTempAllGatherMesh1D(const OpParam &param, const u32 rankId,
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}
InsTempAllGatherMesh1D::~InsTempAllGatherMesh1D() {}

HcclResult InsTempAllGatherMesh1D::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                           AlgResourceRequest &resourceRequest)
{
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherMesh1D::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1D::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
}

u64 InsTempAllGatherMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 0;
    if (opMode_ == OpMode::OPBASE){
        scratchMultiple = templateRankSize_;
    }
    return scratchMultiple;
}

HcclResult InsTempAllGatherMesh1D::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                             const TemplateResource &templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAllGatherMesh1D] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize ==0) {
        HCCL_INFO("[InsTempAllGatherMesh1D] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    HCCL_DEBUG("[InsTempAllGatherMesh1D] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);
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
    if (opMode_ == OpMode::OPBASE) {
        CHK_RET(PostLocalCopy(templateResource.threads));
    }
    HCCL_INFO("[InsTempAllGatherMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                                    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] RunAllGatherMesh RankIDs[%d].", myRank_);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    u64 sliceSize = tempAlgParams_.sliceSize;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    // 当输入为hcclbuffer时，可以直接用read模式+dma消减，跳过后拷贝
    bool dmaRead = (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER && tempAlgParams_.buffInfo.outBuffType != BufferType::HCCL_BUFFER);
    // 尾块模式 + dmawrite，本端write数据大小
    if (tempAlgParams_.tailSize !=0 && !dmaRead && myAlgRank == templateRankSize_ -1) {
        sliceSize = tempAlgParams_.tailSize;
    }
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
            u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];

            u32 connectedAlgRank = 0;
            CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
            HCCL_INFO("[InsTempAllGatherMesh1D] RunAllGatherMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
                      myRank_, connectedRank, connectedAlgRank);

            CHK_PRT_RET(threadIdx >= threads.size() || channels.count(connectedRank) == 0 || 
                        channels.at(connectedRank).empty(),
                        HCCL_ERROR("[InsTempAllGatherMesh1D][RankID]=%u threadIdx=%u, threads.size=%u, "
                                   "connectedRank=%d, channels.size=%u",
                                   myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                        HcclResult::HCCL_E_INTERNAL);
            if (dmaRead) {
                // 对端为最后一个rank，读取tailSize大小
                if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
                    sliceSize = tempAlgParams_.tailSize;
                } else {
                    sliceSize = tempAlgParams_.sliceSize;
                }
            }
            const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
            void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;

            u64 txOutOffset = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;
            u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank;
            u64 txDstOffset = (!enableRemoteMemAccess_) ? txScratchOffset : txOutOffset;

            u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff;
            u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank;
            u64 rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;

            void *txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
            void *txDstPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
            void *rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
            // write模式使用tx, rx地址不生效，仅使用对端link做Post/Wait
            // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
            u64 sliceCount = sliceSize / dataTypeSize;
            std::vector<DataSlice> txSrcSlices{
                DataSlice(txSrcPtr, txOutOffset, sliceSize, sliceCount)};  // 本地(send)
            std::vector<DataSlice> txDstSlices{
                DataSlice(txDstPtr, txDstOffset, sliceSize, sliceCount)};  // 远程(send)
            // read模式使用rx
            std::vector<DataSlice> rxDstSlices{
                DataSlice(rxDstPtr, rxOutOffset, sliceSize, sliceCount)};  // 本地(recv)
            std::vector<DataSlice> rxSrcSlices{
                DataSlice(rxSrcPtr, rxSrcOffset, sliceSize, sliceCount)};  // 远程(recv)

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] txSrcSlices: "
                       "offset[%d] sliceSize[%d] count[%d].",
                       myRank_, connectedRank, txOutOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] txDstSlices: "
                       "offset[%d] sliceSize[%d] count[%d].",
                       myRank_, connectedRank, txDstOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rxSrcSlices: "
                       "offset[%d] sliceSize[%d] count[%d].",
                       myRank_, connectedRank, rxOutOffset, sliceSize, sliceCount);

            HCCL_DEBUG("[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rxDrcSlices: "
                       "offset[%d] sliceSize[%d] count[%d].",
                       myRank_, connectedRank, rxSrcOffset, sliceSize, sliceCount);

            TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
            TxRxChannels sendRecvChannels(linkRemote, linkRemote);
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
            // 当数据本身就在cclbuffer上时，直接使用DMA消减读模式到output上，同事要跳过后拷贝
            if (dmaRead){
                CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[threadIdx]),
                            HCCL_ERROR("[InsTempAllGatherMesh1D] RunAllGather Send failed"), HcclResult::HCCL_E_INTERNAL);
            } else {
                CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                            HCCL_ERROR("[InsTempAllGatherMesh1D] RunAllGather Send failed"), HcclResult::HCCL_E_INTERNAL);
            }

        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] LocalDataCopy.");

    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;
    // 尾块模式
    if (tempAlgParams_.tailSize !=0 && myAlgRank == templateRankSize_ -1) {
        sliceSize = tempAlgParams_.tailSize;
    }
    u64 sliceCount = sliceSize / dataTypeSize;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        // repeat 造成的偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        // 数据块rank编号造成的偏移
        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
        const u64 outOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);
        if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr && inOff == outOff) {
            continue;
        }
        HCCL_DEBUG("[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] srcSlice: inBaseOff[%d] inOff[%d] "
                   "sliceSize[%d] count[%d].",
                   myRank_, myAlgRank, inBaseOff, inOff, sliceSize, sliceCount);
        HCCL_DEBUG("[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] dstSlice: outBaseoff[%d] "
                   "outOff[%d] sliceSize[%d] count[%d].",
                   myRank_, myAlgRank, outBaseOff, outOff, sliceSize, sliceCount);

        LocalCopy(threads[0], srcSlice, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy.");
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy skip because output is scratch" );
        return HcclResult::HCCL_SUCCESS;
    }
    if (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy skip because input is scratch and should be read to output" );
        return HcclResult::HCCL_SUCCESS;
    }
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;
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
            // 尾块模式
            if (tempAlgParams_.tailSize !=0 && algRank == templateRankSize_ -1) {
                sliceSize = tempAlgParams_.tailSize;
            }
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase;
            u64 outOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff;
            u64 sliceCount = sliceSize / dataTypeSize;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, sliceSize, sliceCount);
            HCCL_DEBUG("[InsTempAllGatherMesh1D] LocalDataCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                       "scratchBase[%d] outBaseOff[%d] scratchOffset[%d] outOffset[%d].",
                       myRank_, rank, algRank, outBaseOff, outBaseOff, scratchOffset, outOffset);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

}  // namespace ops_hccl
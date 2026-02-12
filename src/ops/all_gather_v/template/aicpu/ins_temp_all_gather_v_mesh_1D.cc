/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_v_mesh_1D.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
namespace ops_hccl {
InsTempAllGatherVMesh1D::InsTempAllGatherVMesh1D(const OpParam &param, const u32 rankId,  // 传通信域的rankId，userRank
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}
InsTempAllGatherVMesh1D::~InsTempAllGatherVMesh1D() {}
 
HcclResult InsTempAllGatherVMesh1D::CalcRes(HcclComm comm, const OpParam &param, const TopoInfo *topoInfo,
                                           AlgResourceRequest &resourceRequest)
{
    // mesh 算法只做level 0 层级的
    GetResWithoutLinks(resourceRequest);
 
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherVMesh1D::GetResWithoutLinks(AlgResourceRequest &resourceRequest)
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;  // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}
 
u64 InsTempAllGatherVMesh1D::GetThreadNum()
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
}
 
u64 InsTempAllGatherVMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}
 
HcclResult InsTempAllGatherVMesh1D::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                             const TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempAllGatherVMesh1D] Run start");
    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("[InsTempAllGatherVMesh1D] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }
    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.vDataDes.dataType;
    HCCL_DEBUG("[InsTempAllGatherVMesh1D] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);
    CHK_RET(LocalDataCopy(templateResource.threads));
    if (templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
 
    CHK_RET(RunAllGatherVMesh(templateResource.threads, templateResource.channels));
 
    if (threadNum_ > 1) {
        // 定义在alg_v2_template_base.h中的基类
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    CHK_RET(PostLocalCopy(templateResource.threads));
    HCCL_INFO("[InsTempAllGatherVMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempAllGatherVMesh1D::RunAllGatherVMesh(const std::vector<ThreadHandle> &threads,
                                                    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempAllGatherVMesh1D] RunAllGatherVMesh RankIDs[%d].", myRank_);
    u64 dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * dataTypeSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
 
        for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
            u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];
 
            u32 connectedAlgRank = 0;
            CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
            HCCL_INFO("[InsTempAllGatherVMesh1D] RunAllGatherVMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].",
                      myRank_, connectedRank, connectedAlgRank);
 
            // 异常检查
            CHK_PRT_RET(threadIdx >= threads.size() || !channels.count(connectedRank),
                        HCCL_ERROR("[InsTempAllGatherVMesh1D][RankID]=%u threadIdx=%u, threads.size=%u, "
                                   "connectedRank=%d, channels.size=%u",
                                   myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                        HcclResult::HCCL_E_INTERNAL);
 
            ThreadHandle currQue = threads[threadIdx];
            // 预留兼容offload模式
            const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
            void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
 
            u64 txOutOffset = tempAlgParams_.allRankDispls[myAlgRank] * dataTypeSize_ + outBaseOff;
            u64 txScratchOffset = scratchBase + tempAlgParams_.sliceSize * myAlgRank;
            u64 txDstOffset = txScratchOffset;
 
            u64 rxOutOffset = tempAlgParams_.allRankDispls[connectedAlgRank] * dataTypeSize_ + outBaseOff;
            u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank;
            u64 rxSrcOffset = rxScratchOffset;
 
            void *txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
            void *txDstPtr = remoteCclBuffAddr;
            void *rxSrcPtr = remoteCclBuffAddr;
            void *rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
            // write模式使用tx,rx地址不生效，仅使用对端link做Post/Wait
            // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
            std::vector<DataSlice> txSrcSlices{
                DataSlice(txSrcPtr, txOutOffset, tempAlgParams_.allRankSliceSize[myAlgRank], tempAlgParams_.count)};  // 本地(send)
            std::vector<DataSlice> txDstSlices{
                DataSlice(txDstPtr, txDstOffset, tempAlgParams_.allRankSliceSize[myAlgRank], tempAlgParams_.count)};  // 远程(send)
            // read模式使用rx
            std::vector<DataSlice> rxDstSlices{
                DataSlice(rxDstPtr, rxSrcOffset, tempAlgParams_.allRankSliceSize[connectedAlgRank], tempAlgParams_.count)};  // 本地(recv)
            std::vector<DataSlice> rxSrcSlices{
                DataSlice(rxSrcPtr, rxOutOffset, tempAlgParams_.allRankSliceSize[connectedAlgRank], tempAlgParams_.count)};  // 远程(recv)
 
            HCCL_DEBUG("[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] txSrcSlices: "
                       "offset[%d] sliceSize[%d] count[%d].",
                       myRank_, connectedRank, txOutOffset, tempAlgParams_.allRankSliceSize[myAlgRank], tempAlgParams_.count);
 
            HCCL_DEBUG("[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] txDstSlices: "
                       "offset[%d] sliceSize[%d] count[%d].",
                       myRank_, connectedRank, txDstOffset, tempAlgParams_.allRankSliceSize[myAlgRank], tempAlgParams_.count);
 
            HCCL_DEBUG("[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] rxSrcSlices: "
                       "offset[%d] sliceSize[%d] count[%d].",
                       myRank_, connectedRank, rxOutOffset, tempAlgParams_.allRankSliceSize[connectedAlgRank], tempAlgParams_.count);
 
            HCCL_DEBUG("[InsTempAllGatherVMesh1D][RunAllGatherVMesh] rankId [%d] connectedRank [%d] rxDrcSlices: "
                       "offset[%d] sliceSize[%d] count[%d].",
                       myRank_, connectedRank, rxSrcOffset, tempAlgParams_.allRankSliceSize[connectedAlgRank], tempAlgParams_.count);
 
            TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
            TxRxChannels sendRecvChannels(linkRemote, linkRemote);
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
 
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                        HCCL_ERROR("[InsTempAllGatherVMesh1D] RunAllGatherV Send failed"), HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempAllGatherVMesh1D::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherVMesh1D] LocalDataCopy.");
    if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr) {
        return HcclResult::HCCL_SUCCESS;
    }
    u32 myAlgRank;
    u64 dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        // repeat 造成的偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        // 数据块rank编号造成的偏移
        const u64 inOff = inBaseOff;
        const u64 outOff = tempAlgParams_.allRankDispls[myAlgRank] * dataTypeSize_ + outBaseOff;
 
        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, tempAlgParams_.allRankSliceSize[myAlgRank], tempAlgParams_.count);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, tempAlgParams_.allRankSliceSize[myAlgRank], tempAlgParams_.count);
 
        HCCL_DEBUG("[InsTempAllGatherVMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] srcSlice: inBaseOff[%d] inOff[%d] "
                   "sliceSize[%d] count[%d].",
                   myRank_, myAlgRank, inBaseOff, inOff, tempAlgParams_.allRankSliceSize[myAlgRank], tempAlgParams_.count);
        HCCL_DEBUG("[InsTempAllGatherVMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] dstSlice: outBaseoff[%d] "
                   "outOff[%d] sliceSize[%d] count[%d].",
                   myRank_, myAlgRank, outBaseOff, outOff, tempAlgParams_.allRankSliceSize[myAlgRank], tempAlgParams_.count);
 
        LocalCopy(threads[0], srcSlice, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempAllGatherVMesh1D::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherVMesh1D] PostLocalCopy.");
    u64 dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
 
        for (auto rank : subCommRanks_[0]) {
            if (rank == myRank_) {
                continue;
            }
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase;
            u64 outBaseOff = tempAlgParams_.allRankProcessedDataCount[algRank] * dataTypeSize_ + rpt * tempAlgParams_.outputRepeatStride;
            u64 outOffset = tempAlgParams_.allRankDispls[algRank] * dataTypeSize_ + outBaseOff;
            u64 processSize = tempAlgParams_.allRankSliceSize[algRank];
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, processSize,
                               tempAlgParams_.count);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, processSize,
                               tempAlgParams_.count);
            HCCL_DEBUG("[InsTempAllGatherVMesh1D] PostLocalCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                       "scratchBase[%d] outBaseOff[%d] scratchOffset[%d] outOffset[%d].",
                       myRank_, rank, algRank, scratchBase, outBaseOff, scratchOffset, outOffset);
            LocalCopy(threads[0], srcSlice, dstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherVMesh1D::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAllGatherVMesh1D::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}
 
}  // namespace ops_hccl
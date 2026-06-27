/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_allgather_mesh_1D_intra.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
namespace ops_hccl {
InsTempAllGatherMesh1DIntra::InsTempAllGatherMesh1DIntra(const OpParam &param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}
InsTempAllGatherMesh1DIntra::~InsTempAllGatherMesh1DIntra() {}
 
HcclResult InsTempAllGatherMesh1DIntra::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    // mesh 算法只做level 0 层级的
    GetResWithoutLinks(resourceRequest);
 
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherMesh1DIntra::GetResWithoutLinks(AlgResourceRequest &resourceRequest)
{
    u32 level0RankSize = templateRankSize_;
    u32 threadNum = level0RankSize > 1 ? level0RankSize - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;  // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}
 
u64 InsTempAllGatherMesh1DIntra::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
}
 
u64 InsTempAllGatherMesh1DIntra::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}
 
HcclResult InsTempAllGatherMesh1DIntra::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
    TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempAllGatherMesh1DIntra] Run start");

    threadNum_ = templateResource.threads.size();
    tempAlgParams_ = tempAlgParams;
    sliceSize_ = tempAlgParams.sliceSize;
    tailSize_ = tempAlgParams.tailSize;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    HCCL_DEBUG("[InsTempAllGatherMesh1DIntra] Rank [%d], get threadNum_[%d].", myRank_, threadNum_);

    if (templateRankSize_ == 1) {
        CHK_RET(PostLocalCopy(templateResource.threads));
        HCCL_INFO("[InsTempAllGatherMesh1DIntra] RankSize == 1, Run End");
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
    HCCL_INFO("[InsTempAllGatherMesh1DIntra] Run End");
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempAllGatherMesh1DIntra::RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
    const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    HCCL_DEBUG("[InsTempAllGatherMesh1DIntra] RunAllGatherMesh RankIDs[%d], myAlgRank[%d].", myRank_, myAlgRank);

    u64 localSize = tempAlgParams_.allRankSliceSize.at(myAlgRank);
    u64 localCount = tempAlgParams_.allRankProcessedDataCount.at(myAlgRank);
    u64 localOffset = tempAlgParams_.allRankDispls.at(myAlgRank);

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
 
        for (u32 threadIdx = 0; threadIdx < subCommRanks_[0].size() - 1; threadIdx++) {
            u32 connectedRank = subCommRanks_[0][(myAlgRank + 1 + threadIdx) % subCommRanks_[0].size()];

            u32 connectedAlgRank = 0;
            CHK_RET(GetAlgRank(connectedRank, subCommRanks_[0], connectedAlgRank));
            HCCL_DEBUG("[InsTempAllGatherMesh1DIntra] RunAllGatherMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].", 
                myRank_, connectedRank, connectedAlgRank);
 
            // 异常检查
            CHK_PRT_RET(threadIdx >= threads.size() || !channels.count(connectedRank),
                        HCCL_ERROR("[InsTempAllGatherMesh1DIntra][RankID]=%u threadIdx=%u, threads.size=%u, "
                                   "connectedRank=%d, channels.size=%u",
                                   myRank_, threadIdx, threads.size(), connectedRank, channels.size()),
                        HcclResult::HCCL_E_INTERNAL);

            u64 remoteSize = tempAlgParams_.allRankSliceSize.at(connectedAlgRank);
            u64 remoteCount = tempAlgParams_.allRankProcessedDataCount.at(connectedAlgRank);
            u64 remoteOffset = tempAlgParams_.allRankDispls.at(connectedAlgRank);
    
            // 既不发送也不接受
            if (localSize == 0 && remoteSize == 0) {
                continue;
            }
            
            ThreadHandle currQue = threads[threadIdx];
            // 预留兼容offload模式
            const ChannelInfo &linkRemote = channels.at(connectedRank)[0];
            void *remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
 
            u64 txOutOffset = outBaseOff + localOffset;
            u64 txScratchOffset = scratchBase + localOffset;
            u64 txDstOffset = txScratchOffset;
 
            u64 rxOutOffset = outBaseOff + remoteOffset;
            u64 rxScratchOffset = scratchBase + remoteOffset;
            u64 rxSrcOffset = rxScratchOffset;
 
            void *txSrcPtr = tempAlgParams_.buffInfo.hcclBuff.addr;
            void *txDstPtr = remoteCclBuffAddr;
            void *rxSrcPtr = tempAlgParams_.buffInfo.hcclBuff.addr;
            void *rxDstPtr = remoteCclBuffAddr;
            // write模式使用tx,rx地址不生效，仅使用对端link做Post/Wait
            // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
            std::vector<DataSlice> txSrcSlices{
                DataSlice(txSrcPtr, txOutOffset, localSize, localCount)};  // 本地(send)
            std::vector<DataSlice> txDstSlices{
                DataSlice(txDstPtr, txDstOffset, localSize, localCount)};  // 远程(send)
            // read模式使用rx
            std::vector<DataSlice> rxDstSlices{
                DataSlice(rxDstPtr, rxSrcOffset, remoteSize, remoteCount)};  // 本地(recv)
            std::vector<DataSlice> rxSrcSlices{
                DataSlice(rxSrcPtr, rxOutOffset, remoteSize, remoteCount)};  // 远程(recv)
 
            if (localSize == 0) { // 发送数据片为0时，只接收数据
                SlicesList recvSliceList(rxSrcSlices, rxDstSlices);
                DataInfo recvInfo(linkRemote, recvSliceList);
                HCCL_INFO("[InsTempAllGatherMesh1DIntra][RecvOnly] myRank=%u connRank=%u thread=%u rpt=%u remoteSize=%llu",
                    myRank_, connectedRank, threadIdx, rpt, remoteSize);
                CHK_PRT_RET(RecvWrite(recvInfo, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherMesh1DIntra][RunAllGatherMesh] Recv failed."),
                    HcclResult::HCCL_E_INTERNAL);
            } else if (remoteSize == 0) { // 接收数据片为0时，只发送数据
                SlicesList sendSliceList(txSrcSlices, txDstSlices);
                DataInfo sendInfo(linkRemote, sendSliceList);
                HCCL_INFO("[InsTempAllGatherMesh1DIntra][SendOnly] myRank=%u connRank=%u thread=%u rpt=%u localSize=%llu",
                    myRank_, connectedRank, threadIdx, rpt, localSize);
                CHK_PRT_RET(SendWrite(sendInfo, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherMesh1dIntra][RunAllGatherMesh] Send failed."),
                    HcclResult::HCCL_E_INTERNAL);
            } else {
                SendRecvInfo sendRecvInfo{
                    {linkRemote, linkRemote}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
                HCCL_INFO("[InsTempAllGatherMesh1DIntra][SendRecv] myRank=%u connRank=%u thread=%u rpt=%u localSize=%llu remoteSize=%llu",
                    myRank_, connectedRank, threadIdx, rpt, localSize, remoteSize);
                CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                    HCCL_ERROR("[InsTempAllGatherMesh1DIntra][RunAllGatherMesh] Send failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempAllGatherMesh1DIntra::LocalDataCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1DIntra] LocalDataCopy.");
    if (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr) {
        return HcclResult::HCCL_SUCCESS;
    }
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    u64 sliceSize = tempAlgParams_.allRankSliceSize.at(myAlgRank);
    u64 sliceCount = tempAlgParams_.allRankProcessedDataCount.at(myAlgRank);
    u64 sliceOffset = tempAlgParams_.allRankDispls.at(myAlgRank);

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        // repeat 造成的偏移
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        // 数据块rank编号造成的偏移
        const u64 inOff = inBaseOff;
        const u64 outOff = sliceOffset + outBaseOff;
 
        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);
 
        HCCL_DEBUG("[InsTempAllGatherMesh1DIntra][LocalCopy] RankID [%d] AlgRank [%d] srcSlice: inBaseOff[%d] inOff[%d]"
                   "sliceSize[%d] count[%d] inBuffType[%d] inputPtr[%u].",
                   myRank_, myAlgRank, inBaseOff, inOff, sliceSize, sliceCount, tempAlgParams_.buffInfo.inBuffType,
                   tempAlgParams_.buffInfo.inputPtr);
        HCCL_DEBUG("[InsTempAllGatherMesh1DIntra][LocalCopy] RankID [%d] AlgRank [%d] dstSlice: outBaseoff[%d] "
                   "outOff[%d] sliceSize[%d] count[%d] outBuffType[%d] outputPtr[%u].",
                   myRank_, myAlgRank, outBaseOff, outOff, sliceSize, sliceCount, tempAlgParams_.buffInfo.outBuffType,
                   tempAlgParams_.buffInfo.outputPtr);

        LocalCopy(threads[0], srcSlice, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}
 
HcclResult InsTempAllGatherMesh1DIntra::PostLocalCopy(const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1DIntra] PostLocalCopy.");
 
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        u64 sliceCount = tempAlgParams_.count;
        u64 sliceSize = sliceCount * dataTypeSize_;

        u64 scratchOffset = scratchBase;
        u64 outOffset = outBaseOff;
        DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize,
                            sliceCount);
        DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, sliceSize,
                            sliceCount);
        LocalCopy(threads[0], srcSlice, dstSlice);
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherMesh1DIntra::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = threadNum_;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}
 
void InsTempAllGatherMesh1DIntra::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = threadNum_;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

}  // namespace ops_hccl
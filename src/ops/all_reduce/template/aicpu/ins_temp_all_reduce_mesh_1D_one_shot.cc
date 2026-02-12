/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_reduce_mesh_1D_one_shot.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
InsTempAllReduceMesh1DOneShot::InsTempAllReduceMesh1DOneShot(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempAllReduceMesh1DOneShot::~InsTempAllReduceMesh1DOneShot()
{
}

HcclResult InsTempAllReduceMesh1DOneShot::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
                                               AlgResourceRequest& resourceRequest)
{
    // mesh 算法只做level 0 层级的
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    resourceRequest.slaveThreadNum = threadNum - 1; // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

u64 InsTempAllReduceMesh1DOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_;
    return scratchMultiple;
}

HcclResult InsTempAllReduceMesh1DOneShot::CalcSlice(const u64 dataSize, RankSliceInfo &sliceInfoVec)
{
    std::vector<SliceInfo> tmp(1);
    sliceInfoVec.resize(templateRankSize_, tmp);

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < sliceInfoVec.size(); rankIdx++) {
        SliceInfo slice = {accumOff, dataSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += dataSize;
    }
    CHK_PRT_RET(
        (sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size != dataSize * templateRankSize_),
        HCCL_ERROR("[InsTempAllReduceMesh1DOneShot] Rank [%d], SliceInfo calculation error!", myRank_), HcclResult::HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DOneShot::KernelRun(const OpParam& param,
    const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    HCCL_INFO("[InsTempAllReduceMesh1DOneShot] Run Start");
    // 这里不支持绕路的时候，应该就用原始的tempInsQues就行
    CHK_PRT_RET(threadNum_ != templateResource.threads.size(),
                HCCL_ERROR("[InsTempAllReduceMesh1DOneShot] Rank [%d], requiredThread Error.", myRank_),
                HcclResult::HCCL_E_INTERNAL);
    
    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSlice(processSize_, sliceInfoVec));

    CHK_RET(RunAllReduce(templateResource.channels, templateResource.threads, tempAlgParams, sliceInfoVec));
    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][Run] AllReduceMesh1DOneShot finished: rank[%d] end", myRank_);
    return HCCL_SUCCESS;
}

HcclResult InsTempAllReduceMesh1DOneShot::RunAllReduce(const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                        const std::vector<ThreadHandle> &threads,
                                                        const TemplateDataParams &tempAlgParams,
                                                        const RankSliceInfo &sliceInfoVec)
{
    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][RunAllReduce] send/recv: rank[%d]", myRank_);

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
    }

    DataSlice usrInSlices = DataSlice(tempAlgParams.buffInfo.inputPtr,
                                    tempAlgParams.buffInfo.inBuffBaseOff,
                                    processSize_, count_);
    DataSlice usrOutSlices = DataSlice(tempAlgParams.buffInfo.outputPtr,
                                    tempAlgParams.buffInfo.outBuffBaseOff,
                                    processSize_, count_);

    // 主流 - 本地拷贝
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], usrInSlices, usrOutSlices)));

    // 特殊场景
    if (subCommRanks_[0].size() == 1) {
        return HCCL_SUCCESS;
    }

    // 从流
    for (u32 queIdx = 1; queIdx < threadNum_; queIdx++) {
        u32 nextRank = (myRank_ + queIdx) % templateRankSize_;
        u32 fromRank = subCommRanks_[0][nextRank];
        u32 toRank = subCommRanks_[0][nextRank];

        const ChannelInfo &linkRecv = channels.at(fromRank)[0]; // linkRecv - 从fromRank接收的链路
        const ChannelInfo &linkSend = channels.at(toRank)[0]; // linkSend - 向toRank发送的链路

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        void* txCclBuffAddr = linkSend.remoteCclMem.addr;
        u64 txDstOffset   = sliceInfoVec[myRank_][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff;
        u64 txDstSize     = sliceInfoVec[myRank_][0].size;
        DataSlice txSrcSlice = usrInSlices;
        DataSlice txDstSlice = DataSlice(txCclBuffAddr, txDstOffset, txDstSize, count_);
        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);

        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        void* rxCclBuffAddr = linkRecv.remoteCclMem.addr;
        u64 rxDstOffset   = sliceInfoVec[fromRank][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff;
        u64 rxDstSize     = sliceInfoVec[fromRank][0].size;
        DataSlice rxSrcSlice = usrInSlices;
        DataSlice rxDstSlice = DataSlice(rxCclBuffAddr, rxDstOffset, rxDstSize, count_);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);

        SendRecvInfo sendRecvInfo{{linkSend, linkRecv},
                             {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
        CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[queIdx]),
            HCCL_ERROR("[InsTempAllReduceMesh1DOneShot] RunAllReduce SendRecv failed"),
            HcclResult::HCCL_E_INTERNAL);
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][RunAllReduce] reduce: rank[%d]", myRank_);
    for (u32 rankIdx = 0; rankIdx < subCommRanks_[0].size(); rankIdx++) {
        u32 curRank = rankIdx;
        // 遍历除自身外所有rank，计算reduce(scratch, local-usrout)
        if (curRank == myRank_) {
            continue;
        }

        // 执行本地归约
        void *RemotePtr = tempAlgParams.buffInfo.hcclBuff.addr;
        u64 curSrcOffset = sliceInfoVec[curRank][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff;
        u64 curSrcSize = sliceInfoVec[curRank][0].size;
        DataSlice curSrcSlice = DataSlice(RemotePtr, curSrcOffset, curSrcSize, count_);
        DataSlice curDstSlice = usrOutSlices;

        CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], curSrcSlice, curDstSlice, dataType_, reduceOp_)));
    }

    return HCCL_SUCCESS;
}

void InsTempAllReduceMesh1DOneShot::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempAllReduceMesh1DOneShot::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

}
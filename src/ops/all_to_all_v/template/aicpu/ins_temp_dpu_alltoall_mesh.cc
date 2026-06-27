/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_dpu_alltoall_mesh.h"

#define NET_NUM 2

namespace ops_hccl {

InsTempDpuAlltoAllMesh::InsTempDpuAlltoAllMesh() {}
InsTempDpuAlltoAllMesh::InsTempDpuAlltoAllMesh(const OpParam &param, const u32 rankId,
                                               const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempDpuAlltoAllMesh::~InsTempDpuAlltoAllMesh() {}

HcclResult InsTempDpuAlltoAllMesh::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                           AlgResourceRequest &resourceRequest)
{
    u32 threadNum = 0;
    std::vector<HcclChannelDesc> level0Channels;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        // 框内threadNum最大取MAX_RANK_NUM_PER_SERVER
        threadNum = (templateRankSize_ > MAX_RANK_NUM_PER_SERVER) ? MAX_RANK_NUM_PER_SERVER :
                    (templateRankSize_ > 1)                       ? (templateRankSize_ - 1) :
                                                                    1;
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    } else {
        CHK_PRT_RET(subCommRanks_.size() != NET_NUM,
                    HCCL_ERROR("[InsTempDpuAlltoAllMesh][CalcRes] subCommRankNum[%zu] is not [%u]",
                    subCommRanks_.size(), NET_NUM), HCCL_E_PARA);
        u32 intraRankNum = subCommRanks_[0].size();
        threadNum = (intraRankNum > 1) ? (intraRankNum - 1) : 1;
        subCommRanks_ = {subCommRanks_[1]};
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_,
                                                         level0Channels, CommTopo::COMM_TOPO_1DMESH));
    }
    // 计算从流以及Notify数量
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    resourceRequest.channels.push_back(level0Channels);
    HCCL_DEBUG("[InsTempDpuAlltoAllMesh][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u]", myRank_,
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempDpuAlltoAllMesh::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // hcclbuf切分为CCL_IN和CCL_OUT,每部分切分为ranksize块hcclbuffBlockMem
    u64 scratchMultiple = CCLBUF_SPLIT_PARTS * templateRankSize_;
    HCCL_INFO("[InsTempDpuAlltoAllMesh] scratchMultiple[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempDpuAlltoAllMesh::KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                                             TemplateResource &templateResource)
{
    // HCCL_BUF均分为CCL_IN和CCL_OUT两部分
    halfMaxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size / CCLBUF_SPLIT_PARTS;
    hcclbuffBlockMemSize_ = tempAlgParams.inputSliceStride;
    threads_ = templateResource.threads;
    threadNum_ = threads_.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = SIZE_TABLE[dataType_];
    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempDpuAlltoAllMesh] Rank [%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(subCommRanks_.empty(), HCCL_ERROR("[InsTempDpuAlltoAllMesh] [KernelRun] subCommRanks is empty"),
                HCCL_E_PARA);
    CHK_PRT_RET(subCommRanks_[0].empty(), HCCL_ERROR("[InsTempDpuAlltoAllMesh] [KernelRun] subCommRanks[0] is empty"),
                HCCL_E_PARA);

    CHK_RET(LocalCopyforMyRank(subCommRanks_[0], tempAlgParams, threads_));

    // 将待发送数据从sendBuf拷贝到CCL_IN上
    CHK_RET(PreCopyDataToCclInBuf(subCommRanks_[0], tempAlgParams, threads_));

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempDpuAlltoAllMesh] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
    if (HcommThreadSynchronize(threads_[0]) != 0) {
        HCCL_ERROR("HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }
    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempDpuAlltoAllMesh] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    // 从CCL_IN发送数据给对端CCL_OUT
    CHK_RET(SendRecvData(param, subCommRanks_[0], tempAlgParams, templateResource, threads_));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    // 将接收到的数据从CCL_OUT拷贝recvBuf
    CHK_RET(PostCopyDataToRecvBuf(subCommRanks_[0], tempAlgParams, threads_));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempDpuAlltoAllMesh::LocalCopyforMyRank(const std::vector<u32> &commRanks,
                                                      const TemplateDataParams &tempAlgParams,
                                                      std::vector<ThreadHandle> &threads)
{
    u64 sendCount = tempAlgParams.sendCounts[myRank_];
    u64 recvCount = tempAlgParams.recvCounts[myRank_];
    u64 sendSliceSize = sendCount * dataTypeSize_;
    u64 recvSliceSize = recvCount * dataTypeSize_;
    u64 sendOffset = tempAlgParams.sdispls[myRank_] * dataTypeSize_;
    u64 recvOffset = tempAlgParams.rdispls[myRank_] * dataTypeSize_;
    // 将待myRank的切片直接从sendBuf拷贝到recvBuf上
    DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, sendOffset, sendSliceSize, sendCount);
    DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, recvOffset, recvSliceSize, recvCount);
    if (sendCount != 0) {
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuAlltoAllMesh::PreCopyDataToCclInBuf(const std::vector<u32> &commRanks,
                                                         const TemplateDataParams &tempAlgParams,
                                                         std::vector<ThreadHandle> &threads)
{
    for (u32 i = 0; i < commRanks.size(); i++) {
        u32 remoteRank = commRanks[i];
        u64 sendCount = tempAlgParams.sendCounts[remoteRank];
        if (remoteRank != myRank_ && sendCount > 0) {
            u64 sendSliceSize = sendCount * dataTypeSize_;
            u64 sendOffset = tempAlgParams.sdispls[remoteRank] * dataTypeSize_;
            DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, sendOffset, sendSliceSize, sendCount);
            // 将待sendBuf上待发送数据，拷贝到HCCL_BUF的前半部分CCL_IN上
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                                           tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_,
                                           sendSliceSize, sendCount);
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuAlltoAllMesh::SendRecvData(const OpParam &param, const std::vector<u32> &commRanks,
                                                const TemplateDataParams &tempAlgParams,
                                                const TemplateResource &templateResource,
                                                std::vector<ThreadHandle> &threads)
{
    // 写共享内存，唤醒dpu线程
    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempDpuAlltoAllMesh";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.subCommRanks = subCommRanks_;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    u32 sendMsgId = 0;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();
    if (HcommSendRequest(reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
                         static_cast<void *>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId) != 0) {
        HCCL_ERROR("[InsTempDpuAlltoAllMesh] [SendRecvData] HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }
    // AICPU部分数据传输
    u32 threadIdx = 0;
    for (u32 i = 0; i < commRanks.size(); i++) {
        u32 remoteRank = commRanks[i];
        if (remoteRank == myRank_) {
            HCCL_INFO("[InsTempDpuAlltoAllMesh] [SendRecvData] myRank[%u] is eaqul with remoteRank[%u] skip aicpu data "
                      "transfer",
                      myRank_, remoteRank);
            continue;
        }

        // 获取channelInfo
        auto it = templateResource.channels.find(remoteRank);
        CHK_PRT_RET(
            it == templateResource.channels.end(),
            HCCL_ERROR("[InsTempDpuAlltoAllMesh] [SendRecvData] myRank[%u] key not found in channels map", remoteRank),
            HCCL_E_PARA);
        CHK_PRT_RET(templateResource.channels.at(remoteRank).empty(),
                    HCCL_ERROR("[InsTempDpuAlltoAllMesh] [SendRecvData] ChannelInfo Vector for myRank[%u] key is empty",
                               remoteRank),
                    HCCL_E_PARA);
        const ChannelInfo &link = templateResource.channels.at(remoteRank)[0];

        if (link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
            HCCL_WARNING("[InsTempDpuAlltoAllMesh][SendRecvData] skip myRank[%u] transfer data to remoteRank[%u] by "
                         "AICPU , the EndpointLocType must be DEVICE",
                         myRank_, remoteRank);
            continue;
        }

        if (threadIdx >= threadNum_) {
            HCCL_ERROR("[InsTempDpuAlltoAllMesh] [SendRecvData] thread index [%u] exceeds thread count [%u]", threadIdx,
                       threadNum_);
            return HCCL_E_INTERNAL;
        }

        u64 sendCount = tempAlgParams.sendCounts[remoteRank];
        u64 recvCount = tempAlgParams.recvCounts[remoteRank];
        // 无需发送和接收数据
        if (sendCount == 0 && recvCount == 0) {
            HCCL_INFO("[InsTempDpuAlltoAllMesh] [SendRecvData] myRank[%u] send data to remoteRank[%u] and myRank[%u] "
                      "recv data from remoteRank[%u] are zero, skip data transfer",
                      myRank_, remoteRank, myRank_, remoteRank);
            continue;
        }
        u64 sendSliceSize = sendCount * dataTypeSize_;
        u64 recvSliceSize = recvCount * dataTypeSize_;
        void *remoteCclBuffAddr = link.remoteCclMem.addr;
        if (remoteCclBuffAddr == nullptr) {
            HCCL_ERROR("[InsTempDpuAlltoAllMesh] [SendRecvData] myRank[%u] Remote CCL buffer address is null for "
                       "remoteRank[%u]",
                       myRank_, remoteRank);
            return HCCL_E_INTERNAL;
        }
        if (sendCount > 0 && recvCount > 0) {
            // 待发和待收数据量不为0的情况
            DataSlice sendSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_, sendSliceSize, sendCount);
            DataSlice sendDstSlice =
                DataSlice(remoteCclBuffAddr,
                          tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_,
                          sendSliceSize, sendCount);

            DataSlice recvSrcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                                               tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_,
                                               recvSliceSize, recvCount);
            DataSlice recvDstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
            std::vector<DataSlice> txSrcSlices{sendSrcSlice};
            std::vector<DataSlice> txDstSlices{sendDstSlice};
            std::vector<DataSlice> rxSrcSlices{recvSrcSlice};
            std::vector<DataSlice> rxDstSlices{recvDstSlice};
            HCCL_INFO("[InsTempDpuAlltoAllMesh] [SendRecvData] myRank[%u] send data to remoteRank[%u], myRank's CCLIN "
                      "startAddr[%llu] to remoteRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                      myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_,
                      tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_,
                      sendSliceSize);

            HCCL_INFO("[InsTempDpuAlltoAllMesh] [SendRecvData] myRank[%u] recv data from remoteRank[%u], remoteRank's "
                      "CCLIN startAddr[%llu] to myRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                      myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_,
                      tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                      recvSliceSize);
            SendRecvInfo sendRecvInfo{{link, link}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                        HCCL_ERROR("[InsTempDpuAlltoAllMesh] [SendRecvData] AlltoAll AICPU SendRecv failed"),
                        HcclResult::HCCL_E_INTERNAL);
        } else if (sendCount > 0) {
            // 待发数据量不为0,待收数据为0的情况
            DataSlice sendSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_, sendSliceSize, sendCount);
            DataSlice sendDstSlice =
                DataSlice(remoteCclBuffAddr,
                          tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_,
                          sendSliceSize, sendCount);
            std::vector<DataSlice> txSrcSlices{sendSrcSlice};
            std::vector<DataSlice> txDstSlices{sendDstSlice};
            HCCL_INFO("[InsTempDpuAlltoAllMesh] [SendRecvData] myRank[%u] send data to remoteRank[%u], myRank's CCLIN "
                      "startAddr[%llu] to remoteRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                      myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_,
                      tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_,
                      sendSliceSize);
            DataInfo sendDataInfo{link, {txSrcSlices, txDstSlices}};
            CHK_PRT_RET(SendWrite(sendDataInfo, threads[threadIdx]),
                        HCCL_ERROR("[InsTempDpuAlltoAllMesh] [SendRecvData] AlltoAll AICPU only Send failed"),
                        HcclResult::HCCL_E_INTERNAL);
        } else if (recvCount > 0) {
            // 待收数据量不为0,待发数据为0的情况
            DataSlice recvSrcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                                               tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_,
                                               recvSliceSize, recvCount);
            DataSlice recvDstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
            std::vector<DataSlice> rxSrcSlices{recvSrcSlice};
            std::vector<DataSlice> rxDstSlices{recvDstSlice};
            HCCL_INFO("[InsTempDpuAlltoAllMesh] [SendRecvData] myRank[%u] recv data from remoteRank[%u], remoteRank's "
                      "CCLIN startAddr[%llu] to myRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                      myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_,
                      tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                      recvSliceSize);
            DataInfo recvDataInfo{link, {rxSrcSlices, rxDstSlices}};
            CHK_PRT_RET(RecvWrite(recvDataInfo, threads[threadIdx]),
                        HCCL_ERROR("[InsTempDpuAlltoAllMesh] [SendRecvData] AlltoAll AICPU only Recv failed"),
                        HcclResult::HCCL_E_INTERNAL);
        }
        threadIdx++;
    }

    // 等待dpu完成
    HCCL_INFO("[InsTempDpuAlltoAllMesh] [SendRecvData] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);
    // 等待DPU数据传输，然后回写结果回来
    void *recvData = nullptr;
    u32 recvMsgId = 0;

    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempDpuAlltoAllMesh][SendRecvData] HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO(" [InsTempDpuAlltoAllMesh] [SendRecvData]HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("[InsTempDpuAlltoAllMesh] [SendRecvData]recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId,
                   sendMsgId);
        return HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuAlltoAllMesh::PostCopyDataToRecvBuf(const std::vector<u32> &commRanks,
                                                         const TemplateDataParams &tempAlgParams,
                                                         std::vector<ThreadHandle> &threads)
{
    for (u32 i = 0; i < commRanks.size(); i++) {
        u32 remoteRank = commRanks[i];
        u64 recvCount = tempAlgParams.recvCounts[remoteRank];
        if (remoteRank != myRank_ && recvCount > 0) {
            u64 recvSliceSize = recvCount * dataTypeSize_;
            u64 recvOffset = tempAlgParams.rdispls[remoteRank] * dataTypeSize_;
            DataSlice srcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
            // 将HCCL_BUF的后半部分CCL_OUT上接收到的数据，拷贝到recvBuf上
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, recvOffset, recvSliceSize, recvCount);
            HCCL_INFO("[InsTempDpuAlltoAllMesh] [PostCopyDataToRecvBuf] myRank[%u] localCopy remoteRank[%u]'s data "
                      "from CCLOUT startAddr[%llu] to recvBuf startAddr[%llu] and Size is [%llu]",
                      myRank_, remoteRank,
                      tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                      recvOffset, recvSliceSize);
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempDpuAlltoAllMesh::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = threadNum_ - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempDpuAlltoAllMesh::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = threadNum_ - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempDpuAlltoAllMesh::DPUKernelRun(const TemplateDataParams &tempAlgParams,
                                                const std::map<u32, std::vector<ChannelInfo>> &channels,
                                                const u32 myRank, const std::vector<std::vector<u32>> &subCommRanks)
{
#ifndef AICPU_COMPILE
    // 网卡通信流程
    HCCL_INFO("[InsTempDpuAlltoAllMesh] [DPUKernelRun] start");
    CHK_PRT_RET(subCommRanks.empty(), HCCL_ERROR("[InsTempDpuAlltoAllMesh] [DPUKernelRun] subCommRanks is empty"),
                HCCL_E_PARA);
    CHK_PRT_RET(subCommRanks[0].empty(), HCCL_ERROR("[InsTempDpuAlltoAllMesh] [DPUKernelRun] subCommRanks[0] is empty"),
                HCCL_E_PARA);
    const std::vector<u32> &commRanks = subCommRanks[0];
    const u32 templateRankSize = static_cast<u32>(commRanks.size());

    u64 splitParts = 2;
    u64 halfMaxTmpMemSize = tempAlgParams.buffInfo.hcclBuff.size / splitParts;
    u64 hcclbuffBlockMemSize = tempAlgParams.inputSliceStride;
    // dpu部分数据发送（三阶段批量：前同步 → 写数据 → 后同步）
    // 阶段 0：收集所有有效 rank-pair 信息
    std::vector<DpuTransferCtx> pairs;

    for (u32 i = 0; i < templateRankSize; i++) {
        u32 remoteRank = commRanks[i];
        if (remoteRank == myRank) {
            continue;
        }
        auto it = channels.find(remoteRank);
        CHK_PRT_RET(it == channels.end(),
            HCCL_ERROR("[InsTempDpuAlltoAllMesh] [DPUKernelRun] myRank[%u] key not found in channels map", remoteRank),
            HCCL_E_PARA);
        CHK_PRT_RET(channels.at(remoteRank).empty(),
            HCCL_ERROR("[InsTempDpuAlltoAllMesh] [DPUKernelRun] channelInfo Vector for myRank[%u] key is empty", remoteRank),
            HCCL_E_PARA);
        const ChannelInfo &link = channels.at(remoteRank)[0];
        if (link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
            HCCL_WARNING("[InsTempDpuAlltoAllMesh][DPUKernelRun] skip myRank[%u] transfer data to remoteRank[%u] by "
                         "DPU , the EndpointLocType must be HOST", myRank, remoteRank);
            continue;
        }
        u64 sendCount = tempAlgParams.sendCounts[remoteRank];
        u64 recvCount = tempAlgParams.recvCounts[remoteRank];
        if (sendCount == 0 && recvCount == 0) {
            continue;
        }
        u64 sendSliceSize = sendCount * SIZE_TABLE[tempAlgParams.dataType];
        u64 recvSliceSize = recvCount * SIZE_TABLE[tempAlgParams.dataType];

        DpuTransferCtx ctx;
        ctx.txCh = &link;
        ctx.rxCh = &link;  // mesh alltoall: samePeer
        if (sendCount > 0) {
            ctx.txSrcSlices.push_back(DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize,
                sendSliceSize, sendCount));
            ctx.txDstSlices.push_back(DataSlice(link.remoteCclMem.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize + myRank * hcclbuffBlockMemSize,
                sendSliceSize, sendCount));
        }
        if (recvCount > 0) {
            ctx.rxSrcSlices.push_back(DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + myRank * hcclbuffBlockMemSize,
                recvSliceSize, recvCount));
            ctx.rxDstSlices.push_back(DataSlice(tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize + remoteRank * hcclbuffBlockMemSize,
                recvSliceSize, recvCount));
        }
        pairs.push_back(ctx);
    }

    CHK_RET(DpuBatchTransfer(pairs));
    HCCL_INFO("[InsTempDpuAlltoAllMesh] [DPUKernelRun] end");
#endif
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE_V2("InsTempDpuAlltoAllMesh", InsTempDpuAlltoAllMesh);
}  // namespace ops_hccl
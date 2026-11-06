/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_dpu_alltoall_mesh_nda.h"

#include <algorithm>
#include <set>
#include "alg_v2_template_register.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "channel.h"

namespace ops_hccl {

InsTempDpuAlltoAllMeshNda::InsTempDpuAlltoAllMeshNda() {}
InsTempDpuAlltoAllMeshNda::InsTempDpuAlltoAllMeshNda(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempDpuAlltoAllMeshNda::~InsTempDpuAlltoAllMeshNda() {}

u32 InsTempDpuAlltoAllMeshNda::GetIntraRankNumFromPhysicalLevels(const TopoInfoWithNetLayerDetails* topoInfo)
{
    u32 intraRankNum = 0;
    s32 hostIdx = -1;
    for (size_t i = 0; i < topoInfo->physicalLevels.size(); i++) {
        if (topoInfo->physicalLevels[i].locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
            hostIdx = static_cast<s32>(i);
            break;
        }
    }
    if (hostIdx < 0) {
        HCCL_ERROR("[InsTempDpuAlltoAllMeshNda] no HOST level found in physicalLevels.");
        return 0;
    }
    s32 idx = hostIdx - 1;
    while (idx >= 0 && !topoInfo->physicalLevels[idx].hasTopoInst) {
        idx--;
    }
    if (idx < 0) {
        HCCL_ERROR("[InsTempDpuAlltoAllMeshNda] no level with hasTopoInst=true below HOST level.");
        return 0;
    }
    for (auto size : topoInfo->physicalLevels[idx].instSizeListByLayer) {
        intraRankNum = std::max(intraRankNum, size);
    }
    return intraRankNum;
}

HcclResult InsTempDpuAlltoAllMeshNda::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    u32 threadNum = 0;
    std::vector<HcclChannelDesc> level0Channels;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS || topoInfo->level0PcieMix
        || topoInfo->deviceNumPerModule == 1) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));

        // DEVICE 对端各使用一个 AICPU thread，HOST 对端共用主 thread 执行 NDA 通信。
        std::set<u32> deviceRemoteRanks;
        // 同一 remoteRank 可能有多条 channel，只按 rank 去重。
        for (const auto& channel : level0Channels) {
            if (channel.remoteRank == myRank_) {
                continue;
            }
            if (channel.remoteEndpoint.loc.locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
                deviceRemoteRanks.insert(channel.remoteRank);
            }
        }
        // 主 thread 只负责本地拷贝、线程同步与 HOST 对端收发，DEVICE 对端各占一条从流。
        threadNum = static_cast<u32>(deviceRemoteRanks.size()) + 1U;
    } else {
        u32 intraRankNum = GetIntraRankNumFromPhysicalLevels(topoInfo);
        // intraRankNum 含本端，主 thread 不再承担 DEVICE 对端收发，故线程数与模块内 rank 数一致。
        threadNum = (intraRankNum > 1) ? intraRankNum : 1;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(
            comm, param, topoInfo, subCommRanks_, level0Channels, CommTopo::COMM_TOPO_1DMESH));
    }

    // 计算从流以及Notify数量
    SetThreadNotifyRequest(threadNum, resourceRequest);

    resourceRequest.channels.push_back(level0Channels);
    HCCL_DEBUG(
        "[InsTempDpuAlltoAllMeshNda][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u]", myRank_,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempDpuAlltoAllMeshNda::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // hcclbuf切分为CCL_IN和CCL_OUT,每部分切分为ranksize块hcclbuffBlockMem
    u64 scratchMultiple = CCLBUF_SPLIT_PARTS * templateRankSize_;
    HCCL_INFO("[InsTempDpuAlltoAllMeshNda] scratchMultiple[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempDpuAlltoAllMeshNda::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    // HCCL_BUF均分为CCL_IN和CCL_OUT两部分
    halfMaxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size / CCLBUF_SPLIT_PARTS;
    hcclbuffBlockMemSize_ = tempAlgParams.inputSliceStride;
    threads_ = templateResource.threads;
    threadNum_ = threads_.size();
    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = HCCL_SIZE_TABLE[dataType_];
    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempDpuAlltoAllMeshNda] Rank [%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(
        subCommRanks_.empty(), HCCL_ERROR("[InsTempDpuAlltoAllMeshNda] [KernelRun] subCommRanks is empty"),
        HCCL_E_PARA);
    CHK_PRT_RET(
        subCommRanks_[0].empty(), HCCL_ERROR("[InsTempDpuAlltoAllMeshNda] [KernelRun] subCommRanks[0] is empty"),
        HCCL_E_PARA);

    CHK_RET(LocalCopyforMyRank(subCommRanks_[0], tempAlgParams, threads_));

    // 仅 HOST(ROCE) 对端需要先搬到 CCL_IN，其余对端直接从输入内存写对端 CCL buffer
    CHK_RET(PreCopyDataToCclInBuf(subCommRanks_[0], tempAlgParams, templateResource.channels, threads_));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(SendRecvData(subCommRanks_[0], tempAlgParams, templateResource));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    // 将接收到的数据从CCL_OUT拷贝recvBuf
    CHK_RET(PostCopyDataToRecvBuf(subCommRanks_[0], tempAlgParams, threads_));

    return HCCL_SUCCESS;
}

HcclResult InsTempDpuAlltoAllMeshNda::LocalCopyforMyRank(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, std::vector<ThreadHandle>& threads)
{
    (void)commRanks;
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

HcclResult InsTempDpuAlltoAllMeshNda::PreCopyDataToCclInBuf(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>>& channels, std::vector<ThreadHandle>& threads)
{
    for (u32 i = 0; i < commRanks.size(); i++) {
        u32 remoteRank = commRanks[i];
        u64 sendCount = tempAlgParams.sendCounts[remoteRank];
        if (remoteRank == myRank_ || sendCount == 0) {
            continue;
        }
        auto it = channels.find(remoteRank);
        CHK_PRT_RET(
            it == channels.end() || it->second.empty(),
            HCCL_ERROR("[InsTempDpuAlltoAllMeshNda][PreCopyDataToCclInBuf] no channel for remoteRank[%u]", remoteRank),
            HCCL_E_PARA);
        // 非 ROCE 链路直接由从流从输入内存写对端 CCL buffer，无需本端 CCL_IN 中转
        if (it->second[0].protocol != COMM_PROTOCOL_ROCE) {
            continue;
        }
        u64 sendSliceSize = sendCount * dataTypeSize_;
        u64 sendOffset = tempAlgParams.sdispls[remoteRank] * dataTypeSize_;
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, sendOffset, sendSliceSize, sendCount);
        // 将sendBuf上待发送数据，拷贝到HCCL_BUF的前半部分CCL_IN上
        DataSlice dstSlice = DataSlice(
            tempAlgParams.buffInfo.hcclBuff.addr,
            tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_, sendSliceSize, sendCount);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuAlltoAllMeshNda::SendRecvData(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    // 0 号 thread 保留给本地拷贝、线程同步与 HOST 对端，DEVICE 对端从 1 号 thread 开始分配。
    u32 threadIdx = 1;
    for (u32 remoteRank : commRanks) {
        if (remoteRank == myRank_) {
            continue;
        }

        auto it = templateResource.channels.find(remoteRank);
        CHK_PRT_RET(
            it == templateResource.channels.end() || it->second.empty(),
            HCCL_ERROR("[InsTempDpuAlltoAllMeshNda][SendRecvData] no channel for remoteRank[%u]", remoteRank),
            HCCL_E_PARA);
        const ChannelInfo& link = it->second[0];
        CHK_PRT_RET(
            link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST
                && link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE,
            HCCL_ERROR(
                "[InsTempDpuAlltoAllMeshNda][SendRecvData] unsupported locationType for remoteRank[%u]", remoteRank),
            HCCL_E_NOT_SUPPORT);

        u64 sendCount = tempAlgParams.sendCounts[remoteRank];
        u64 recvCount = tempAlgParams.recvCounts[remoteRank];
        if (sendCount == 0 && recvCount == 0) {
            continue;
        }
        void* remoteCclBuffAddr = link.remoteCclMem.addr;
        CHK_PRT_RET(
            remoteCclBuffAddr == nullptr,
            HCCL_ERROR(
                "[InsTempDpuAlltoAllMeshNda][SendRecvData] null remote CCL buffer for remoteRank[%u]", remoteRank),
            HCCL_E_INTERNAL);

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        if (sendCount > 0) {
            u64 sendSliceSize = sendCount * dataTypeSize_;
            if (link.protocol != COMM_PROTOCOL_ROCE) {
                // 非 ROCE 链路（UB 直连对端）直接从输入内存写出，省去 CCL_IN 中转
                txSrcSlices.emplace_back(
                    tempAlgParams.buffInfo.inputPtr, tempAlgParams.sdispls[remoteRank] * dataTypeSize_, sendSliceSize,
                    sendCount);
            } else {
                txSrcSlices.emplace_back(
                    tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_, sendSliceSize,
                    sendCount);
            }
            txDstSlices.emplace_back(
                remoteCclBuffAddr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_,
                sendSliceSize, sendCount);
        }
        if (recvCount > 0) {
            u64 recvSliceSize = recvCount * dataTypeSize_;
            // rxSlices 仅 read 编排（对端读到本端）使用，当前编排走 write 分支不会读取。
            // read 语义下对端已将发往本 rank 的切片暂存到自己的 CCL_IN 分块（按目的 rank 索引），
            // 本端读到本地 CCL_OUT 的 remoteRank 分块，与 write 编排的落点一致。
            rxSrcSlices.emplace_back(
                remoteCclBuffAddr, tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
            rxDstSlices.emplace_back(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
        }

        SendRecvInfo sendRecvInfo{{link, link}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
        if (link.locationType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
            // HOST 通道直接在 AICPU 主 thread 上通信；单方向时走轻量 NDA 语义，省去一次多余的同步
            if (sendCount > 0 && recvCount > 0) {
                CHK_RET(SendRecvWriteForNda(sendRecvInfo, threads_[0]));
            } else if (sendCount > 0) {
                DataInfo sendInfo{link, {txSrcSlices, txDstSlices}};
                CHK_RET(SendWriteForNda(sendInfo, threads_[0]));
            } else {
                DataInfo recvInfo{link, {rxSrcSlices, rxDstSlices}};
                CHK_RET(RecvWriteForNda(recvInfo, threads_[0]));
            }
            continue;
        }

        CHK_PRT_RET(
            threadIdx >= threadNum_,
            HCCL_ERROR(
                "[InsTempDpuAlltoAllMeshNda][SendRecvData] thread index[%u] exceeds thread count[%u]", threadIdx,
                threadNum_),
            HCCL_E_INTERNAL);
        if (sendCount > 0 && recvCount > 0) {
            CHK_RET(SendRecvWrite(sendRecvInfo, threads_[threadIdx]));
        } else if (sendCount > 0) {
            DataInfo sendDataInfo{link, {txSrcSlices, txDstSlices}};
            CHK_RET(SendWrite(sendDataInfo, threads_[threadIdx]));
        } else {
            DataInfo recvDataInfo{link, {rxSrcSlices, rxDstSlices}};
            CHK_RET(RecvWrite(recvDataInfo, threads_[threadIdx]));
        }
        threadIdx++;
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempDpuAlltoAllMeshNda::PostCopyDataToRecvBuf(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, std::vector<ThreadHandle>& threads)
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
            HCCL_INFO(
                "[InsTempDpuAlltoAllMeshNda] [PostCopyDataToRecvBuf] myRank[%u] localCopy remoteRank[%u]'s data "
                "from CCLOUT startAddr[%llu] to recvBuf startAddr[%llu] and Size is [%llu]",
                myRank_, remoteRank,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                recvOffset, recvSliceSize);
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempDpuAlltoAllMeshNda::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = threadNum_ - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempDpuAlltoAllMeshNda::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = threadNum_ - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

REGISTER_TEMPLATE_V2("InsTempDpuAlltoAllMeshNda", InsTempDpuAlltoAllMeshNda);
} // namespace ops_hccl

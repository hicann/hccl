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
#include "exec_timeout_manager.h"

namespace ops_hccl {

HcclResult InsTempDpuAlltoAllMeshNda::RunWindowChunk(
    const ChannelInfo& channel, ThreadHandle thread, HcclDataType dataType, const RunWindowChunkSlices& slices)
{
    const DataSlice& input = slices.input;
    const DataSlice& output = slices.output;
    const DataSlice& sendBuffer = slices.sendBuffer;
    const DataSlice& recvBuffer = slices.recvBuffer;
    const bool nda = channel.protocol == COMM_PROTOCOL_ROCE;
    if (nda && input.size_ != 0) {
        CHK_RET(LocalCopy(thread, input, sendBuffer));
    }
    std::vector<DataSlice> sendSources;
    std::vector<DataSlice> sendTargets;
    if (input.size_ != 0) {
        CHK_PRT_RET(
            channel.remoteCclMem.addr == nullptr,
            HCCL_ERROR("[InsTempDpuAlltoAllMeshNda][RunWindowChunk] null remote CCL buffer"), HCCL_E_INTERNAL);
        sendSources.push_back(nda ? sendBuffer : input);
        sendTargets.emplace_back(channel.remoteCclMem.addr, recvBuffer.offset_, input.size_, input.count_);
    }
    if (input.size_ != 0 && output.size_ != 0) {
        SendRecvInfo info{{channel, channel}, {{sendSources, sendTargets}, {{}, {}}}, dataType};
        CHK_RET(nda ? SendRecvWriteForNda(info, thread) : SendRecvWrite(info, thread));
    } else if (input.size_ != 0) {
        DataInfo info{channel, {sendSources, sendTargets}, dataType};
        CHK_RET(nda ? SendWriteForNda(info, thread) : SendWrite(info, thread));
    } else if (output.size_ != 0) {
        DataInfo info{channel, {{}, {}}, dataType};
        CHK_RET(nda ? RecvWriteForNda(info, thread) : RecvWrite(info, thread));
    }
    if (output.size_ != 0) {
        CHK_RET(LocalCopy(thread, recvBuffer, output));
    }
    return HCCL_SUCCESS;
}

namespace {
    HcclResult AppendCopy(
        const ThreadHandle& thread, const DataSlice& source, const DataSlice& destination, DataSlice& pendingSource,
        DataSlice& pendingDestination)
    {
        if (pendingSource.size_ != 0 && IsContinuousSlice(source, pendingSource)
            && IsContinuousSlice(destination, pendingDestination) && source.size_ <= UB_MAX_DATA_SIZE
            && pendingSource.size_ <= UB_MAX_DATA_SIZE - source.size_) {
            pendingSource.size_ += source.size_;
            pendingSource.count_ += source.count_;
            pendingDestination.size_ += destination.size_;
            pendingDestination.count_ += destination.count_;
            return HCCL_SUCCESS;
        }
        if (pendingSource.size_ != 0) {
            CHK_RET(LocalCopy(thread, pendingSource, pendingDestination));
        }
        pendingSource = source;
        pendingDestination = destination;
        return HCCL_SUCCESS;
    }
} // namespace

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

    std::vector<ThreadHandle> subThreads;
    if (threadNum_ > 1) {
        subThreads.assign(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(PreCopyDataToCclInBuf(subCommRanks_[0], tempAlgParams, templateResource.channels, threads_));

    CHK_RET(SendRecvData(param, subCommRanks_[0], tempAlgParams, templateResource));

    if (threadNum_ > 1) {
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    // 将接收到的数据从CCL_OUT拷贝recvBuf
    CHK_RET(PostCopyDataToRecvBuf(subCommRanks_[0], tempAlgParams, templateResource.channels, threads_));

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
    DataSlice pendingSource(nullptr, 0, 0, 0);
    DataSlice pendingDestination(nullptr, 0, 0, 0);
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
        CHK_RET(AppendCopy(threads[0], srcSlice, dstSlice, pendingSource, pendingDestination));
    }
    if (pendingSource.size_ != 0) {
        CHK_RET(LocalCopy(threads[0], pendingSource, pendingDestination));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuAlltoAllMeshNda::SendRecvData(
    const OpParam& param, const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource)
{
    u32 threadIdx = 1;
    const ChannelInfo emptyChannel{};
    SendRecvInfo sendRecvInfo{{emptyChannel, emptyChannel}, {{{}, {}}, {{}, {}}}, dataType_};
    auto& txSrcSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.srcSlices_;
    auto& txDstSlices = sendRecvInfo.sendRecvSlices_.txSlicesList_.dstSlices_;
    auto& rxSrcSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.srcSlices_;
    auto& rxDstSlices = sendRecvInfo.sendRecvSlices_.rxSlicesList_.dstSlices_;
    auto sendRecvPeer = [&](u32 remoteRank, const ChannelInfo& link) -> HcclResult {
        CHK_PRT_RET(
            link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST
                && link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE,
            HCCL_ERROR(
                "[InsTempDpuAlltoAllMeshNda][SendRecvData] unsupported locationType for remoteRank[%u]", remoteRank),
            HCCL_E_NOT_SUPPORT);

        u64 sendCount = tempAlgParams.sendCounts[remoteRank];
        u64 recvCount = tempAlgParams.recvCounts[remoteRank];
        if (sendCount == 0 && recvCount == 0) {
            return HCCL_SUCCESS;
        }
        void* remoteCclBuffAddr = link.remoteCclMem.addr;
        CHK_PRT_RET(
            remoteCclBuffAddr == nullptr,
            HCCL_ERROR(
                "[InsTempDpuAlltoAllMeshNda][SendRecvData] null remote CCL buffer for remoteRank[%u]", remoteRank),
            HCCL_E_INTERNAL);

        if (link.protocol == COMM_PROTOCOL_ROCE) {
            return SendRecvNda(remoteRank, link, tempAlgParams);
        }

        txSrcSlices.clear();
        txDstSlices.clear();
        rxSrcSlices.clear();
        rxDstSlices.clear();
        if (sendCount > 0) {
            u64 sendSliceSize = sendCount * dataTypeSize_;
            // 非 ROCE 链路（UB 直连对端）直接从输入内存写出，省去 CCL_IN 中转；ROCE 链路已在上面提前返回
            txSrcSlices.emplace_back(
                tempAlgParams.buffInfo.inputPtr, tempAlgParams.sdispls[remoteRank] * dataTypeSize_, sendSliceSize,
                sendCount);
            txDstSlices.emplace_back(
                remoteCclBuffAddr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_,
                sendSliceSize, sendCount);
        }
        if (recvCount > 0) {
            u64 recvSliceSize = recvCount * dataTypeSize_;
            rxSrcSlices.emplace_back(
                remoteCclBuffAddr, tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
            rxDstSlices.emplace_back(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
        }

        sendRecvInfo.sendRecvChannels_.txChannel_ = link;
        sendRecvInfo.sendRecvChannels_.rxChannel_ = link;
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
        if (recvCount > 0) {
            DataSlice outputSlice(
                tempAlgParams.buffInfo.outputPtr, tempAlgParams.rdispls[remoteRank] * dataTypeSize_,
                recvCount * dataTypeSize_, recvCount);
            CHK_RET(LocalCopy(threads_[threadIdx], rxDstSlices[0], outputSlice));
        }
        threadIdx++;
        return HCCL_SUCCESS;
    };

    std::vector<std::pair<u32, const ChannelInfo*>> ndaPeers;
    for (u32 remoteRank : commRanks) {
        if (remoteRank == myRank_) {
            continue;
        }
        auto channelIt = templateResource.channels.find(remoteRank);
        CHK_PRT_RET(
            channelIt == templateResource.channels.end() || channelIt->second.empty(),
            HCCL_ERROR("[InsTempDpuAlltoAllMeshNda][SendRecvData] no channel for remoteRank[%u]", remoteRank),
            HCCL_E_PARA);
        const ChannelInfo& link = channelIt->second[0];
        if (link.protocol == COMM_PROTOCOL_ROCE) {
            if (tempAlgParams.sendCounts[remoteRank] != 0 || tempAlgParams.recvCounts[remoteRank] != 0) {
                ndaPeers.emplace_back(remoteRank, &link);
            }
        } else {
            CHK_RET(sendRecvPeer(remoteRank, link));
        }
    }
    if (ndaPeers.empty()) {
        return HCCL_SUCCESS;
    }

    std::sort(ndaPeers.begin(), ndaPeers.end(), [this](const auto& left, const auto& right) {
        return (myRank_ ^ left.first) < (myRank_ ^ right.first);
    });

    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    HcclResult transferRet = HCCL_SUCCESS;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        transferRet = SendRecvNdaMesh(ndaPeers, tempAlgParams);
    } else {
        for (const auto& peer : ndaPeers) {
            transferRet = sendRecvPeer(peer.first, *peer.second);
            if (transferRet != HCCL_SUCCESS) {
                break;
            }
        }
    }
    HcclResult batchRet = static_cast<HcclResult>(HcommBatchModeStart(param.algTag));
    CHK_RET(transferRet);
    CHK_RET(batchRet);
    return HCCL_SUCCESS;
}

HcclResult
InsTempDpuAlltoAllMeshNda::SendRecvNda(u32 remoteRank, const ChannelInfo& link, const TemplateDataParams& tempAlgParams)
{
    const u64 sendCount = tempAlgParams.sendCounts[remoteRank];
    const u64 recvCount = tempAlgParams.recvCounts[remoteRank];
    const ThreadHandle thread = threads_[0];
    if (sendCount > 0 && recvCount == 0) {
        const u64 bytes = sendCount * dataTypeSize_;
        DataSlice source(
            tempAlgParams.buffInfo.hcclBuff.addr,
            tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_, bytes, sendCount);
        DataSlice destination(
            link.remoteCclMem.addr,
            tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_, bytes,
            sendCount);
        DataInfo sendInfo{link, {{source}, {destination}}, dataType_};
        return SendWriteForNda(sendInfo, thread);
    }
    if (sendCount == 0 && recvCount > 0) {
        DataInfo recvInfo{link, {{}, {}}, dataType_};
        return RecvWriteForNda(recvInfo, thread);
    }
    const u32 timeout = ExecTimeoutManager::Instance().GetExecTimeout();
    if (recvCount > 0) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, link.handle, NOTIFY_IDX_ACK)));
    }
    if (sendCount > 0) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, link.handle, NOTIFY_IDX_ACK, timeout)));
        const void* source = static_cast<s8*>(tempAlgParams.buffInfo.hcclBuff.addr)
                             + tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_;
        void* destination = static_cast<s8*>(link.remoteCclMem.addr) + tempAlgParams.buffInfo.hcclBuffBaseOff
                            + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_;
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
            thread, link.handle, destination, source, sendCount * dataTypeSize_, NOTIFY_IDX_DATA_SIGNAL)));
    }
    if (recvCount > 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, link.handle, NOTIFY_IDX_DATA_SIGNAL, timeout)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, link.handle, NOTIFY_IDX_FIN_ACK)));
    }
    if (sendCount > 0) {
        CHK_RET(
            static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, link.handle, NOTIFY_IDX_FIN_ACK, timeout)));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempDpuAlltoAllMeshNda::SendRecvNdaMesh(
    const std::vector<std::pair<u32, const ChannelInfo*>>& peers, const TemplateDataParams& tempAlgParams)
{
    const ThreadHandle thread = threads_[0];
    const u32 timeout = ExecTimeoutManager::Instance().GetExecTimeout();
    for (const auto& peer : peers) {
        const ChannelInfo& link = *peer.second;
        CHK_PRT_RET(
            link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST
                && link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE,
            HCCL_ERROR(
                "[InsTempDpuAlltoAllMeshNda][SendRecvData] unsupported locationType for remoteRank[%u]", peer.first),
            HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(
            link.remoteCclMem.addr == nullptr,
            HCCL_ERROR(
                "[InsTempDpuAlltoAllMeshNda][SendRecvData] null remote CCL buffer for remoteRank[%u]", peer.first),
            HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, link.handle, NOTIFY_IDX_ACK)));
    }
    for (const auto& peer : peers) {
        const ChannelInfo& link = *peer.second;
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, link.handle, NOTIFY_IDX_ACK, timeout)));
        // 纯收方向的对端无数据要写：跳过 0 长度 WriteWithNotify，单独补一个 DATA_SIGNAL 通知对端
        if (tempAlgParams.sendCounts[peer.first] == 0) {
            CHK_RET(
                static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, link.handle, NOTIFY_IDX_DATA_SIGNAL)));
            continue;
        }
        const void* source = static_cast<s8*>(tempAlgParams.buffInfo.hcclBuff.addr)
                             + tempAlgParams.buffInfo.hcclBuffBaseOff + peer.first * hcclbuffBlockMemSize_;
        void* destination = static_cast<s8*>(link.remoteCclMem.addr) + tempAlgParams.buffInfo.hcclBuffBaseOff
                            + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_;
        CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
            thread, link.handle, destination, source, tempAlgParams.sendCounts[peer.first] * dataTypeSize_,
            NOTIFY_IDX_DATA_SIGNAL)));
    }
    for (const auto& peer : peers) {
        const ChannelInfo& link = *peer.second;
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, link.handle, NOTIFY_IDX_DATA_SIGNAL, timeout)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, link.handle, NOTIFY_IDX_FIN_ACK)));
    }
    for (const auto& peer : peers) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, peer.second->handle, NOTIFY_IDX_FIN_ACK, timeout)));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempDpuAlltoAllMeshNda::PostCopyDataToRecvBuf(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>>& channels, std::vector<ThreadHandle>& threads)
{
    DataSlice pendingSource(nullptr, 0, 0, 0);
    DataSlice pendingDestination(nullptr, 0, 0, 0);
    for (u32 i = 0; i < commRanks.size(); i++) {
        u32 remoteRank = commRanks[i];
        u64 recvCount = tempAlgParams.recvCounts[remoteRank];
        if (remoteRank != myRank_ && recvCount > 0 && channels.at(remoteRank)[0].protocol == COMM_PROTOCOL_ROCE) {
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
            CHK_RET(AppendCopy(threads[0], srcSlice, dstSlice, pendingSource, pendingDestination));
        }
    }
    if (pendingSource.size_ != 0) {
        CHK_RET(LocalCopy(threads[0], pendingSource, pendingDestination));
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

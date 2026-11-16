/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_dpu_reduce_scatter_v_mesh.h"
#include "exec_timeout_manager.h"

#include <algorithm>
#include <set>

namespace ops_hccl {

InsTempDpuReduceScatterVMesh::InsTempDpuReduceScatterVMesh() {}
InsTempDpuReduceScatterVMesh::InsTempDpuReduceScatterVMesh(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempDpuReduceScatterVMesh::~InsTempDpuReduceScatterVMesh() {}

HcclResult InsTempDpuReduceScatterVMesh::GetIntraRankNumFromPhysicalLevels(
    const TopoInfoWithNetLayerDetails* topoInfo, u32& intraRankNum)
{
    intraRankNum = 0;
    s32 hostIdx = -1;
    for (size_t i = 0; i < topoInfo->physicalLevels.size(); i++) {
        if (topoInfo->physicalLevels[i].locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
            hostIdx = static_cast<s32>(i);
            break;
        }
    }
    if (hostIdx < 0) {
        HCCL_ERROR("[InsTempDpuReduceScatterVMesh] no HOST level found in physicalLevels.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    s32 idx = hostIdx - 1;
    while (idx >= 0 && !topoInfo->physicalLevels[idx].hasTopoInst) {
        idx--;
    }
    if (idx < 0) {
        HCCL_ERROR("[InsTempDpuReduceScatterVMesh] no level with hasTopoInst=true below HOST level.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    for (auto size : topoInfo->physicalLevels[idx].instSizeListByLayer) {
        intraRankNum = std::max(intraRankNum, size);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuReduceScatterVMesh::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    u32 threadNum = 0;
    std::vector<HcclChannelDesc> level0Channels;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS || topoInfo->level0PcieMix
        || topoInfo->deviceNumPerModule == 1) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));

        // DEVICE 对端使用 AICPU thread，HOST 对端由 DPU 处理。
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
        // 至少保留主 thread，threadNum 包含 threads[0]。
        threadNum = std::max(1U, static_cast<u32>(deviceRemoteRanks.size()));
    } else {
        u32 intraRankNum = 0;
        CHK_RET(GetIntraRankNumFromPhysicalLevels(topoInfo, intraRankNum));
        threadNum = (intraRankNum > 1) ? (intraRankNum - 1) : 1;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(
            comm, param, topoInfo, subCommRanks_, level0Channels, CommTopo::COMM_TOPO_1DMESH));
    }

    // 计算从流以及Notify数量
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    resourceRequest.channels.push_back(level0Channels);
    HCCL_DEBUG(
        "[InsTempDpuReduceScatterVMesh][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u]", myRank_,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempDpuReduceScatterVMesh::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // hcclbuf切分为CCL_IN和CCL_OUT,每部分切分为ranksize块hcclbuffBlockMem
    u64 scratchMultiple = CCLBUF_SPLIT_PARTS * templateRankSize_;
    HCCL_INFO("[InsTempDpuReduceScatterVMesh] scratchMultiple[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempDpuReduceScatterVMesh::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    // HCCL_BUF均分为CCL_IN和CCL_OUT两部分
    halfMaxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size / CCLBUF_SPLIT_PARTS;
    hcclbuffBlockMemSize_ = tempAlgParams.inputSliceStride;
    threads_ = templateResource.threads;
    threadNum_ = threads_.size();
    dataType_ = param.vDataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempDpuReduceScatterVMesh] Rank [%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(
        subCommRanks_.empty(), HCCL_ERROR("[InsTempDpuReduceScatterVMesh] [KernelRun] subCommRanks is empty"),
        HCCL_E_PARA);
    CHK_PRT_RET(
        subCommRanks_[0].empty(), HCCL_ERROR("[InsTempDpuReduceScatterVMesh] [KernelRun] subCommRanks[0] is empty"),
        HCCL_E_PARA);

    // 阶段2：各对端变长分片从userIn拷贝到CCL_IN各对端块（reducescatterv无本地直达段，DPU要求发送源为注册内存）
    CHK_RET(PreCopyDataToCclInBuf(subCommRanks_[0], tempAlgParams, threads_));

    if (!templateResource.isDpuThreadEnabled) {
        // 转换成eager-mode，保障AICPU指令下发执行完成
        if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempDpuReduceScatterVMesh] failed set eager mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }
        if (HcommThreadSynchronize(threads_[0]) != 0) {
            HCCL_ERROR("HcommThreadSynchronize failed");
            return HCCL_E_INTERNAL;
        }
        // 将执行模式转换回到batch
        if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempDpuReduceScatterVMesh] failed set batch mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }
    }

    // 阶段3：HOST对端走DPU、DEVICE对端走AICPU从线程，逐对端交换
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(SendRecvData(param, subCommRanks_[0], tempAlgParams, templateResource, threads_));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    // 阶段4：收齐后本地规约（本rank分片直拷 + 各来源块LocalReduce，DPU只搬数，规约在HBM由AICPU完成）
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(PostCopyAndLocalReduce(param, tempAlgParams, threads_));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempDpuReduceScatterVMesh::PreCopyDataToCclInBuf(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, std::vector<ThreadHandle>& threads)
{
    for (u32 i = 0; i < commRanks.size(); i++) {
        u32 remoteRank = commRanks[i];
        if (remoteRank == myRank_) {
            continue;
        }
        // 发给对端的变长分片（字节）；allRankDispls已由executor转为字节
        u64 sendSliceSize = tempAlgParams.allRankSliceSize[remoteRank];
        u64 sendCount = sendSliceSize / dataTypeSize_;
        // sendCounts[j]为0的对端跳过
        if (sendSliceSize == 0) {
            continue;
        }
        u64 srcOffset = tempAlgParams.buffInfo.inBuffBaseOff + tempAlgParams.allRankDispls[remoteRank];
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, sendSliceSize, sendCount);
        // 将待发送分片拷贝到HCCL_BUF的前半部分CCL_IN上
        DataSlice dstSlice = DataSlice(
            tempAlgParams.buffInfo.hcclBuff.addr,
            tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_, sendSliceSize, sendCount);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuReduceScatterVMesh::SendRecvData(
    const OpParam& param, const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource, std::vector<ThreadHandle>& threads)
{
    if (templateResource.isDpuThreadEnabled) {
        // 新路径：主流通知DPU线程开工，再由主流把搬运任务下发到DPU线程队列，最后等其完工通知
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[0], templateResource.dpuThread, HOST_DPU_THREAD_NOTIFY_IDX)));
        u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(templateResource.dpuThread, HOST_DPU_THREAD_NOTIFY_IDX, execTimeout)));
        // 照抄DPUKernelRun逻辑：把批量搬运任务下发到DPU线程
        CHK_RET(RunDpuTransfer(
            templateResource.channels, tempAlgParams, templateResource.dpuThread,
            templateResource.isHcommChannelDrainSupported));
        CHK_RET(RunAicpuTransfer(commRanks, tempAlgParams, templateResource, threads));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            templateResource.dpuThread, threads[0], templateResource.dpuThreadNotifyIdxOnMain)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[0], templateResource.dpuThreadNotifyIdxOnMain, execTimeout)));
        return HcclResult::HCCL_SUCCESS;
    }

    // 写共享内存，唤醒dpu线程
    DPURunInfo dpuRunInfo;
    dpuRunInfo.execTimeout = param.opConfig.execTimeout;
    dpuRunInfo.templateName = "InsTempDpuReduceScatterVMesh";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.subCommRanks = subCommRanks_;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    u32 sendMsgId = 0;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();
    if (HcommSendRequest(
            reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
            static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId)
        != 0) {
        HCCL_ERROR("[InsTempDpuReduceScatterVMesh] [SendRecvData] HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }
    // AICPU部分数据传输
    CHK_RET(RunAicpuTransfer(commRanks, tempAlgParams, templateResource, threads));

    // 等待dpu完成
    HCCL_INFO("[InsTempDpuReduceScatterVMesh] [SendRecvData] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);
    void* recvData = nullptr;
    u32 recvMsgId = 0;

    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempDpuReduceScatterVMesh][SendRecvData] HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempDpuReduceScatterVMesh] [SendRecvData] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR(
            "[InsTempDpuReduceScatterVMesh] [SendRecvData] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId,
            sendMsgId);
        return HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuReduceScatterVMesh::RunAicpuTransfer(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams,
    const TemplateResource& templateResource, std::vector<ThreadHandle>& threads)
{
    u32 threadIdx = 0;
    for (u32 i = 0; i < commRanks.size(); i++) {
        u32 remoteRank = commRanks[i];
        if (remoteRank == myRank_) {
            continue;
        }

        auto it = templateResource.channels.find(remoteRank);
        CHK_PRT_RET(
            it == templateResource.channels.end(),
            HCCL_ERROR(
                "[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] myRank[%u] remoteRank[%u] key not found in "
                "channels map",
                myRank_, remoteRank),
            HCCL_E_PARA);
        CHK_PRT_RET(
            templateResource.channels.at(remoteRank).empty(),
            HCCL_ERROR(
                "[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] myRank[%u] ChannelInfo Vector for remoteRank[%u] "
                "key is empty",
                myRank_, remoteRank),
            HCCL_E_PARA);
        const ChannelInfo& link = templateResource.channels.at(remoteRank)[0];

        if (link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
            HCCL_WARNING(
                "[InsTempDpuReduceScatterVMesh][RunAicpuTransfer] skip myRank[%u] transfer data to remoteRank[%u] by "
                "AICPU , the EndpointLocType must be DEVICE",
                myRank_, remoteRank);
            continue;
        }

        if (threadIdx >= threadNum_) {
            HCCL_ERROR(
                "[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] thread index [%u] exceeds thread count [%u]",
                threadIdx, threadNum_);
            return HCCL_E_INTERNAL;
        }

        // 发给对端的量为发给remote的分片，从对端收的量按变长契约恒同为本rank分片
        u64 sendSliceSize = tempAlgParams.allRankSliceSize[remoteRank];
        u64 recvSliceSize = tempAlgParams.allRankSliceSize[myRank_];
        u64 sendCount = sendSliceSize / dataTypeSize_;
        u64 recvCount = recvSliceSize / dataTypeSize_;
        // 无需发送和接收数据
        if (sendSliceSize == 0 && recvSliceSize == 0) {
            HCCL_INFO(
                "[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] myRank[%u] send data to remoteRank[%u] and "
                "myRank[%u] recv data from remoteRank[%u] are zero, skip data transfer",
                myRank_, remoteRank, myRank_, remoteRank);
            continue;
        }
        void* remoteCclBuffAddr = link.remoteCclMem.addr;
        if (remoteCclBuffAddr == nullptr) {
            HCCL_ERROR(
                "[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] myRank[%u] Remote CCL buffer address is null for "
                "remoteRank[%u]",
                myRank_, remoteRank);
            return HCCL_E_INTERNAL;
        }
        if (sendSliceSize > 0 && recvSliceSize > 0) {
            // 待发和待收数据量不为0的情况
            DataSlice sendSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_, sendSliceSize, sendCount);
            DataSlice sendDstSlice = DataSlice(
                remoteCclBuffAddr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_,
                sendSliceSize, sendCount);

            DataSlice recvSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_, recvSliceSize, recvCount);
            DataSlice recvDstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
            std::vector<DataSlice> txSrcSlices{sendSrcSlice};
            std::vector<DataSlice> txDstSlices{sendDstSlice};
            std::vector<DataSlice> rxSrcSlices{recvSrcSlice};
            std::vector<DataSlice> rxDstSlices{recvDstSlice};
            HCCL_INFO(
                "[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] myRank[%u] send data to remoteRank[%u], myRank's "
                "CCLIN startAddr[%llu] to remoteRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_,
                sendSliceSize);
            HCCL_INFO(
                "[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] myRank[%u] recv data from remoteRank[%u], "
                "remoteRank's CCLIN startAddr[%llu] to myRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize);
            SendRecvInfo sendRecvInfo{
                {link, link}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, tempAlgParams.dataType};
            CHK_PRT_RET(
                SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                HCCL_ERROR("[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] ReduceScatterV AICPU SendRecv failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else if (sendSliceSize > 0) {
            // 待发数据量不为0,待收数据为0的情况
            DataSlice sendSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_, sendSliceSize, sendCount);
            DataSlice sendDstSlice = DataSlice(
                remoteCclBuffAddr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_,
                sendSliceSize, sendCount);
            std::vector<DataSlice> txSrcSlices{sendSrcSlice};
            std::vector<DataSlice> txDstSlices{sendDstSlice};
            HCCL_INFO(
                "[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] myRank[%u] send data to remoteRank[%u], myRank's "
                "CCLIN startAddr[%llu] to remoteRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize_,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + myRank_ * hcclbuffBlockMemSize_,
                sendSliceSize);
            DataInfo sendDataInfo{link, {txSrcSlices, txDstSlices}, tempAlgParams.dataType};
            CHK_PRT_RET(
                SendWrite(sendDataInfo, threads[threadIdx]),
                HCCL_ERROR("[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] ReduceScatterV AICPU only Send failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            // 待收数据量不为0,待发数据为0的情况
            DataSlice recvSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_, recvSliceSize, recvCount);
            DataSlice recvDstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
            std::vector<DataSlice> rxSrcSlices{recvSrcSlice};
            std::vector<DataSlice> rxDstSlices{recvDstSlice};
            HCCL_INFO(
                "[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] myRank[%u] recv data from remoteRank[%u], "
                "remoteRank's CCLIN startAddr[%llu] to myRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize);
            DataInfo recvDataInfo{link, {rxSrcSlices, rxDstSlices}, tempAlgParams.dataType};
            CHK_PRT_RET(
                RecvWrite(recvDataInfo, threads[threadIdx]),
                HCCL_ERROR("[InsTempDpuReduceScatterVMesh] [RunAicpuTransfer] ReduceScatterV AICPU only Recv failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
        threadIdx++;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuReduceScatterVMesh::RunDpuTransfer(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const TemplateDataParams& tempAlgParams,
    ThreadHandle thread, bool isDrainSupported)
{
    // 网卡通信流程，HOST对端批量搬运
    HCCL_INFO("[InsTempDpuReduceScatterVMesh] [RunDpuTransfer] start");
    const std::vector<u32>& commRanks = subCommRanks_[0];
    const u32 templateRankSize = static_cast<u32>(commRanks.size());

    u64 halfMaxTmpMemSize = tempAlgParams.buffInfo.hcclBuff.size / CCLBUF_SPLIT_PARTS;
    u64 hcclbuffBlockMemSize = tempAlgParams.inputSliceStride;
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    // 阶段0：收集所有有效rank-pair信息
    std::vector<DpuTransferCtx> pairs;

    for (u32 i = 0; i < templateRankSize; i++) {
        u32 remoteRank = commRanks[i];
        if (remoteRank == myRank_) {
            continue;
        }
        auto it = channels.find(remoteRank);
        CHK_PRT_RET(
            it == channels.end(),
            HCCL_ERROR(
                "[InsTempDpuReduceScatterVMesh] [RunDpuTransfer] myRank[%u] remoteRank[%u] key not found in channels "
                "map",
                myRank_, remoteRank),
            HCCL_E_PARA);
        CHK_PRT_RET(
            channels.at(remoteRank).empty(),
            HCCL_ERROR(
                "[InsTempDpuReduceScatterVMesh] [RunDpuTransfer] myRank[%u] ChannelInfo Vector for remoteRank[%u] "
                "key is empty",
                myRank_, remoteRank),
            HCCL_E_PARA);
        const ChannelInfo& link = channels.at(remoteRank)[0];
        if (link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
            HCCL_WARNING(
                "[InsTempDpuReduceScatterVMesh][RunDpuTransfer] skip myRank[%u] transfer data to remoteRank[%u] by "
                "DPU , the EndpointLocType must be HOST",
                myRank_, remoteRank);
            continue;
        }
        u64 sendSliceSize = tempAlgParams.allRankSliceSize[remoteRank];
        u64 recvSliceSize = tempAlgParams.allRankSliceSize[myRank_];
        u64 sendCount = sendSliceSize / dataTypeSize;
        u64 recvCount = recvSliceSize / dataTypeSize;
        if (sendSliceSize == 0 && recvSliceSize == 0) {
            continue;
        }

        DpuTransferCtx ctx;
        ctx.txCh = &link;
        ctx.rxCh = &link; // mesh reducescatterv: samePeer
        if (sendSliceSize > 0) {
            void* remoteCclBuffAddr = link.remoteCclMem.addr;
            if (remoteCclBuffAddr == nullptr) {
                HCCL_ERROR(
                    "[InsTempDpuReduceScatterVMesh] [RunDpuTransfer] myRank[%u] Remote CCL buffer address is null "
                    "for remoteRank[%u]",
                    myRank_, remoteRank);
                return HCCL_E_INTERNAL;
            }
            ctx.txSrcSlices.push_back(DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + remoteRank * hcclbuffBlockMemSize, sendSliceSize, sendCount));
            ctx.txDstSlices.push_back(DataSlice(
                remoteCclBuffAddr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize + myRank_ * hcclbuffBlockMemSize,
                sendSliceSize, sendCount));
        }
        if (recvSliceSize > 0) {
            // rx切片仅作计数（决定批量接口前/后同步次数），不作为读写地址
            ctx.rxSrcSlices.push_back(DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize, recvSliceSize, recvCount));
            ctx.rxDstSlices.push_back(DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize + remoteRank * hcclbuffBlockMemSize,
                recvSliceSize, recvCount));
        }
        HCCL_INFO(
            "[InsTempDpuReduceScatterVMesh][RunDpuTransfer] myRank[%u], remoteRank[%u], sendSliceSize[%llu], "
            "recvSliceSize[%llu], channel[0x%llx], txSliceNum[%zu], rxSliceNum[%zu].",
            myRank_, remoteRank, sendSliceSize, recvSliceSize, link.handle, ctx.txSrcSlices.size(),
            ctx.rxSrcSlices.size());
        pairs.push_back(ctx);
    }

    CHK_RET(DpuBatchTransfer(pairs, thread, isDrainSupported));
    HCCL_INFO("[InsTempDpuReduceScatterVMesh] [RunDpuTransfer] end");
    return HCCL_SUCCESS;
}

HcclResult InsTempDpuReduceScatterVMesh::PostCopyAndLocalReduce(
    const OpParam& param, const TemplateDataParams& tempAlgParams, std::vector<ThreadHandle>& threads)
{
    // 本rank分片（字节）：recvCount==0时本rank不参与
    u64 mySliceSize = tempAlgParams.allRankSliceSize[myRank_];
    if (mySliceSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    u64 myCount = mySliceSize / dataTypeSize_;

    // 通信结束后，先把本rank的自留分片从userIn搬运到userOut，然后再在userOut上做规约
    DataSlice mySrcSlice = DataSlice(
        tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff + tempAlgParams.allRankDispls[myRank_],
        mySliceSize, myCount);
    DataSlice myDstSlice
        = DataSlice(tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, mySliceSize, myCount);
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], mySrcSlice, myDstSlice)));

    // INT64/PROD归约走同步分支，保障直拷在规约前完成
    if (dataType_ == HCCL_DATA_TYPE_INT64 || reduceOp_ == HcclReduceOp::HCCL_REDUCE_PROD) {
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto& thread : threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }
    }

    // 把其他来源块从CCL_OUT累加规约到userOut
    for (u32 i = 0; i < subCommRanks_[0].size(); i++) {
        u32 remoteRank = subCommRanks_[0][i];
        if (remoteRank == myRank_) {
            continue;
        }
        DataSlice srcSlice = DataSlice(
            tempAlgParams.buffInfo.hcclBuff.addr,
            tempAlgParams.buffInfo.hcclBuffBaseOff + halfMaxTmpMemSize_ + remoteRank * hcclbuffBlockMemSize_,
            mySliceSize, myCount);
        DataSlice dstSlice
            = DataSlice(tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, mySliceSize, myCount);
        CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_)));
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempDpuReduceScatterVMesh::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = threadNum_ - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempDpuReduceScatterVMesh::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = threadNum_ - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempDpuReduceScatterVMesh::DPUKernelRun(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<u32>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    myRank_ = myRank;
    subCommRanks_ = subCommRanks;
    CHK_PRT_RET(
        subCommRanks_.empty(), HCCL_ERROR("[InsTempDpuReduceScatterVMesh] [DPUKernelRun] subCommRanks is empty"),
        HCCL_E_PARA);
    CHK_PRT_RET(
        subCommRanks_[0].empty(), HCCL_ERROR("[InsTempDpuReduceScatterVMesh] [DPUKernelRun] subCommRanks[0] is empty"),
        HCCL_E_PARA);
    CHK_RET(RunDpuTransfer(channels, tempAlgParams));
#endif
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE_V2("InsTempDpuReduceScatterVMesh", InsTempDpuReduceScatterVMesh);
} // namespace ops_hccl

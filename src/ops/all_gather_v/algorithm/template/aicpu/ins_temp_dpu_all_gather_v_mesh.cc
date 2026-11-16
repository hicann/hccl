/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_dpu_all_gather_v_mesh.h"
#include "exec_timeout_manager.h"

#include <algorithm>
#include <set>

namespace ops_hccl {

InsTempDpuAllGatherVMesh::InsTempDpuAllGatherVMesh() {}
InsTempDpuAllGatherVMesh::InsTempDpuAllGatherVMesh(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempDpuAllGatherVMesh::~InsTempDpuAllGatherVMesh() {}

HcclResult InsTempDpuAllGatherVMesh::GetIntraRankNumFromPhysicalLevels(
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
        HCCL_ERROR("[InsTempDpuAllGatherVMesh] no HOST level found in physicalLevels.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    s32 idx = hostIdx - 1;
    while (idx >= 0 && !topoInfo->physicalLevels[idx].hasTopoInst) {
        idx--;
    }
    if (idx < 0) {
        HCCL_ERROR("[InsTempDpuAllGatherVMesh] no level with hasTopoInst=true below HOST level.");
        return HcclResult::HCCL_E_INTERNAL;
    }
    for (auto size : topoInfo->physicalLevels[idx].instSizeListByLayer) {
        intraRankNum = std::max(intraRankNum, size);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuAllGatherVMesh::CalcRes(
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
        "[InsTempDpuAllGatherVMesh][CalcRes] myRank[%u], notifyNumOnMainThread[%u], slaveThreadNum[%u]", myRank_,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

u64 InsTempDpuAllGatherVMesh::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // CCL布局：1块发送暂存块 + rankSize块接收块。AGV广播内容恒同，发送侧只暂存1份、
    // 全部对端的tx切片共享该暂存块，相比逐对端分块（2*rankSize）单轮容量提升近一倍，
    // 且每轮省去rankSize-2次冗余本地拷贝
    u64 scratchMultiple = templateRankSize_ + 1;
    HCCL_INFO("[InsTempDpuAllGatherVMesh] scratchMultiple[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempDpuAllGatherVMesh::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    // CCL布局：CCL_IN发送暂存块[0, block) + CCL_OUT接收区[block, block + rankSize*block)，按来源rank索引
    hcclbuffBlockMemSize_ = tempAlgParams.inputSliceStride;
    cclOutBaseOff_ = hcclbuffBlockMemSize_;
    threads_ = templateResource.threads;
    threadNum_ = threads_.size();
    dataType_ = param.vDataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempDpuAllGatherVMesh] Rank [%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_PRT_RET(
        subCommRanks_.empty(), HCCL_ERROR("[InsTempDpuAllGatherVMesh] [KernelRun] subCommRanks is empty"), HCCL_E_PARA);
    CHK_PRT_RET(
        subCommRanks_[0].empty(), HCCL_ERROR("[InsTempDpuAllGatherVMesh] [KernelRun] subCommRanks[0] is empty"),
        HCCL_E_PARA);
    CHK_PRT_RET(
        tempAlgParams.buffInfo.inputPtr == tempAlgParams.buffInfo.outputPtr,
        HCCL_ERROR("[InsTempDpuAllGatherVMesh] [KernelRun] AllGatherV does not support in-place operation."),
        HCCL_E_PARA);

    // 阶段1：本rank自留块直拷 sendBuf -> recvBuf 本rank槽位
    CHK_RET(LocalCopyForMyRank(subCommRanks_[0], tempAlgParams, threads_));

    // 阶段2：本rank块拷贝到CCL_IN暂存块（allgatherv广播内容恒同，只暂存1份，全部对端tx共享）
    CHK_RET(PreCopyDataToCclInBuf(subCommRanks_[0], tempAlgParams, threads_));

    if (!templateResource.isDpuThreadEnabled) {
        // 转换成eager-mode，保障AICPU指令下发执行完成
        if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempDpuAllGatherVMesh] failed set eager mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }
        if (HcommThreadSynchronize(threads_[0]) != 0) {
            HCCL_ERROR("HcommThreadSynchronize failed");
            return HCCL_E_INTERNAL;
        }
        // 将执行模式转换回到batch
        if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("[InsTempDpuAllGatherVMesh] failed set batch mode, tag is %s.", param.algTag);
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

    // 阶段4：CCL_OUT各对端块 -> recvBuf变长落位
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(PostCopyDataToRecvBuf(subCommRanks_[0], tempAlgParams, threads_));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }

    return HCCL_SUCCESS;
}

HcclResult InsTempDpuAllGatherVMesh::LocalCopyForMyRank(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, std::vector<ThreadHandle>& threads)
{
    (void)commRanks;
    u64 sendSliceSize = tempAlgParams.allRankSliceSize[myRank_];
    u64 sendCount = sendSliceSize / dataTypeSize_;
    // 本rank变长块为0，无自留段
    if (sendSliceSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    u64 srcOffset = tempAlgParams.buffInfo.inBuffBaseOff;
    // 本rank块按recvDispls[myRank]落位到recvBuf；allRankDispls为count单位（executor未转字节），此处×dataTypeSize_
    u64 dstOffset = tempAlgParams.allRankDispls[myRank_] * dataTypeSize_ + tempAlgParams.buffInfo.outBuffBaseOff;
    DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, sendSliceSize, sendCount);
    DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, sendSliceSize, sendCount);
    // AGV不支持in-place(入口参数检查已显式拒绝同址)，本rank自留块恒需拷贝
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuAllGatherVMesh::PreCopyDataToCclInBuf(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, std::vector<ThreadHandle>& threads)
{
    u64 sendSliceSize = tempAlgParams.allRankSliceSize[myRank_];
    u64 sendCount = sendSliceSize / dataTypeSize_;
    // 广播量为0时不向任何对端发送
    if (sendSliceSize == 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    // 无对端时无需暂存（单rank场景）
    bool hasRemoteRank = false;
    for (u32 i = 0; i < commRanks.size(); i++) {
        if (commRanks[i] != myRank_) {
            hasRemoteRank = true;
            break;
        }
    }
    if (!hasRemoteRank) {
        return HcclResult::HCCL_SUCCESS;
    }
    // 广播内容恒同：只暂存1份到CCL_IN暂存块，阶段3全部对端的tx切片共享该源块。
    // 暂存块唯一读者是本rank自己的tx引擎（rx侧为纯notify同步、不读地址），均在本轮同步窗口内完成，
    // 下一轮重写无跨轮竞态
    DataSlice srcSlice
        = DataSlice(tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, sendSliceSize, sendCount);
    DataSlice dstSlice = DataSlice(
        tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, sendSliceSize, sendCount);
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuAllGatherVMesh::SendRecvData(
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
    dpuRunInfo.templateName = "InsTempDpuAllGatherVMesh";
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
        HCCL_ERROR("[InsTempDpuAllGatherVMesh] [SendRecvData] HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }
    // AICPU部分数据传输
    CHK_RET(RunAicpuTransfer(commRanks, tempAlgParams, templateResource, threads));

    // 等待dpu完成
    HCCL_INFO("[InsTempDpuAllGatherVMesh] [SendRecvData] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);
    void* recvData = nullptr;
    u32 recvMsgId = 0;

    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempDpuAllGatherVMesh][SendRecvData] HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[InsTempDpuAllGatherVMesh] [SendRecvData] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR(
            "[InsTempDpuAllGatherVMesh] [SendRecvData] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuAllGatherVMesh::RunAicpuTransfer(
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
                "[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] myRank[%u] remoteRank[%u] key not found in channels map",
                myRank_, remoteRank),
            HCCL_E_PARA);
        CHK_PRT_RET(
            templateResource.channels.at(remoteRank).empty(),
            HCCL_ERROR(
                "[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] myRank[%u] ChannelInfo Vector for remoteRank[%u] key "
                "is empty",
                myRank_, remoteRank),
            HCCL_E_PARA);
        const ChannelInfo& link = templateResource.channels.at(remoteRank)[0];

        if (link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
            HCCL_WARNING(
                "[InsTempDpuAllGatherVMesh][RunAicpuTransfer] skip myRank[%u] transfer data to remoteRank[%u] by "
                "AICPU , the EndpointLocType must be DEVICE",
                myRank_, remoteRank);
            continue;
        }

        if (threadIdx >= threadNum_) {
            HCCL_ERROR(
                "[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] thread index [%u] exceeds thread count [%u]", threadIdx,
                threadNum_);
            return HCCL_E_INTERNAL;
        }

        // allgatherv本rank广播块大小恒同，发给对端的量为本rank块，从对端收的量为对端块
        u64 sendSliceSize = tempAlgParams.allRankSliceSize[myRank_];
        u64 recvSliceSize = tempAlgParams.allRankSliceSize[remoteRank];
        u64 sendCount = sendSliceSize / dataTypeSize_;
        u64 recvCount = recvSliceSize / dataTypeSize_;
        // 无需发送和接收数据
        if (sendSliceSize == 0 && recvSliceSize == 0) {
            HCCL_INFO(
                "[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] myRank[%u] send data to remoteRank[%u] and myRank[%u] "
                "recv data from remoteRank[%u] are zero, skip data transfer",
                myRank_, remoteRank, myRank_, remoteRank);
            continue;
        }
        void* remoteCclBuffAddr = link.remoteCclMem.addr;
        if (remoteCclBuffAddr == nullptr) {
            HCCL_ERROR(
                "[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] myRank[%u] Remote CCL buffer address is null for "
                "remoteRank[%u]",
                myRank_, remoteRank);
            return HCCL_E_INTERNAL;
        }
        if (sendSliceSize > 0 && recvSliceSize > 0) {
            // 待发和待收数据量不为0的情况；send源为共享CCL_IN暂存块（广播内容恒同，全部对端共用）
            DataSlice sendSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, sendSliceSize, sendCount);
            DataSlice sendDstSlice = DataSlice(
                remoteCclBuffAddr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff_ + myRank_ * hcclbuffBlockMemSize_,
                sendSliceSize, sendCount);

            // rx切片不作为读写地址（SendRecvWrite只消费tx切片，rx侧为notify握手），按共享暂存块布局填对端约定地址
            DataSlice recvSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, recvSliceSize, recvCount);
            DataSlice recvDstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
            std::vector<DataSlice> txSrcSlices{sendSrcSlice};
            std::vector<DataSlice> txDstSlices{sendDstSlice};
            std::vector<DataSlice> rxSrcSlices{recvSrcSlice};
            std::vector<DataSlice> rxDstSlices{recvDstSlice};
            HCCL_INFO(
                "[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] myRank[%u] send data to remoteRank[%u], myRank's "
                "CCLIN startAddr[%llu] to remoteRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff_ + myRank_ * hcclbuffBlockMemSize_,
                sendSliceSize);
            HCCL_INFO(
                "[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] myRank[%u] recv data from remoteRank[%u], remoteRank's "
                "CCLIN startAddr[%llu] to myRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize);
            SendRecvInfo sendRecvInfo{
                {link, link}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, tempAlgParams.dataType};
            CHK_PRT_RET(
                SendRecvWrite(sendRecvInfo, threads[threadIdx]),
                HCCL_ERROR("[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] AllgatherV AICPU SendRecv failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else if (sendSliceSize > 0) {
            // 待发数据量不为0,待收数据为0的情况
            DataSlice sendSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, sendSliceSize, sendCount);
            DataSlice sendDstSlice = DataSlice(
                remoteCclBuffAddr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff_ + myRank_ * hcclbuffBlockMemSize_,
                sendSliceSize, sendCount);
            std::vector<DataSlice> txSrcSlices{sendSrcSlice};
            std::vector<DataSlice> txDstSlices{sendDstSlice};
            HCCL_INFO(
                "[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] myRank[%u] send data to remoteRank[%u], myRank's "
                "CCLIN startAddr[%llu] to remoteRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff_ + myRank_ * hcclbuffBlockMemSize_,
                sendSliceSize);
            DataInfo sendDataInfo{link, {txSrcSlices, txDstSlices}, tempAlgParams.dataType};
            CHK_PRT_RET(
                SendWrite(sendDataInfo, threads[threadIdx]),
                HCCL_ERROR("[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] AllgatherV AICPU only Send failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            // 待收数据量不为0,待发数据为0的情况；rx切片不作为读写地址，仅参与notify握手
            DataSlice recvSrcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, recvSliceSize, recvCount);
            DataSlice recvDstSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
            std::vector<DataSlice> rxSrcSlices{recvSrcSlice};
            std::vector<DataSlice> rxDstSlices{recvDstSlice};
            HCCL_INFO(
                "[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] myRank[%u] recv data from remoteRank[%u], remoteRank's "
                "CCLIN startAddr[%llu] to myRank's CCLOUT startAddr[%llu] and Size is [%llu]",
                myRank_, remoteRank, tempAlgParams.buffInfo.hcclBuffBaseOff,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize);
            DataInfo recvDataInfo{link, {rxSrcSlices, rxDstSlices}, tempAlgParams.dataType};
            CHK_PRT_RET(
                RecvWrite(recvDataInfo, threads[threadIdx]),
                HCCL_ERROR("[InsTempDpuAllGatherVMesh] [RunAicpuTransfer] AllgatherV AICPU only Recv failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
        threadIdx++;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempDpuAllGatherVMesh::RunDpuTransfer(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const TemplateDataParams& tempAlgParams,
    ThreadHandle thread, bool isDrainSupported)
{
    // 网卡通信流程，HOST对端批量搬运
    HCCL_INFO("[InsTempDpuAllGatherVMesh] [RunDpuTransfer] start");
    const std::vector<u32>& commRanks = subCommRanks_[0];
    const u32 templateRankSize = static_cast<u32>(commRanks.size());

    // CCL布局与AICPU侧一致：CCL_IN暂存块[0, block) + CCL_OUT接收区[block, block + rankSize*block)
    u64 hcclbuffBlockMemSize = tempAlgParams.inputSliceStride;
    u64 cclOutBaseOff = hcclbuffBlockMemSize;
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
                "[InsTempDpuAllGatherVMesh] [RunDpuTransfer] myRank[%u] remoteRank[%u] key not found in channels map",
                myRank_, remoteRank),
            HCCL_E_PARA);
        CHK_PRT_RET(
            channels.at(remoteRank).empty(),
            HCCL_ERROR(
                "[InsTempDpuAllGatherVMesh] [RunDpuTransfer] myRank[%u] ChannelInfo Vector for remoteRank[%u] key is "
                "empty",
                myRank_, remoteRank),
            HCCL_E_PARA);
        const ChannelInfo& link = channels.at(remoteRank)[0];
        if (link.locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
            HCCL_WARNING(
                "[InsTempDpuAllGatherVMesh][RunDpuTransfer] skip myRank[%u] transfer data to remoteRank[%u] by "
                "DPU , the EndpointLocType must be HOST",
                myRank_, remoteRank);
            continue;
        }
        u64 sendSliceSize = tempAlgParams.allRankSliceSize[myRank_];
        u64 recvSliceSize = tempAlgParams.allRankSliceSize[remoteRank];
        u64 sendCount = sendSliceSize / dataTypeSize;
        u64 recvCount = recvSliceSize / dataTypeSize;
        if (sendSliceSize == 0 && recvSliceSize == 0) {
            continue;
        }

        DpuTransferCtx ctx;
        ctx.txCh = &link;
        ctx.rxCh = &link; // mesh allgatherv: samePeer
        if (sendSliceSize > 0) {
            void* remoteCclBuffAddr = link.remoteCclMem.addr;
            if (remoteCclBuffAddr == nullptr) {
                HCCL_ERROR(
                    "[InsTempDpuAllGatherVMesh] [RunDpuTransfer] myRank[%u] Remote CCL buffer address is null for "
                    "remoteRank[%u]",
                    myRank_, remoteRank);
                return HCCL_E_INTERNAL;
            }
            // send源为共享CCL_IN暂存块（广播内容恒同，全部对端tx共用同一源块）
            ctx.txSrcSlices.push_back(DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, sendSliceSize,
                sendCount));
            ctx.txDstSlices.push_back(DataSlice(
                remoteCclBuffAddr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff + myRank_ * hcclbuffBlockMemSize, sendSliceSize,
                sendCount));
        }
        if (recvSliceSize > 0) {
            // rx切片仅作计数（决定批量接口前/后同步次数），不作为读写地址
            ctx.rxSrcSlices.push_back(DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, recvSliceSize,
                recvCount));
            ctx.rxDstSlices.push_back(DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff + remoteRank * hcclbuffBlockMemSize,
                recvSliceSize, recvCount));
        }
        HCCL_INFO(
            "[InsTempDpuAllGatherVMesh][RunDpuTransfer] myRank[%u], remoteRank[%u], sendSliceSize[%llu], "
            "recvSliceSize[%llu], channel[0x%llx], txSliceNum[%zu], rxSliceNum[%zu].",
            myRank_, remoteRank, sendSliceSize, recvSliceSize, link.handle, ctx.txSrcSlices.size(),
            ctx.rxSrcSlices.size());
        pairs.push_back(ctx);
    }

    CHK_RET(DpuBatchTransfer(pairs, thread, isDrainSupported));
    HCCL_INFO("[InsTempDpuAllGatherVMesh] [RunDpuTransfer] end");
    return HCCL_SUCCESS;
}

HcclResult InsTempDpuAllGatherVMesh::PostCopyDataToRecvBuf(
    const std::vector<u32>& commRanks, const TemplateDataParams& tempAlgParams, std::vector<ThreadHandle>& threads)
{
    for (u32 i = 0; i < commRanks.size(); i++) {
        u32 remoteRank = commRanks[i];
        u64 recvSliceSize = tempAlgParams.allRankSliceSize[remoteRank];
        if (remoteRank != myRank_ && recvSliceSize > 0) {
            u64 recvCount = recvSliceSize / dataTypeSize_;
            // 按recvDispls变长落位；对端本轮chunk在本rank输出的基址取对端自身进度（per-rank），
            // 与AICPU版PostLocalCopy口径一致，不能用本rank进度的outBuffBaseOff（counts不均且多轮时会落错位）
            // allRankDispls为count单位（executor未转字节），此处×dataTypeSize_
            u64 recvOffset = tempAlgParams.allRankDispls[remoteRank] * dataTypeSize_
                             + tempAlgParams.allRankProcessedDataCount[remoteRank] * dataTypeSize_;
            DataSlice srcSlice = DataSlice(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff_ + remoteRank * hcclbuffBlockMemSize_,
                recvSliceSize, recvCount);
            // 将CCL_OUT接收区上收到的对端数据，拷贝到recvBuf上
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, recvOffset, recvSliceSize, recvCount);
            HCCL_INFO(
                "[InsTempDpuAllGatherVMesh] [PostCopyDataToRecvBuf] myRank[%u] localCopy remoteRank[%u]'s data "
                "from CCLOUT startAddr[%llu] to recvBuf startAddr[%llu] and Size is [%llu]",
                myRank_, remoteRank,
                tempAlgParams.buffInfo.hcclBuffBaseOff + cclOutBaseOff_ + remoteRank * hcclbuffBlockMemSize_,
                recvOffset, recvSliceSize);
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempDpuAllGatherVMesh::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = threadNum_ - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempDpuAllGatherVMesh::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = threadNum_ - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult InsTempDpuAllGatherVMesh::DPUKernelRun(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<u32>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    myRank_ = myRank;
    subCommRanks_ = subCommRanks;
    CHK_PRT_RET(
        subCommRanks_.empty(), HCCL_ERROR("[InsTempDpuAllGatherVMesh] [DPUKernelRun] subCommRanks is empty"),
        HCCL_E_PARA);
    CHK_PRT_RET(
        subCommRanks_[0].empty(), HCCL_ERROR("[InsTempDpuAllGatherVMesh] [DPUKernelRun] subCommRanks[0] is empty"),
        HCCL_E_PARA);
    CHK_RET(RunDpuTransfer(channels, tempAlgParams));
#endif
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE_V2("InsTempDpuAllGatherVMesh", InsTempDpuAllGatherVMesh);
} // namespace ops_hccl

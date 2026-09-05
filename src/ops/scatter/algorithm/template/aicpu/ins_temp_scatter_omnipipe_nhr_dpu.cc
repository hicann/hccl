/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_omnipipe_nhr_dpu.h"

namespace ops_hccl {
InsTempScatterOmniPipeNHRDpu::InsTempScatterOmniPipeNHRDpu() {}

InsTempScatterOmniPipeNHRDpu::InsTempScatterOmniPipeNHRDpu(
    const OpParam& param,
    const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempScatterOmniPipeNHRDpu::~InsTempScatterOmniPipeNHRDpu() {}

void InsTempScatterOmniPipeNHRDpu::SetRoot(u32 root)
{
    HCCL_DEBUG("[InsTempScatterOmniPipeNHRDpu][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root);
    root_ = root;
}

HcclResult InsTempScatterOmniPipeNHRDpu::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    // host网卡资源，不新增从流和对应Notify，只申请DPU上面
    resourceRequest.slaveThreadNum = 0; // 主thread可以通过接口传入的stream来做转换
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_DEBUG(
        "[InsTempScatterOmniPipeNHRDpu][CalcRes]slaveThreadNum[%u] notifyNumPerThread[%u] notifyNumOnMainThread[%u]"
        " level0Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread, resourceRequest.notifyNumOnMainThread,
        level0Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempScatterOmniPipeNHRDpu::GetThreadNum() const { return 1; }

HcclResult InsTempScatterOmniPipeNHRDpu::GetRes(AlgResourceRequest& resourceRequest) const
{
    // host网卡资源，不新增从流和对应notify，只申请DPU上面
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

// 语义改为返回当前template的类型，mesh返回1，nhr返回0
u64 InsTempScatterOmniPipeNHRDpu::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) { return 1; }

void InsTempScatterOmniPipeNHRDpu::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) { return; }

void InsTempScatterOmniPipeNHRDpu::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) { return; }

void InsTempScatterOmniPipeNHRDpu::SetDoTask(bool doTask)
{
    HCCL_DEBUG("[InsTempScatterOmniPipeNHRDpu][SetDoTask] myRank_ [%u], set doTask_ [%u] ", myRank_, doTask);
    doTask_.store(doTask, std::memory_order_relaxed);
}

HcclResult InsTempScatterOmniPipeNHRDpu::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    if (templateRankSize_ == 1) {
        HCCL_DEBUG("templateRankSize_ ==1");
        return HcclResult::HCCL_SUCCESS;
    }
    if (!doTask_.load(std::memory_order_relaxed)) {
        HCCL_DEBUG("[InsTempScatterOmniPipeNHRDpu] Rank [%d], doTask_ is false, skip KernelRun.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    threadNum_ = templateResource.threads.size();
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;

    HCCL_DEBUG("[%s]Run Start, threadNum_=%u, count_=%llu, dataType_=%u", __func__, threadNum_, count_, dataType_);

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempScatterOmniPipeNHRDpu] Rank [%d], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // executor 已提前调用 PreLocalCopy 时跳过，避免重复提交 DMA
    if (!preLocalCopyDone_.load(std::memory_order_relaxed)) {
        CHK_RET(PreLocalCopy(tempAlgParams, templateResource.threads));
    }
    preLocalCopyDone_ = false; // 复位，供下一 step 使用

    CHK_RET(RunDpuDataExchange(param, tempAlgParams, templateResource));

    HCCL_DEBUG("[%s]Run End", __func__);
    return HcclResult::HCCL_SUCCESS;
}

// DPU数据交换：eager模式切换、DPURunInfo序列化、SendRequest/WaitResponse、msgId校验
HcclResult InsTempScatterOmniPipeNHRDpu::RunDpuDataExchange(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    CHK_RET(PrepareDpuExecution(param, templateResource.threads[0]));
    u32 sendMsgId = 0;
    CHK_RET(SendDpuRequest(param, tempAlgParams, templateResource, sendMsgId));
    return WaitDpuResponse(param, templateResource, sendMsgId);
}

HcclResult InsTempScatterOmniPipeNHRDpu::PrepareDpuExecution(const OpParam& param, const ThreadHandle& thread)
{
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempScatterOmniPipeNHRDpu] failed set eager mode, tag is %s.", param.algTag);
        HcommBatchModeStart(param.algTag);
        return HCCL_E_INTERNAL;
    }
    if (HcommThreadSynchronize(thread) != 0) {
        HCCL_ERROR("[InsTempScatterOmniPipeNHRDpu] HcommThreadSynchronize failed");
        HcommBatchModeStart(param.algTag);
        return HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeNHRDpu::SendDpuRequest(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource,
    u32& sendMsgId)
{
    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempScatterOmniPipeNHRDpu";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    dpuRunInfo.subCommRanks = subCommRanks_;
    dpuRunInfo.tempAlgParams.root = root_;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();
    if (HcommSendRequest(
            reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
            static_cast<void*>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId)
        != 0) {
        HCCL_ERROR("[InsTempScatterOmniPipeNHRDpu] HcommSendRequest failed");
        HcommBatchModeStart(param.algTag);
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("[InsTempScatterOmniPipeNHRDpu] HcommSendRequest run over, sendMsgId[%u]", sendMsgId);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeNHRDpu::WaitDpuResponse(
    const OpParam& param, const TemplateResource& templateResource, u32 sendMsgId)
{
    void* recvData = nullptr;
    u32 recvMsgId = 0;
    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempScatterOmniPipeNHRDpu] HcommWaitResponse failed");
        HcommBatchModeStart(param.algTag);
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("[InsTempScatterOmniPipeNHRDpu] HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempScatterOmniPipeNHRDpu] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("[InsTempScatterOmniPipeNHRDpu] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

// DPU无法访问NPU的userIn，root在DPU发送前需先把userIn数据搬到cclBuff
HcclResult InsTempScatterOmniPipeNHRDpu::PreLocalCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    // 只有数据源是INPUT（userIn）时才需要PreLocalCopy
    // OmniPipe流水线中：i=0时inBuffType=INPUT需copy；i>0时inBuffType=HCCL_BUFFER数据已在cclBuff跳过
    if (tempAlgParams.buffInfo.inBuffType != BufferType::INPUT) {
        return HcclResult::HCCL_SUCCESS;
    }

    const auto& stepSliceInfo = tempAlgParams.stepSliceInfo;
    const u32 dim0Idx = myRank_ % (stepSliceInfo.stepSliceSize.size());
    u64 inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    u64 outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    void* inputPtr = tempAlgParams.buffInfo.inputPtr;
    void* cclBuffAddr = tempAlgParams.buffInfo.outputPtr;

    u64 totalPieceNum = stepSliceInfo.stepSliceSize[dim0Idx].size();
    for (u64 i = 0; i < totalPieceNum; i++) {
        u64 sz = stepSliceInfo.stepSliceSize[dim0Idx][i];
        if (sz == 0) {
            continue;
        }
        u64 inputStride = stepSliceInfo.inputOmniPipeSliceStride[dim0Idx][i];
        u64 outputStride = stepSliceInfo.outputOmniPipeSliceStride[dim0Idx][i];
        auto srcSlice = DataSlice(inputPtr, inBuffBaseOff + inputStride, sz, sz / dataTypeSize);
        auto dstSlice = DataSlice(cclBuffAddr, outBuffBaseOff + outputStride, sz, sz / dataTypeSize);
        HCCL_DEBUG(
            "myRank[%u], srcSlice:%s, dstSlice:%s", myRank_, srcSlice.Describe().c_str(), dstSlice.Describe().c_str());
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

// Scatter语义：Root单向分发，非Root接收；NHR多步递归，中间rank转发已收数据
HcclResult InsTempScatterOmniPipeNHRDpu::GetStepInfo(
    u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo, u32 rootAlgRank, u32 myAlgRank)
{
#ifndef AICPU_COMPILE
    u32 rankSize = templateRankSize_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.step = step;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.myRank = myRank_;

    u32 deltaRoot = (rootAlgRank + rankSize - myAlgRank) % rankSize;
    u32 deltaRankPair = 1 << step;

    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    u32 nRanks = 0;
    bool isPerfect = (rankSize & (rankSize - 1)) == 0;
    if (!isPerfect && step == nSteps - 1) {
        nRanks = rankSize - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) { // 需要发
        u32 sendTo = (myAlgRank + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            stepInfo.txSliceIdxs.push_back(txSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }
        stepInfo.toRank = subCommRanks_[0].at(sendTo);
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom = (myAlgRank + deltaRankPair) % rankSize;
        u32 rxSliceIdx = myAlgRank;
        for (u32 i = 0; i < nSlices; i++) {
            stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }
        stepInfo.fromRank = subCommRanks_[0].at(recvFrom);
        stepInfo.nSlices = nSlices;
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

// Scatter语义：Root单向分发数据到非Root，非Root仅接收；NHR多步递归实现
HcclResult InsTempScatterOmniPipeNHRDpu::DPUKernelRun(
    const TemplateDataParams& tempAlgParam, const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunNHR(channels, tempAlgParam));
#endif
    return HcclResult::HCCL_SUCCESS;
}

// NHR主体：stepSliceInfo驱动多步分发，Root从inputPtr发，非Root从cclBuff转发，收发用SendWrite/RecvWrite/SendRecvWrite
// 构建Tx批数据切片：查找toRank的tx
// channel，按txSliceIdxs×repeatNum构建src/dst切片，root从inputPtr发，非root从localCclBuff转发
HcclResult InsTempScatterOmniPipeNHRDpu::BuildTxBatchSlices(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const AicpuNHRStepInfo& stepInfo,
    const StepSliceInfo& stepSliceInfo, u32 dim0Idx, u64 repeatNum, u64 outBuffBaseOff, void* localCclBuffAddr,
    u32 dataTypeSize, u32 rootAlgRank, const ChannelInfo*& txCh, std::vector<DataSlice>& txSrcSlices,
    std::vector<DataSlice>& txDstSlices)
{
#ifndef AICPU_COMPILE
    auto txIt = channels.find(stepInfo.toRank);
    CHK_PRT_RET(
        txIt == channels.end() || txIt->second.empty(),
        HCCL_ERROR("[RunNHR] tx channel not found for toRank[%u]", stepInfo.toRank), HCCL_E_INTERNAL);
    txCh = &txIt->second[0];
    void* remoteCclBuffAddr = txCh->remoteCclMem.addr;
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txIdx = stepInfo.txSliceIdxs[i];
        u32 originIndex = (txIdx < rootAlgRank) ? txIdx : txIdx - 1;
        for (u64 rpt = 0; rpt < repeatNum; rpt++) {
            u64 idx = repeatNum * originIndex + rpt;
            u64 sz = stepSliceInfo.stepSliceSize[dim0Idx][idx];
            if (sz == 0) {
                continue;
            }
            u64 outputStride = stepSliceInfo.outputOmniPipeSliceStride[dim0Idx][idx];
            // 统一从cclBuff读取：root的userIn数据已在PreLocalCopy中搬到cclBuff
            u64 srcOff = outBuffBaseOff + outputStride;
            txSrcSlices.emplace_back(localCclBuffAddr, srcOff, sz, sz / dataTypeSize);
            u64 dstOff = outBuffBaseOff + outputStride;
            txDstSlices.emplace_back(remoteCclBuffAddr, dstOff, sz, sz / dataTypeSize);
            HCCL_DEBUG(
                "[RunNHR] myRank[%u] send to Rank[%u] sz[%llu], outputStride[%llu]", myRank_, stepInfo.toRank, sz,
                outputStride);
            HCCL_DEBUG(
                "myRank[%u], txSrcSlices:%s, txDstSlices:%s", myRank_, txSrcSlices[i].Describe().c_str(),
                txDstSlices[i].Describe().c_str());
        }
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

// 构建Rx批数据切片：查找fromRank的rx channel，按rxSliceIdxs×repeatNum构建src/dst切片，远端cclBuff→本地cclBuff
HcclResult InsTempScatterOmniPipeNHRDpu::BuildRxBatchSlices(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const AicpuNHRStepInfo& stepInfo,
    const StepSliceInfo& stepSliceInfo, u32 dim0Idx, u64 repeatNum, u64 outBuffBaseOff, void* localCclBuffAddr,
    u32 dataTypeSize, u32 rootAlgRank, const ChannelInfo*& rxCh, std::vector<DataSlice>& rxSrcSlices,
    std::vector<DataSlice>& rxDstSlices)
{
#ifndef AICPU_COMPILE
    auto rxIt = channels.find(stepInfo.fromRank);
    CHK_PRT_RET(
        rxIt == channels.end() || rxIt->second.empty(),
        HCCL_ERROR("[RunNHR] rx channel not found for fromRank[%u]", stepInfo.fromRank), HCCL_E_INTERNAL);
    rxCh = &rxIt->second[0];
    void* remoteCclBuffAddr = rxCh->remoteCclMem.addr;
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxIdx = stepInfo.rxSliceIdxs[i];
        u32 originIndex = (rxIdx < rootAlgRank) ? rxIdx : rxIdx - 1;
        for (u64 rpt = 0; rpt < repeatNum; rpt++) {
            u64 idx = repeatNum * originIndex + rpt;
            u64 sz = stepSliceInfo.stepSliceSize[dim0Idx][idx];
            if (sz == 0) {
                continue;
            }
            u64 outputStride = stepSliceInfo.outputOmniPipeSliceStride[dim0Idx][idx];
            u64 off = outBuffBaseOff + outputStride;
            rxSrcSlices.emplace_back(remoteCclBuffAddr, off, sz, sz / dataTypeSize);
            rxDstSlices.emplace_back(localCclBuffAddr, off, sz, sz / dataTypeSize);
            HCCL_DEBUG(
                "[RunNHR] myRank[%u] recv from Rank[%u] sz[%llu], outputStride[%llu]", myRank_, stepInfo.fromRank, sz,
                outputStride);
            HCCL_DEBUG(
                "myRank[%u], rxSrcSlices:%s, rxDstSlices:%s", myRank_, rxSrcSlices[i].Describe().c_str(),
                rxDstSlices[i].Describe().c_str());
        }
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

// 按Tx/Rx情况选择DPU通信原语：仅Tx用SendWrite，仅Rx用RecvWrite，同时用SendRecvWrite
HcclResult InsTempScatterOmniPipeNHRDpu::ExecuteDpuCommPrimitive(
    bool hasTx, bool hasRx, const ChannelInfo* txCh, const ChannelInfo* rxCh, const std::vector<DataSlice>& txSrcSlices,
    const std::vector<DataSlice>& txDstSlices, const std::vector<DataSlice>& rxSrcSlices,
    const std::vector<DataSlice>& rxDstSlices, u32 step)
{
#ifndef AICPU_COMPILE
    if (hasTx && !hasRx) {
        SlicesList txSlicesList(txSrcSlices, txDstSlices);
        DataInfo sendData(*txCh, txSlicesList);
        CHK_PRT_RET(SendWrite(sendData), HCCL_ERROR("[RunNHR] SendWrite failed at step[%u]", step), HCCL_E_INTERNAL);
    } else if (!hasTx && hasRx) {
        SlicesList rxSlicesList(rxSrcSlices, rxDstSlices);
        DataInfo recvData(*rxCh, rxSlicesList);
        CHK_PRT_RET(RecvWrite(recvData), HCCL_ERROR("[RunNHR] RecvWrite failed at step[%u]", step), HCCL_E_INTERNAL);
    } else if (hasTx && hasRx) {
        TxRxChannels channelsPair(*txCh, *rxCh);
        TxRxSlicesList slicesPair(SlicesList(txSrcSlices, txDstSlices), SlicesList(rxSrcSlices, rxDstSlices));
        SendRecvInfo sendRecvInfo(channelsPair, slicesPair);
        CHK_PRT_RET(
            SendRecvWrite(sendRecvInfo), HCCL_ERROR("[RunNHR] SendRecvWrite failed at step[%u]", step),
            HCCL_E_INTERNAL);
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeNHRDpu::RunNHR(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const TemplateDataParams& tempAlgParam)
{
#ifndef AICPU_COMPILE
    HCCL_DEBUG("[RunNHR] myRank[%u], root[%u], tempAlgParam.root[%u]", myRank_, root_, tempAlgParam.root);
    dataType_ = tempAlgParam.dataType;

    u32 myAlgRank = 0;
    u32 rootAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    CHK_RET(GetAlgRank(tempAlgParam.root, subCommRanks_[0], rootAlgRank));

    const auto& stepSliceInfo = tempAlgParam.stepSliceInfo;
    const u32 dim0Idx = myRank_ % (stepSliceInfo.stepSliceSize.size());
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u32 nSteps = GetNHRStepNum(templateRankSize_);

    // stepSliceInfo第二维按rank分组（跳过root），每组repeatNum个piece
    u64 totalPieceNum = stepSliceInfo.stepSliceSize[dim0Idx].size();
    u64 peerNum = templateRankSize_ - 1;
    CHK_PRT_RET(
        peerNum == 0 || totalPieceNum % peerNum != 0,
        HCCL_ERROR(
            "[InsTempScatterOmniPipeNHRDpu][RunNHR] totalPieceNum[%llu] not divisible by peerNum[%llu]", totalPieceNum,
            peerNum),
        HCCL_E_INTERNAL);
    u64 repeatNum = totalPieceNum / peerNum;

    for (u32 step = 0; step < nSteps; step++) {
        CHK_RET(ExecuteNhrStep(channels, tempAlgParam, repeatNum, rootAlgRank, myAlgRank, step, nSteps));
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeNHRDpu::ExecuteNhrStep(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const TemplateDataParams& tempAlgParam, u64 repeatNum,
    u32 rootAlgRank, u32 myAlgRank, u32 step, u32 nSteps)
{
#ifndef AICPU_COMPILE
    AicpuNHRStepInfo stepInfo;
    CHK_RET(GetStepInfo(step, nSteps, stepInfo, rootAlgRank, myAlgRank));
    if (stepInfo.txSliceIdxs.empty() && stepInfo.rxSliceIdxs.empty()) {
        HCCL_DEBUG("[RunNHR] step[%u] no tx/rx slice, skip.", step);
        return HcclResult::HCCL_SUCCESS;
    }
    bool hasTx = !stepInfo.txSliceIdxs.empty();
    bool hasRx = !stepInfo.rxSliceIdxs.empty();
    std::vector<DataSlice> txSrcSlices, txDstSlices;
    std::vector<DataSlice> rxSrcSlices, rxDstSlices;
    const ChannelInfo* txCh = nullptr;
    const ChannelInfo* rxCh = nullptr;
    const auto& stepSliceInfo = tempAlgParam.stepSliceInfo;
    u32 dim0Idx = myRank_ % stepSliceInfo.stepSliceSize.size();
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    void* localCclBuffAddr = tempAlgParam.buffInfo.outputPtr;
    u64 outBuffBaseOff = tempAlgParam.buffInfo.outBuffBaseOff;
    if (hasTx) {
        CHK_RET(BuildTxBatchSlices(
            channels, stepInfo, stepSliceInfo, dim0Idx, repeatNum, outBuffBaseOff, localCclBuffAddr, dataTypeSize,
            rootAlgRank, txCh, txSrcSlices, txDstSlices));
    }
    if (hasRx) {
        CHK_RET(BuildRxBatchSlices(
            channels, stepInfo, stepSliceInfo, dim0Idx, repeatNum, outBuffBaseOff, localCclBuffAddr, dataTypeSize,
            rootAlgRank, rxCh, rxSrcSlices, rxDstSlices));
    }
    if ((hasTx && txSrcSlices.empty()) || (hasRx && rxSrcSlices.empty())) {
        HCCL_DEBUG("[RunNHR] step[%u] skip due to empty txSrcSlices or rxSrcSlices.", step);
        return HcclResult::HCCL_SUCCESS;
    }
    CHK_RET(
        ExecuteDpuCommPrimitive(hasTx, hasRx, txCh, rxCh, txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices, step));
#endif
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempScatterOmniPipeNHRDpu", InsTempScatterOmniPipeNHRDpu);

} // namespace ops_hccl

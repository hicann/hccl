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
#include <algorithm>
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "hccl_sym_win.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
namespace ops_hccl {
std::vector<CostModelParam> InsTempAllGatherMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    float A = 0;
    float B = 0;
    float C = 0;
    float D = 0;
    if (param.rankSize == 1) {
        HCCL_DEBUG("[InsTempAllGatherMesh1D][CalcCostCoeff] no transfer data");
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
        std::vector<CostModelParam> params;
        params.push_back({A, B, C, D});
        return params;
    }

    int portNum = param.portNum.empty() ? 1 : param.portNum[0];
    int kernelNum = 8;
    int taskNum
        = CostModelManager::CalcTransTaskNum(param.rankSize) + CostModelManager::CalcSyncTaskNum(param.rankSize) * 2;

    CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    float B1 = 0; // userin->ccl
    float B2 = 0; // userin->userout
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B1);
    }
    if (param.inputBuffer != param.outputBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B2);
    }
    B = B1 + B2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    D = 0.6e-6f * taskNum;

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

InsTempAllGatherMesh1D::InsTempAllGatherMesh1D(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks),
      isConcurrent_(std::string(param.algName) == "AicpuAllGatherConcurMeshNHR")
{}
InsTempAllGatherMesh1D::~InsTempAllGatherMesh1D() {}

HcclResult InsTempAllGatherMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMesh1D][CalcRes] start");
    std::vector<HcclChannelDesc> level0Channels;
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[InsTempAllGatherMesh1D][CalcRes] topoInfo is nullptr"), HCCL_E_PARA);
    if (isConcurrent_ && concurrentTopo_ == COMM_TOPO_CLOS) {
        // 为所有peer申请CLOS物理链路上的共享binding通道。
        CHK_RET(
            CalcChannelRequestMeshClosMultiJetty(comm, param, topoInfo, subCommRanks_, level0Channels, false, false));
    } else if (isConcurrent_) {
        // 优先选择Mesh物理层上的直连通道。
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(
            comm, param, topoInfo, subCommRanks_, level0Channels, COMM_TOPO_1DMESH));
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    channelsPerRank_ = CalcChannelsPerRankMin(level0Channels);
    maxChannelsPerRank_ = CalcChannelsPerRank(level0Channels);
    HCCL_INFO(
        "[InsTempAllGatherMesh1D][CalcRes] channelsPerRank_[%u], maxChannelsPerRank_[%u]", channelsPerRank_,
        maxChannelsPerRank_);
    CHK_RET(GetRes(resourceRequest));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}
HcclResult InsTempAllGatherMesh1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = GetThreadNum();
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1D::GetThreadNum() const
{
    if (isConcurrent_) {
        // 主流处理本卡拷贝，每个 peer/channel 使用独立从流。
        return 1 + (templateRankSize_ > 1 ? templateRankSize_ - 1 : 0) * maxChannelsPerRank_;
    }
    return templateRankSize_ > 1 ? (templateRankSize_ - 1) * maxChannelsPerRank_ : 1;
}

HcclResult InsTempAllGatherMesh1D::SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRankMin(channels);
    maxChannelsPerRank_ = CalcChannelsPerRank(channels);
    HCCL_INFO(
        "[InsTempAllGatherMesh1D][SetchannelsPerRank] channelsPerRank_[%u], maxChannelsPerRank_[%u]", channelsPerRank_,
        maxChannelsPerRank_);
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 0;
    // 为单算子或Concurrent的CCL暂存区按rank预留分片空间。
    if (opMode_ == OpMode::OPBASE || isConcurrent_) {
        scratchMultiple = templateRankSize_;
    }
    return scratchMultiple;
}

HcclResult InsTempAllGatherMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    HCCL_INFO("[InsTempAllGatherMesh1D] Run start");
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_INFO("[InsTempAllGatherMesh1D] Rank [%d], get slicesize zero.", myRank_);
        return HCCL_SUCCESS;
    }

    threadNum_ = GetThreadNum();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    HCCL_DEBUG(
        "[InsTempAllGatherMesh1D][KernelRun] rank[%u], threadNum[%u], concurrent[%d], "
        "concurrentTopo[%d], remoteMemAccess[%d].",
        myRank_, threadNum_, isConcurrent_, static_cast<int>(concurrentTopo_), enableRemoteMemAccess_);
    CHK_PRT_RET(
        isConcurrent_ && templateResource.threads.size() < threadNum_,
        HCCL_ERROR(
            "[InsTempAllGatherMesh1D][KernelRun] rank[%u] has insufficient Concurrent threads: "
            "available[%zu], required[%u].",
            myRank_, templateResource.threads.size(), threadNum_),
        HCCL_E_INTERNAL);
    // 普通Mesh先完成本卡拷贝，Concurrent非对称路径仅暂存CCL。
    if (!isConcurrent_ || !enableRemoteMemAccess_) {
        CHK_RET(LocalDataCopy(templateResource.threads));
    }
    if (!isConcurrent_ && templateRankSize_ == 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(RunAllGatherMesh(templateResource.threads, templateResource.channels));
    if (isConcurrent_) {
        // peer 从流读取数据时，主流并行拷贝本卡输出分片。
        CHK_RET(LocalDataCopy(templateResource.threads, true));
    }

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_INFO("[InsTempAllGatherMesh1D] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::RunAllGatherMeshOnChannel(
    const ThreadHandle& thread, const ChannelInfo& linkRemote, const u32 connectedRank, const u32 connectedAlgRank,
    const u32 channelIdx)
{
    void* remoteCclBuffAddr = linkRemote.remoteCclMem.addr;
    std::vector<DataSlice> rxDstSlicesAll;
    std::vector<DataSlice> rxSrcSlicesAll;

    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
        u64 scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
        if (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
            scratchBase = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        }

        u64 rxOutOffset = tempAlgParams_.outputSliceStride * connectedAlgRank + outBaseOff + elemOffset_[channelIdx];
        u64 rxSrcOffset = 0;
        void* rxSrcPtr = nullptr;
        void* rxDstPtr = tempAlgParams_.buffInfo.outputPtr;
        u64 rxScratchOffset = scratchBase + tempAlgParams_.sliceSize * connectedAlgRank + elemOffset_[channelIdx];
        if (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
            rxScratchOffset
                = scratchBase + tempAlgParams_.inputSliceStride * connectedAlgRank + elemOffset_[channelIdx];
        }
        rxSrcOffset = (!enableRemoteMemAccess_) ? rxScratchOffset : rxOutOffset;
        rxSrcPtr = (!enableRemoteMemAccess_) ? remoteCclBuffAddr : linkRemote.remoteOutputGraphMode.addr;
        if (isConcurrent_ && enableRemoteMemAccess_) {
            // 将Concurrent对称路径的读取源设为peer的input窗口。
            rxSrcPtr = linkRemote.remoteInputGraphMode.addr;
            // 按peer的输入分片布局计算读取偏移。
            rxSrcOffset = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride
                          + connectedAlgRank * tempAlgParams_.inputSliceStride + elemOffset_[channelIdx];
        }
        CHK_PRT_RET(
            isConcurrent_ && rxSrcPtr == nullptr,
            HCCL_ERROR(
                "[InsTempAllGatherMesh1D][RunAllGatherMeshOnChannel] peer[%u] %s buffer is unavailable.", connectedRank,
                enableRemoteMemAccess_ ? "input" : "ccl"),
            HCCL_E_INTERNAL);

        rxDstSlicesAll.emplace_back(rxDstPtr, rxOutOffset, sizeOut_[channelIdx], elemCountOut_[channelIdx]);
        rxSrcSlicesAll.emplace_back(rxSrcPtr, rxSrcOffset, sizeOut_[channelIdx], elemCountOut_[channelIdx]);

        HCCL_DEBUG(
            "[InsTempAllGatherMesh1D][RunAllGatherMesh] rankId [%d] connectedRank [%d] rpt [%d] "
            "channelIdx[%u] offset[%llu] sliceSize[%llu] count[%llu] srcOffset[%llu] outOffset[%llu].",
            myRank_, connectedRank, rpt, channelIdx, elemOffset_[channelIdx], sizeOut_[channelIdx],
            elemCountOut_[channelIdx], rxSrcOffset, rxOutOffset);
    }

    std::vector<DataSlice> emptySlices;
    TxRxSlicesList sendRecvSlicesList({emptySlices, emptySlices}, {rxSrcSlicesAll, rxDstSlicesAll});
    TxRxChannels sendRecvChannels(linkRemote, linkRemote);
    SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList, dataType_);
    if (isConcurrent_ && concurrentTopo_ == COMM_TOPO_CLOS) {
        // 在CLOS轴的peer线程上提交普通Read任务。
        CHK_RET(SendRecvRead(sendRecvInfo, thread));
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        SendRecvBatchRead(sendRecvInfo, thread), HCCL_ERROR("[InsTempAllGatherMesh1D] RunAllGather Send failed"),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::RunAllGatherMesh(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] RunAllGatherMesh RankIDs[%d].", myRank_);

    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    // Concurrent的0号线程执行本卡拷贝，peer读取从1号线程开始。
    u32 queIdx = isConcurrent_ ? 1 : 0;
    for (u32 rankIdx = 1; rankIdx < templateRankSize_; rankIdx++) {
        u32 connectedAlgRank = (myAlgRank + rankIdx) % templateRankSize_;
        u32 connectedRank = subCommRanks_[0][connectedAlgRank];

        HCCL_INFO(
            "[InsTempAllGatherMesh1D] RunAllGatherMesh RankIDs[%d], connectedRank[%d], connectedAlgRank[%d].", myRank_,
            connectedRank, connectedAlgRank);

        CHK_PRT_RET(
            channels.count(connectedRank) == 0 || channels.at(connectedRank).empty(),
            HCCL_ERROR(
                "[InsTempAllGatherMesh1D][RankID]=%u connectedRank=%d, channels.size=%zu", myRank_, connectedRank,
                channels.size()),
            HcclResult::HCCL_E_INTERNAL);

        const std::vector<ChannelInfo>& curChannels = channels.at(connectedRank);
        u64 sliceSize = tempAlgParams_.sliceSize;
        if (tempAlgParams_.tailSize != 0 && connectedAlgRank == templateRankSize_ - 1) {
            sliceSize = tempAlgParams_.tailSize;
        }
        u64 sliceCount = sliceSize / dataTypeSize;
        CHK_RET(CalcDataSplitByPortGroup(sliceCount, dataTypeSize, curChannels, elemCountOut_, sizeOut_, elemOffset_));

        const u32 curChannelNum = std::min(static_cast<u32>(curChannels.size()), channelsPerRank_);
        for (u32 channelIdx = 0; channelIdx < curChannelNum; channelIdx++) {
            CHK_PRT_RET(
                queIdx >= threads.size(),
                HCCL_ERROR(
                    "[InsTempAllGatherMesh1D][RankID]=%u queIdx=%u, threads.size=%zu", myRank_, queIdx, threads.size()),
                HcclResult::HCCL_E_INTERNAL);
            const ChannelInfo& linkRemote = curChannels[channelIdx];
            CHK_RET(
                RunAllGatherMeshOnChannel(threads[queIdx], linkRemote, connectedRank, connectedAlgRank, channelIdx));
            queIdx++;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::LocalDataCopy(const std::vector<ThreadHandle>& threads, bool outputOnly)
{
    HCCL_INFO(
        "[InsTempAllGatherMesh1D][LocalDataCopy] rank[%u], concurrent[%d], outputOnly[%d], remoteMemAccess[%d].",
        myRank_, isConcurrent_, outputOnly, enableRemoteMemAccess_);
    if (threads.empty()) {
        return HcclResult::HCCL_E_INTERNAL;
    }

    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 sliceSize = tempAlgParams_.sliceSize;

    if (tempAlgParams_.tailSize != 0 && myAlgRank == templateRankSize_ - 1) {
        sliceSize = tempAlgParams_.tailSize;
    }

    u64 sliceCount = sliceSize / dataTypeSize;
    for (u32 rpt = 0; rpt < tempAlgParams_.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams_.buffInfo.inBuffBaseOff + rpt * tempAlgParams_.inputRepeatStride;
        const u64 outBaseOff = tempAlgParams_.buffInfo.outBuffBaseOff + rpt * tempAlgParams_.outputRepeatStride;
        const u64 inOff = tempAlgParams_.inputSliceStride * myAlgRank + inBaseOff;
        const u64 outOff = tempAlgParams_.outputSliceStride * myAlgRank + outBaseOff;

        DataSlice srcSlice(tempAlgParams_.buffInfo.inputPtr, inOff, sliceSize, sliceCount);
        if (isConcurrent_ && !enableRemoteMemAccess_) {
            const auto& buff = tempAlgParams_.buffInfo;
            // 按重复块和算法rank定位本卡在CCL中的暂存分片。
            const u64 scratchOffset = buff.hcclBuffBaseOff
                                      + static_cast<u64>(rpt) * tempAlgParams_.sliceSize * templateRankSize_
                                      + myAlgRank * tempAlgParams_.sliceSize;
            CHK_PRT_RET(
                buff.hcclBuff.addr == nullptr || scratchOffset > buff.hcclBuff.size
                    || sliceSize > buff.hcclBuff.size - scratchOffset,
                HCCL_ERROR(
                    "[InsTempAllGatherMesh1D][LocalDataCopy] rank[%u] CCL slice is unavailable or out of range: "
                    "offset[%llu], size[%llu], bufferSize[%llu], outputOnly[%d].",
                    myRank_, scratchOffset, sliceSize, buff.hcclBuff.size, outputOnly),
                HCCL_E_INTERNAL);
            if (outputOnly) {
                // 从已暂存的CCL分片读取本卡输出数据。
                srcSlice = DataSlice(buff.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            }
        }
        bool skipOutCopy = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.outputPtr && inOff == outOff);

        // 普通Mesh直接生成本卡输出，Concurrent仅在输出阶段执行该拷贝。
        if (!skipOutCopy && (!isConcurrent_ || outputOnly)) {
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOff, sliceSize, sliceCount);
            HCCL_DEBUG(
                "[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] srcSlice: inBaseOff[%llu] "
                "inOff[%llu] srcOffset[%llu] "
                "sliceSize[%llu] count[%llu].",
                myRank_, myAlgRank, inBaseOff, inOff, srcSlice.offset_, sliceSize, sliceCount);
            HCCL_DEBUG(
                "[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] dstSlice: outBaseoff[%llu] "
                "outOff[%llu] sliceSize[%llu] count[%llu].",
                myRank_, myAlgRank, outBaseOff, outOff, sliceSize, sliceCount);
            CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
        }

        // 前置阶段将本卡input分片写入CCL，供peer读取。
        if (!enableRemoteMemAccess_ && !outputOnly) {
            const u64 scratchRepeatStride = tempAlgParams_.sliceSize * templateRankSize_;
            const u64 cclBaseOff = tempAlgParams_.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;
            u64 cclOff = cclBaseOff + tempAlgParams_.sliceSize * myAlgRank;
            DataSlice cclDstSlice(tempAlgParams_.buffInfo.hcclBuff.addr, cclOff, sliceSize, sliceCount);
            bool skipCclCopy
                = (tempAlgParams_.buffInfo.inputPtr == tempAlgParams_.buffInfo.hcclBuff.addr
                   && tempAlgParams_.buffInfo.inBuffBaseOff == tempAlgParams_.buffInfo.hcclBuffBaseOff
                   && (!isConcurrent_ || inOff == cclOff));
            if (!skipCclCopy) {
                HCCL_DEBUG(
                    "[InsTempAllGatherMesh1D][LocalDataCopy] RankID [%d] AlgRank [%d] copy to ccl: "
                    "cclBaseOff[%llu] cclOff[%llu] sliceSize[%llu] count[%llu].",
                    myRank_, myAlgRank, cclBaseOff, cclOff, sliceSize, sliceCount);
                CHK_RET(LocalCopy(threads[0], srcSlice, cclDstSlice));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherMesh1D::PostLocalCopy(const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy.");
    if (tempAlgParams_.buffInfo.outBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy skip because output is scratch");
        return HcclResult::HCCL_SUCCESS;
    }
    if (tempAlgParams_.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempAllGatherMesh1D] PostLocalCopy skip because input is scratch and should be read to output");
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
            if (tempAlgParams_.tailSize != 0 && algRank == templateRankSize_ - 1) {
                sliceSize = tempAlgParams_.tailSize;
            }
            u64 scratchOffset = tempAlgParams_.sliceSize * algRank + scratchBase;
            u64 outOffset = tempAlgParams_.outputSliceStride * algRank + outBaseOff;
            u64 sliceCount = sliceSize / dataTypeSize;
            DataSlice srcSlice(tempAlgParams_.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams_.buffInfo.outputPtr, outOffset, sliceSize, sliceCount);
            HCCL_DEBUG(
                "[InsTempAllGatherMesh1D] LocalDataCopy RankID [%d] dataRank [%d] dataAlgRank[%d] "
                "scratchBase[%d] outBaseOff[%d] scratchOffset[%d] outOffset[%d].",
                myRank_, rank, algRank, outBaseOff, outBaseOff, scratchOffset, outOffset);
            CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherMesh1D::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub)
{
    notifyIdxMianToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMianToSub.push_back(0);
    }
}

void InsTempAllGatherMesh1D::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace ops_hccl

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_meshchunk.h"
#include "cost_model.h"

namespace ops_hccl {
InsTempReduceScatterMesh1DMeshChunk::InsTempReduceScatterMesh1DMeshChunk(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceScatterMesh1DMeshChunk::~InsTempReduceScatterMesh1DMeshChunk() {}

std::vector<CostModelParam> InsTempReduceScatterMesh1DMeshChunk::CalcCostCoeff(CalcCostCoeffParam param)
{
    if (param.rankSize > 8) {
        return {};
    }
    int portNum = (param.isPod && param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      (param.portNum[0] + param.portNum[1]) :
                      param.portNum[0];
    int kernelNum = param.rankSize * 5;
    int taskNum = CostModelManager::CalcTransTaskNum(param.rankSize) * (param.rankSize - 1)
                  + CostModelManager::CalcSyncTaskNum(param.rankSize) * param.rankSize
                  + CostModelManager::CalcTransTaskNum(param.rankSize);
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
    } else {
        B = 0.0f;
    }
    B = B * 2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_RET(GetRes(resourceRequest));
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DMeshChunk::GetThreadNum() const
{
    return templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = static_cast<u32>(GetThreadNum());
    resourceRequest.slaveThreadNum = threadNum - 1;
    const u32 NOTIFY_NUM_PER_SLAVE_THREAD = 2;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(NOTIFY_NUM_PER_SLAVE_THREAD);
    }
    resourceRequest.notifyNumOnMainThread = (threadNum - 1) * NOTIFY_NUM_PER_SLAVE_THREAD;
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DMeshChunk::ScratchSlotBaseOff(
    const TemplateDataParams& p, u32 repeatIdx, u32 algRank) const
{
    return p.buffInfo.hcclBuffBaseOff + repeatIdx * p.outputRepeatStride + algRank * p.outputSliceStride;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::CalcDataSplitByPortGroup(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset)
{
    elemCountOut.clear();
    sizeOut.clear();
    elemOffset.clear();
    u32 chNum = static_cast<u32>(channels.size());
    if (chNum == 0 || dataTypeSize == 0) {
        return HcclResult::HCCL_E_PARA;
    }
    // 基类默认: 通道均分(sole 场景 1 通道 = 全量, 行为与旧代码一致)
    u64 remain = totalDataCount;
    u64 off = 0;
    for (u32 c = 0; c < chNum; c++) {
        u64 cnt = (c == chNum - 1) ? remain : totalDataCount / chNum;
        elemCountOut.push_back(cnt);
        sizeOut.push_back(cnt * dataTypeSize);
        elemOffset.push_back(off);
        off += cnt * dataTypeSize;
        remain -= cnt;
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DMeshChunk::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = templateRankSize_ - 1;
    return scratchMultiple;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::CalcSliceInfoVec(const u64& dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(subCommRanks_.size());
    sliceInfoVec.resize(templateRankSize_, tmp);
    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < sliceInfoVec.size(); rankIdx++) {
        SliceInfo slice = {accumOff, dataSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += dataSize;
    }
    CHK_PRT_RET(
        (sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size
         != dataSize * templateRankSize_),
        HCCL_ERROR("[CollAlgFactory] Rank [%d], SliceInfo calculation error!", myRank_), HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(
        templateRankSize_ == 0, HCCL_ERROR("[InsTempReduceScatterMesh1DMeshChunk] templateRankSize_ is 0"),
        HCCL_E_INTERNAL);
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx_));
    supportSymmetricMemAccess_ = param.supportSymmetricMemory && param.inputSymWindow != nullptr;
    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSliceInfoVec(tempAlgParams.sliceSize, sliceInfoVec));
    HCCL_INFO("[InsTempReduceScatterMesh1DMeshChunk] Run Start");
    if (!supportSymmetricMemAccess_) {
        PreCopy(tempAlgParams, templateResource.threads);
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    CHK_RET(RunReduceScatter(templateResource.channels, templateResource.threads, tempAlgParams, sliceInfoVec));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    if (supportSymmetricMemAccess_) {
        LocalDataCopy(tempAlgParams, templateResource.threads);
    } else {
        PostCopy(tempAlgParams, templateResource.threads);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::PreCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    HCCL_INFO("[InsTempReduceScatterMesh1DMeshChunk][PreCopy], copy from userIn to scratch");
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        DataSlice srcSlice = DataSlice(
            tempAlgParams.buffInfo.inputPtr,
            tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride
                + rankIdx_ * tempAlgParams.inputSliceStride,
            processSize_);
        DataSlice dstSlice = DataSlice(
            tempAlgParams.buffInfo.hcclBuff.addr, ScratchSlotBaseOff(tempAlgParams, repeatIdx, rankIdx_), processSize_);
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::LocalDataCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempReduceScatterMesh1DMeshChunk][LocalDataCopy], copy from userIn to userOut");
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        const u64 srcOff = tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride
                           + rankIdx_ * tempAlgParams.inputSliceStride;
        // 序列对称场景 outputPtr==inputPtr 且偏移相同(实际收尾由框间模板承载), 跳过自拷贝
        const u64 dstOff = tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                           + rankIdx_ * tempAlgParams.outputSliceStride;
        if (tempAlgParams.buffInfo.inputPtr == tempAlgParams.buffInfo.outputPtr && srcOff == dstOff) {
            continue;
        }
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOff, processSize_);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, dstOff, processSize_);
        CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::RunReduceScatter(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParams, RankSliceInfo& sliceInfoVec)
{
    HCCL_INFO("[InsTempReduceScatterMesh1DMeshChunk][RunReduceScatter] myRank[%d]", myRank_);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    // 单 rank 防御：无对端可交换，跳过轮转（PreCopy/PostCopy 完成数据穿透）
    if (templateRankSize_ == 1) {
        HCCL_INFO("[InsTempReduceScatterMesh1DMeshChunk][RunReduceScatter] single rank, skip rotation.");
        return HcclResult::HCCL_SUCCESS;
    }

    uint64_t sliceNum = templateRankSize_ - 1;
    uint64_t mySliceSize = sliceInfoVec[myAlgRank][0].size; // 获取本rank需要处理的数据量，executor已按4K对齐
    uint64_t mySliceCount = mySliceSize / DATATYPE_SIZE_TABLE[dataType_];

    std::vector<uint64_t> sliceSize;
    // 数据切分为sliceNum块，前sliceNum-1块每块按4K向下对齐，最后一块取余数(无需对齐)。
    // 选向下对齐是为了保证DoMeshChunk第0步接收区[mySliceSize-lastBlock, mySliceSize-lastBlock+sliceSize[0])
    // 不越界(需sliceSize[0] <= lastBlock)；若向上对齐则sliceSize[0] > lastBlock，会跨块越界读取。
    uint64_t alignSliceSize = (mySliceSize / sliceNum) / AICPU_ALIGN_SIZE * AICPU_ALIGN_SIZE;
    if (sliceNum >= 2 && alignSliceSize > 0) {
        for (uint64_t i = 0; i < sliceNum - 1; i++) {
            sliceSize.push_back(alignSliceSize);
        }
        sliceSize.push_back(mySliceSize - alignSliceSize * (sliceNum - 1));
    } else {
        // 数据量过小无法按4K对齐切分时，回退到原有按元素均分(大/小块)逻辑，避免0尺寸分块
        uint64_t bigDataSliceNum = mySliceCount % sliceNum;
        uint64_t bigDataSliceSize = (mySliceCount / sliceNum + 1) * DATATYPE_SIZE_TABLE[dataType_];
        uint64_t smallDataSliceNum = sliceNum - mySliceCount % sliceNum;
        uint64_t smallDataSliceSize = mySliceCount / sliceNum * DATATYPE_SIZE_TABLE[dataType_];
        for (uint64_t i = 0; i < bigDataSliceNum; i++) {
            sliceSize.push_back(bigDataSliceSize);
        }
        for (uint64_t i = 0; i < smallDataSliceNum; i++) {
            sliceSize.push_back(smallDataSliceSize);
        }
    }
    uint64_t sliceRecvBaseOffset = 0;
    uint16_t rankNum = 2;
    for (uint16_t i = 0; i < (templateRankSize_ - rankNum); i++) {
        sliceRecvBaseOffset += sliceSize[i];
    }
    uint64_t sliceSendOffset_;
    uint64_t sliceRecvOffset_;
    DoMeshChunk(
        channels, threads, tempAlgParams, sliceSize, tempAlgParams.repeatNum, myAlgRank, sliceSendOffset_,
        sliceRecvOffset_, sliceRecvBaseOffset);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::DoMeshChunk(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParams, const std::vector<uint64_t>& sliceSize, const u32& repeatNum,
    const u32& myAlgRank, uint64_t& sliceSendOffset_, uint64_t& sliceRecvOffset_, const uint64_t& sliceRecvBaseOffset)
{
    for (uint16_t stepIdx = 0; stepIdx < (templateRankSize_ - 1); stepIdx++) {
        sliceSendOffset_ = 0;
        sliceRecvOffset_ = sliceRecvBaseOffset;
        uint16_t rankNum = 2;
        uint16_t tempNum = 3;
        for (uint16_t i = 0; i < (templateRankSize_ - 1); i++) {
            uint16_t nextNum = stepIdx + i + 1;
            if (nextNum >= templateRankSize_) {
                nextNum += 1;
            }
            uint16_t nextRank = (myAlgRank + nextNum) % templateRankSize_;
            uint16_t frontNum = 2 * myAlgRank - nextRank + templateRankSize_;
            uint16_t frontRank = frontNum % templateRankSize_;
            u32 toRank = subCommRanks_[0][frontRank];
            const std::vector<ChannelInfo>& peerChs = channels.at(toRank);
            u32 curChannelNum = std::min(static_cast<u32>(peerChs.size()), channelsPerRank_);
            // 切分/消费口径一致: 只把实际会消费的前 curChannelNum 条通道视图传给切分,
            // 保证切分份数==发送份数——否则基类模板(channelsPerRank_ 恒 1)在多通道
            // peerChs 下会把数据切成 N 份却只发第 0 份, 造成静默丢数据
            std::vector<ChannelInfo> splitView(peerChs.begin(), peerChs.begin() + curChannelNum);
            CHK_RET(CalcDataSplitByPortGroup(
                sliceSize[i] / dataTypeSize_, dataTypeSize_, splitView, elemCountOut_, sizeOut_, elemOffset_));
            uint16_t frontRankShifted = frontRank < myAlgRank ? frontRank : frontRank - 1;
            for (u32 channelIdx = 0; channelIdx < curChannelNum; channelIdx++) {
                if (sizeOut_[channelIdx] == 0) {
                    continue;
                }
                const ChannelInfo& linkSend = peerChs[channelIdx];
                const ChannelInfo& linkRecv = peerChs[channelIdx];
                void* remoteBuffAddr
                    = supportSymmetricMemAccess_ ? linkSend.remoteInputGraphMode.addr : linkSend.remoteCclMem.addr;
                u64 subSendOff = sliceSendOffset_ + elemOffset_[channelIdx];
                u64 subRecvOff = sliceRecvOffset_ + elemOffset_[channelIdx];
                u64 subSize = sizeOut_[channelIdx];
                u32 queIdx = frontRankShifted * channelsPerRank_ + channelIdx;
                CHK_PRT_RET(
                    queIdx >= threads.size(),
                    HCCL_ERROR(
                        "[InsTempReduceScatterMesh1DMeshChunk][DoMeshChunk] queIdx[%u] out of range, threadNum[%zu], "
                        "channelsPerRank_[%u]",
                        queIdx, threads.size(), channelsPerRank_),
                    HcclResult::HCCL_E_INTERNAL);

                std::vector<DataSlice> txSrcSlices;
                std::vector<DataSlice> txDstSlices;
                std::vector<DataSlice> rxSrcSlices;
                std::vector<DataSlice> rxDstSlices;
                for (u32 repeatIdx = 0; repeatIdx < repeatNum; repeatIdx++) {
                    u64 inputBaseOff = tempAlgParams.buffInfo.inBuffBaseOff
                                       + repeatIdx * tempAlgParams.inputRepeatStride
                                       + myAlgRank * tempAlgParams.inputSliceStride;
                    u64 slotRecvBase = ScratchSlotBaseOff(tempAlgParams, repeatIdx, myAlgRank);
                    u64 slotSendBase = ScratchSlotBaseOff(tempAlgParams, repeatIdx, frontRank);
                    u64 sliceExchOffset = supportSymmetricMemAccess_ ? subSendOff : subRecvOff;
                    DataSlice rxSrcSlice = DataSlice(
                        supportSymmetricMemAccess_ ? remoteBuffAddr : tempAlgParams.buffInfo.inputPtr,
                        inputBaseOff + sliceExchOffset, subSize, subSize / dataTypeSize_);
                    DataSlice rxDstSlice = DataSlice(
                        supportSymmetricMemAccess_ ? tempAlgParams.buffInfo.inputPtr :
                                                     tempAlgParams.buffInfo.hcclBuff.addr,
                        supportSymmetricMemAccess_ ? (inputBaseOff + sliceExchOffset) : (slotRecvBase + subRecvOff),
                        subSize, subSize / dataTypeSize_);
                    // 写方向: 接收方(frontRank)槽位为本 rank 持有的对端 slice 副本偏移;
                    // rx 列表在此占位不被消费, 其偏移仅保持 DataSlice 结构完整
                    // (与下方 WriteReduce 注释一致, 实际搬运只消费 tx 切片)
                    const u64 txBaseOff = tempAlgParams.buffInfo.inBuffBaseOff
                                          + repeatIdx * tempAlgParams.inputRepeatStride
                                          + frontRank * tempAlgParams.inputSliceStride + subSendOff;
                    DataSlice txSrcSlice
                        = DataSlice(tempAlgParams.buffInfo.inputPtr, txBaseOff, subSize, subSize / dataTypeSize_);
                    DataSlice txDstSlice = DataSlice(
                        remoteBuffAddr, supportSymmetricMemAccess_ ? txBaseOff : (slotSendBase + subSendOff), subSize,
                        subSize / dataTypeSize_);

                    rxSrcSlices.push_back(rxSrcSlice);
                    rxDstSlices.push_back(rxDstSlice);
                    txSrcSlices.push_back(txSrcSlice);
                    txDstSlices.push_back(txDstSlice);
                }
                SendRecvReduceInfo sendRecvReduceInfo{
                    {linkSend, linkRecv},
                    {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}},
                    dataType_,
                    reduceOp_};

                // 对称与非对称路径统一推模式(上游 adaptive symm): WriteReduce 只消费 tx 切片,
                // rx 列表仅占位不被消费; 对称写远端 input 窗口, 非对称写远端 cclBuff
                CHK_PRT_RET(
                    SendRecvBatchWriteReduce(sendRecvReduceInfo, threads[queIdx]),
                    HCCL_ERROR("[InsTempReduceScatterMesh1DMeshChunk][DoMeshChunk] SendRecvBatchWriteReduce failed"),
                    HcclResult::HCCL_E_INTERNAL);
            }

            sliceSendOffset_ += sliceSize[i];
            if (templateRankSize_ > rankNum && i < (templateRankSize_ - rankNum)) {
                sliceRecvOffset_ -= sliceSize[templateRankSize_ - tempNum - i];
            }
        }
        if (threadNum_ > 1 && stepIdx < (templateRankSize_ - rankNum)) {
            std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
            NotifyIdxSubToMainInMeshChunk(notifyIdxSubToMain_);
            CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
            NotifyIdxMainToSubInMeshChunk(notifyIdxMainToSub_);
            CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunk::PostCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    // 如果是单算子模式, 并且是最后一步算子，需要将数据从 scratch 拷贝到 userOut
    HCCL_INFO("[InsTempReduceScatterMesh1DMeshChunk][PostCopy], copy from scratch to userOut");
    u32 myAlgRank;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    CHK_PRT_RET(
        threads.empty(), HCCL_ERROR("[InsTempReduceScatterMesh1DMeshChunk][PostCopy] threads is empty"), HCCL_E_PARA);
    // 先把本卡的数据从input搬运到output
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; repeatIdx++) {
        u64 srcOff = ScratchSlotBaseOff(tempAlgParams, repeatIdx, myAlgRank);
        u64 dstOff = tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride
                     + myAlgRank * tempAlgParams.outputSliceStride;
        // 序列框内场景 outputPtr == hcclBuff.addr 且偏移相同时跳过自拷贝
        if (tempAlgParams.buffInfo.hcclBuff.addr == tempAlgParams.buffInfo.outputPtr && srcOff == dstOff) {
            continue;
        }
        DataSlice myRankSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOff, processSize_);
        DataSlice outputSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, dstOff, processSize_);
        CHK_RET(LocalCopy(threads[0], myRankSlice, outputSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempReduceScatterMesh1DMeshChunk::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = static_cast<u32>(GetThreadNum());
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempReduceScatterMesh1DMeshChunk::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = static_cast<u32>(GetThreadNum());
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

void InsTempReduceScatterMesh1DMeshChunk::NotifyIdxMainToSubInMeshChunk(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = static_cast<u32>(GetThreadNum());
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(1);
    }
}

void InsTempReduceScatterMesh1DMeshChunk::NotifyIdxSubToMainInMeshChunk(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = static_cast<u32>(GetThreadNum());
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx + notifyNum);
    }
}
} // namespace ops_hccl

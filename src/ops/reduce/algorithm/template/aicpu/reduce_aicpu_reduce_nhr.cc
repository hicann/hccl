/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_aicpu_reduce_nhr.h"
#include "cost_model.h"

namespace ops_hccl {
std::vector<CostModelParam> ReduceAicpuReduceNHR::CalcCostCoeff(CalcCostCoeffParam param)
{
    // 口径: cost 乘子 dataSize 为单 rank 全量输入,
    // dataRatio×rankSize×dataSize 为本模板单 rank 输入全量
    // (sole 整域 dataRatio=1/rankSize, dataRatio×rankSize=1.0)
    //   传输项 A: 单轮整域 NHR AllGather, 每 rank 各持 1 个全量 slot,
    //             gather 后发送 (rankSize-1) 份全量
    //   B: 入拷贝(PreCopy 全量入 cclBuff) + 出拷贝(全量),
    //      输入/输出已在 cclBuff 的场景由入参 BufferType 跳过
    //   C: 4×NHR 步收发 + 拷贝/规约任务(入拷贝 1 + 出拷贝 1 + 规约 rankSize-1)
    //      + 规约前一次强制同步
    //      (折算 SYNC_EQUIV_TASKS 个 2us task)
    //      (64bit/PROD 的规约吞吐后续在 B 中按 count×单 count 耗时折算, 此处不建模)
    CommTopo netType = param.netType;
    int portNum = 1;
    if (!param.portNum.empty()) {
        portNum = (param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      (param.portNum[0] + param.portNum[1]) :
                      param.portNum[0];
    }
    if (portNum <= 0) {
        portNum = 1;
    }
    float A = 0.0f;
    CostModelManager::Global()->CalcNHRParams(
        param.dataRatio * param.rankSize, netType, portNum, param.rankSize, A, param.isPod);

    // B: 入拷贝/出拷贝各 1 份全量
    float B1 = 0.0f;
    float B2 = 0.0f;
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio * param.rankSize, EngineType::AICPU, B1);
    }
    if (param.outputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio * param.rankSize, EngineType::AICPU, B2);
    }
    const float B = B1 + B2;

    // AICPU 规约工作量: RunReduce 串行规约 (rankSize-1) 次全量
    // (每份 dataRatio×rankSize×dataSize 字节),
    // 64bit/PROD 时由框架 per-call 按 count×1.2ns 折算进 B
    const float reduceCopies
        = static_cast<float>(param.rankSize - 1) * param.dataRatio * static_cast<float>(param.rankSize);

    constexpr int SYNC_EQUIV_TASKS = 50;
    int nSteps = static_cast<int>(GetNHRStepNum(param.rankSize));
    int taskNum = static_cast<int>(param.rankSize) - 1 + 4 * nSteps + 2;
    float C = 0.0f;
    CostModelManager::Global()->CalcLatencyParams(taskNum + SYNC_EQUIV_TASKS, EngineType::AICPU, C);

    // D: 任务下发固定项(不随数据量变化)
    float D = 1e-6f * taskNum;

    HCCL_INFO(
        "[ReduceAicpuReduceNHR][CalcCostCoeff] algName[%s] rankSize[%u] "
        "A=%e B=%e(copy=%e) C=%e D=%e reduceCopies=%f.",
        param.algName != nullptr ? param.algName : "(null)", param.rankSize, A, B, B1 + B2, C, D, reduceCopies);
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D, reduceCopies});
    return params;
}

ReduceAicpuReduceNHR::ReduceAicpuReduceNHR(
    const OpParam& param,
    const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

ReduceAicpuReduceNHR::~ReduceAicpuReduceNHR() {}

void ReduceAicpuReduceNHR::SetRoot(u32 root)
{
    (void)root;
    return;
}

HcclResult ReduceAicpuReduceNHR::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    u32 threadNum = 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO(
        "[ReduceAicpuReduceNHR][ReduceMeshNHR][CalcRes]slaveThreadNum[%u] notifyNumPerThread[%zu] "
        "notifyNumOnMainThread[%u]"
        " level1Channels[%zu] .",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread.size(),
        resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HCCL_SUCCESS;
}

u64 ReduceAicpuReduceNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult ReduceAicpuReduceNHR::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[ReduceAicpuReduceNHR] rank[%d] KernelRun start", myRank_);
    // 处理数据量为0的场景
    CHK_PRT_RET(
        tempAlgParams.sliceSize == 0, HCCL_INFO("[ReduceAicpuReduceNHR] sliceSize is 0, no need to process"),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        templateRankSize_ == 0, HCCL_ERROR("[ReduceAicpuReduceNHR] rankSize is 0"), HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(root_ == UINT32_MAX, HCCL_ERROR("[ReduceAicpuReduceNHR] root is invalid"), HcclResult::HCCL_E_INTERNAL);

    thread_ = templateResource.threads.at(0);
    buffInfo_ = tempAlgParams.buffInfo;

    bool isPcieProtocol = IsPcieProtocol(templateResource.channels); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocol;                                     // 是否使用Read模式
    HCCL_DEBUG("[ReduceAicpuReduceNHR] Use Dma Read[%d]", isDmaRead_);

    CHK_RET(GetAlgRank(myRank_, myIdx_));
    CHK_PRT_RET(
        myIdx_ >= templateRankSize_,
        HCCL_ERROR(
            "[ReduceAicpuReduceNHR] rank idx[%u] in virtRankMap is invalid, it should be less than rankSize[%u]",
            myIdx_, templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    HCCL_INFO(
        "[ReduceAicpuReduceNHR][KernelRun] sliceSize: %u, count_: %u, typeSize: %u", tempAlgParams.sliceSize, count_,
        DATATYPE_SIZE_TABLE[dataType_]);

    const std::map<u32, std::vector<ChannelInfo>>& channels = templateResource.channels;
    HCCL_DEBUG(
        "[ReduceAicpuReduceNHR][Kernel Run] myRank_[%u], channels.size():%zu, channels first[%u]", myRank_,
        channels.size(), channels.begin()->first);

    // 1. 切片
    CHK_RET(CalcSlice(tempAlgParams.sliceSize));

    // 2. PreCopy (OPBASE 模式下将 userIn -> ccl)
    CHK_RET(PreCopy(tempAlgParams));

    // 3. AllGather 阶段
    CHK_RET(RunGather(channels));

    // 4. LocalReduce 阶段
    CHK_RET(RunReduce(channels, tempAlgParams, param.algTag));

    HCCL_INFO("[ReduceAicpuReduceNHR] rank[%d] KernelRun finished", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 将数据按照rank切分为chunk 块，给后续的reduce操作使用
 * param: dataSize: 待处理的输入数据大小
 * return: sliceInfoVec: 存储数据切分结果
 * return: HcclResult
 */
HcclResult ReduceAicpuReduceNHR::CalcSlice(u64 chunkSize)
{
    HCCL_INFO("[ReduceAicpuReduceNHR] rank[%d] CalcSlice start", chunkSize);
    // 按 rank 切分数据（与 AllReduceAicpuReduceNHR 保持一致）
    sliceInfoVec_ = RankSliceInfo(templateRankSize_);

    u64 unitAlignSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 dataSize = chunkSize * templateRankSize_;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64 currChunkSize = std::min<u64>(dataSize - accumOff, chunkSize);
        sliceInfoVec_[rankIdx].emplace_back(SliceInfo{accumOff, currChunkSize});
        HCCL_DEBUG(
            "[ReduceAicpuReduceNHR] rankIdx [%d] CalcSlice accumOff[%u], currChunkSize[%u]", rankIdx, accumOff,
            currChunkSize);
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec_[templateRankSize_ - 1][0].offset + sliceInfoVec_[templateRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR(
            "[ReduceAicpuReduceNHR] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!", chunkSize, myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceAicpuReduceNHR::PreCopy(const TemplateDataParams& tempAlgParams)
{
    HCCL_INFO("[ReduceAicpuReduceNHR][PreCopy] start");
    const BuffInfo& buffInfo = tempAlgParams.buffInfo;
    const u64 sliceSize = tempAlgParams.sliceSize;

    const DataSlice srcSlice(buffInfo.inputPtr, buffInfo.inBuffBaseOff, sliceSize);
    const DataSlice dstSlice(buffInfo.hcclBuff.addr, buffInfo.hcclBuffBaseOff + myIdx_ * sliceSize, sliceSize);
    CHK_RET(static_cast<HcclResult>(LocalCopy(thread_, srcSlice, dstSlice)));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceAicpuReduceNHR::RunGather(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    HCCL_INFO("[ReduceAicpuReduceNHR][RunGather] start");
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));

        const ChannelInfo& channelRecv = channels.at(subCommRanks_.at(0).at(stepInfo.fromRank)).at(0);
        const ChannelInfo& channelSend = channels.at(subCommRanks_.at(0).at(stepInfo.toRank)).at(0);

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            u64 txOffset = sliceInfoVec_[stepInfo.txSliceIdxs[i]][0].offset + buffInfo_.hcclBuffBaseOff;
            u64 txSize = sliceInfoVec_[stepInfo.txSliceIdxs[i]][0].size;
            u64 rxOffset = sliceInfoVec_[stepInfo.rxSliceIdxs[i]][0].offset + buffInfo_.hcclBuffBaseOff;
            u64 rxSize = sliceInfoVec_[stepInfo.rxSliceIdxs[i]][0].size;

            txSrcSlices.push_back(DataSlice(buffInfo_.hcclBuff.addr, txOffset, txSize, txSize / dataTypeSize));
            txDstSlices.push_back(DataSlice(channelSend.remoteCclMem.addr, txOffset, txSize, txSize / dataTypeSize));
            rxSrcSlices.push_back(DataSlice(channelRecv.remoteCclMem.addr, rxOffset, rxSize, rxSize / dataTypeSize));
            rxDstSlices.push_back(DataSlice(buffInfo_.hcclBuff.addr, rxOffset, rxSize, rxSize / dataTypeSize));
        }

        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});

        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, thread_), HCCL_ERROR("[ReduceAicpuReduceNHR] RunGather send/recv failed"),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvWrite(sendRecvInfo, thread_), HCCL_ERROR("[ReduceAicpuReduceNHR] RunGather send/recv failed"),
                HcclResult::HCCL_E_INTERNAL);
        }
    }

    HCCL_INFO("[ReduceAicpuReduceNHR][RunGather] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceAicpuReduceNHR::RunReduce(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const TemplateDataParams& tempAlgParams,
    const std::string& algTag)
{
    (void)channels;
    HCCL_INFO("[ReduceAicpuReduceNHR][RunReduce] start");
    if (myRank_ != root_) {
        return HCCL_SUCCESS;
    }

    const BuffInfo& buffInfo = tempAlgParams.buffInfo;
    const u64 sliceSize = tempAlgParams.sliceSize;
    const DataSlice srcSlice(buffInfo.hcclBuff.addr, buffInfo.hcclBuffBaseOff, sliceSize, count_);
    const DataSlice dstSlice(buffInfo.outputPtr, buffInfo.outBuffBaseOff, sliceSize, count_);
    CHK_RET(static_cast<HcclResult>(LocalCopy(thread_, srcSlice, dstSlice)));

    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(algTag.c_str())));
    CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(algTag.c_str())));
    CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread_, CUSTOM_TIMEOUT)));

    for (u32 idx = 1; idx < subCommRanks_.at(0).size(); ++idx) {
        const DataSlice srcSlice(buffInfo.hcclBuff.addr, buffInfo.hcclBuffBaseOff + sliceSize * idx, sliceSize, count_);

        CHK_RET(static_cast<HcclResult>(LocalReduce(thread_, srcSlice, dstSlice, dataType_, reduceOp_)));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceAicpuReduceNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo)
{
    u32 rankIdx = myIdx_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = rankIdx;
    u32 rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG(
            "[ReduceAicpuReduceNHR][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

//  计算每轮收发的对端以及slice编号
HcclResult ReduceAicpuReduceNHR::GetStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList)
{
    // 将本 rank 号转换成算法使用的索引号
    u32 rankIdx = myIdx_;
    stepInfoList.clear();

    u32 nSteps = GetNHRStepNum(templateRankSize_);
    stepInfoList.resize(nSteps);
    for (u32 step = 0; step < nSteps; step++) {
        // 计算通信对象
        u32 deltaRank = 1 << step;
        u32 sendTo = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
        u32 recvFrom = (rankIdx + deltaRank) % templateRankSize_;

        // 数据份数和数据编号增量
        u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
        u32 deltaSliceIndex = 1 << (step + 1);
        u32 txSliceIdx = sendTo;
        u32 rxSliceIdx = rankIdx;

        AicpuNHRStepInfo& currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.myRank = rankIdx;
        currStepInfo.nSlices = nSlices;
        currStepInfo.toRank = sendTo;
        currStepInfo.fromRank = recvFrom;

        // 计算本rank在每轮收/发中的slice编号
        currStepInfo.txSliceIdxs.reserve(nSlices);
        currStepInfo.rxSliceIdxs.reserve(nSlices);
        for (u32 i = 0; i < nSlices; i++) {
            currStepInfo.txSliceIdxs.push_back(txSliceIdx);
            currStepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            HCCL_DEBUG(
                "[ReduceAicpuReduceNHR][GetStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx,
                rxSliceIdx);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult ReduceAicpuReduceNHR::GetAlgRank(u32 rank, u32& algRank) const
{
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty(),
        HCCL_ERROR("[ReduceAicpuReduceNHR] subCommRanks_[0] is empty"), HcclResult::HCCL_E_INTERNAL);
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), rank);
    CHK_PRT_RET(
        iter == subCommRanks_[0].end(),
        HCCL_ERROR("[ReduceAicpuReduceNHR] Cannot find rank[%u] in subCommRanks_[0]", rank),
        HcclResult::HCCL_E_INTERNAL);
    algRank = static_cast<u32>(std::distance(subCommRanks_[0].begin(), iter));
    return HcclResult::HCCL_SUCCESS;
}

void ReduceAicpuReduceNHR::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub) { (void)notifyIdxMainToSub; }

void ReduceAicpuReduceNHR::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) { (void)notifyIdxSubToMain; }

u64 ReduceAicpuReduceNHR::GetThreadNum() const { return 1; }

} // namespace ops_hccl

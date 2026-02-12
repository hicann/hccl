/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_nhr.h"

namespace ops_hccl {
InsTempReduceNHR::InsTempReduceNHR(const OpParam &param,
    const u32 rankId,  // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempReduceNHR::~InsTempReduceNHR()
{}

HcclResult InsTempReduceNHR::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfo *topoInfo, AlgResourceRequest &resourceRequest)
{
    u32 threadNum = 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO("[InsTempReduceMeshNHR][CalcRes]slaveThreadNum[%u] notifyNumPerThread[%u] notifyNumOnMainThread[%u]"
              " level1Channels[%u] .",
        resourceRequest.slaveThreadNum,
        resourceRequest.notifyNumPerThread.size(),
        resourceRequest.notifyNumOnMainThread,
        level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempReduceNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}

HcclResult InsTempReduceNHR::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, const TemplateResource &templateResource)
{
    HCCL_INFO("[InsTempReduceNHR] rank[%d] KernelRun start", myRank_);
    // 处理数据量为0的场景
    CHK_PRT_RET(
        tempAlgParams.sliceSize == 0, HCCL_INFO("[InsTempReduceNHR] sliceSize is 0, no need to process"), HCCL_SUCCESS);

    CHK_PRT_RET(templateRankSize_ == 0, HCCL_ERROR("[InsTempReduceNHR] rankSize is 0"), HcclResult::HCCL_E_INTERNAL);

    CHK_PRT_RET(root_ == UINT32_MAX, HCCL_ERROR("[InsTempReduceNHR] root is invalid"), HcclResult::HCCL_E_INTERNAL);

    thread_ = templateResource.threads.at(0);
    buffInfo_ = tempAlgParams.buffInfo;

    myIdx_ = getMyAlgRank();
    CHK_PRT_RET(myIdx_ >= templateRankSize_,
        HCCL_ERROR("[InsTempReduceNHR] rank idx[%u] in virtRankMap is invalid, it should be less than rankSize[%u]",
            myIdx_,
            templateRankSize_),
        HcclResult::HCCL_E_INTERNAL);
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;
    HCCL_INFO("[KernelRun] sliceSize: %u, count_: %u, typeSize: %u",
        tempAlgParams.sliceSize,
        count_,
        DATATYPE_SIZE_TABLE[dataType_]);

    const std::map<u32, std::vector<ChannelInfo>> &channels = templateResource.channels;
    HCCL_DEBUG("[Kernel Run] myRank_[%u], channels.size():%u, channels first[%u]",
        myRank_,
        channels.size(),
        channels.begin()->first);

    // 1. 切片
    CHK_RET(CalcSlice(tempAlgParams.sliceSize));

    // 2. PreCopy (OPBASE 模式下将 userIn -> ccl)
    CHK_RET(PreCopy(tempAlgParams));

    // 3. ReduceScatter 阶段 (pairwise reduce)
    CHK_RET(RunReduceScatter(channels));

    // 4. AllGather 阶段
    CHK_RET(RunGather(channels));

    // 5. PostCopy (OPBASE 且在 root 上将 ccl -> userOut)
    CHK_RET(PostCopy(tempAlgParams));

    HCCL_INFO("[InsTempReduceNHR] rank[%d] KernelRun finished", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

/*
 * Desc: 将数据按照rank切分为chunk 块，给后续的reduce操作使用
 * param: dataSize: 待处理的输入数据大小
 * return: sliceInfoVec: 存储数据切分结果
 * return: HcclResult
 */
HcclResult InsTempReduceNHR::CalcSlice(u64 dataSize)
{
    HCCL_INFO("[InsTempReduceNHR] rank[%d] CalcSlice start", dataSize);
    // 按 rank 切分数据（与 AllReduceNHR 保持一致）
    sliceInfoVec_ = RankSliceInfo(templateRankSize_);

    u64 unitAlignSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 chunkSize = RoundUpWithDivisor(dataSize, (templateRankSize_ * unitAlignSize)) / templateRankSize_;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64 currChunkSize = std::min<u64>(dataSize - accumOff, chunkSize);
        sliceInfoVec_[rankIdx].emplace_back(SliceInfo{accumOff, currChunkSize});
        HCCL_DEBUG("[InsTempReduceNHR] rankIdx [%d] CalcSlice accumOff[%u], currChunkSize[%u]",
            rankIdx,
            accumOff,
            currChunkSize);
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec_[templateRankSize_ - 1][0].offset + sliceInfoVec_[templateRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR("[InsTempReduceNHR] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!", chunkSize, myRank_),
        HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceNHR::PreCopy(const TemplateDataParams &tempAlgParams)
{
    const BuffInfo &buffInfo = tempAlgParams.buffInfo;
    const u64 sliceSize = tempAlgParams.sliceSize;
    // 单算子模式，需要先将数据拷贝到cclBuffer
    reduceInBuffType_ = BufferType::HCCL_BUFFER;
    reduceInBuffBaseOff_ = buffInfo.hcclBuffBaseOff;

    if (buffInfo.inBuffType != BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempReduceNHR][PreCopy] Opbase copy from userIn to scratchBuffer");
        void *const src = static_cast<void *>(static_cast<u8 *>(buffInfo.inputPtr) + buffInfo.inBuffBaseOff);
        void *const dst = static_cast<void *>(static_cast<u8 *>(buffInfo.hcclBuff.addr) + buffInfo.hcclBuffBaseOff);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, dst, src, sliceSize)));
        reduceInBuffBaseOff_ = buffInfo.hcclBuffBaseOff;
    } else {
        HCCL_INFO("[InsTempReduceNHR][PreCopy] skip precopy");
    }

    reduceOutBuffType_ = buffInfo.outBuffType;
    reduceOutBuffBaseOff_ = buffInfo.outBuffBaseOff;

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceNHR::RunReduceScatter(const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    HCCL_INFO("[InsTempReduceNHR][RunReduceScatter] start");
    std::vector<AicpuNHRStepInfo> stepInfoList;
    CHK_RET(GetStepInfoList(stepInfoList));

    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (const auto &stepInfo : stepInfoList) {
        HCCL_DEBUG("[InsTempReduceNHR][RunReduceScatter] step[%u], myRank[%u], toRank[%u], fromRank[%u], nSlices[%u].",
            stepInfo.step,
            stepInfo.myRank,
            stepInfo.toRank,
            stepInfo.fromRank,
            stepInfo.nSlices);

        const ChannelInfo &channelRecv = channels.at(subCommRanks_.at(0).at(stepInfo.fromRank)).at(0);
        const ChannelInfo &channelSend = channels.at(subCommRanks_.at(0).at(stepInfo.toRank)).at(0);

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        // 在 inBuff 上进行 ReduceScatter 操作
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            u64 txOffset = sliceInfoVec_[stepInfo.txSliceIdxs[i]][0].offset + buffInfo_.hcclBuffBaseOff;
            u64 txSize = sliceInfoVec_[stepInfo.txSliceIdxs[i]][0].size;
            u64 rxOffset = sliceInfoVec_[stepInfo.rxSliceIdxs[i]][0].offset + buffInfo_.hcclBuffBaseOff;
            u64 rxSize = sliceInfoVec_[stepInfo.rxSliceIdxs[i]][0].size;
            DataSlice txSrcSlice = DataSlice(buffInfo_.hcclBuff.addr, txOffset, txSize, txSize / dataTypeSize);
            DataSlice txDstSlice = DataSlice(channelSend.remoteCclMem.addr, txOffset, txSize, txSize / dataTypeSize);
            DataSlice rxSrcSlice = DataSlice(buffInfo_.hcclBuff.addr, rxOffset, rxSize, rxSize / dataTypeSize);
            DataSlice rxDstSlice = DataSlice(buffInfo_.hcclBuff.addr, rxOffset, rxSize, rxSize / dataTypeSize);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);
            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);
        }
        SendRecvReduceInfo sendRecvReduceInfo{
            {channelSend, channelRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_, reduceOp_};

        CHK_PRT_RET(SendRecvWriteReduce(sendRecvReduceInfo, thread_),
            HCCL_ERROR("[InsTempReduceNHR] RunReduceScatter SendRecvReduce failed"),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceNHR::RunGather(const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));

        const ChannelInfo &channelRecv = channels.at(subCommRanks_.at(0).at(stepInfo.fromRank)).at(0);
        const ChannelInfo &channelSend = channels.at(subCommRanks_.at(0).at(stepInfo.toRank)).at(0);

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
            rxSrcSlices.push_back(DataSlice(buffInfo_.hcclBuff.addr, rxOffset, rxSize, rxSize / dataTypeSize));
            rxDstSlices.push_back(DataSlice(buffInfo_.hcclBuff.addr, rxOffset, rxSize, rxSize / dataTypeSize));
        }

        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});

        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
        CHK_PRT_RET(SendRecvWrite(sendRecvInfo, thread_),
            HCCL_ERROR("[InsTempReduceNHR] RunGather send/recv failed"),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceNHR::PostCopy(const TemplateDataParams &tempAlgParams)
{
    if (reduceOutBuffType_ == BufferType::HCCL_BUFFER) {
        HCCL_DEBUG("[InsTempReduceNHR][PostCopy] skip postcopy rank[%d]", myRank_);
        // 希望输出结果在HCCL_BUFFER上，NHR算法做完已经在HCCL_BUFFER上，无需再进行后拷贝
        return HCCL_SUCCESS;
    }
    // PostCopy 仅在 OPBASE 并且在 root 上执行（root 收到完整结果后写回用户 out）
    const BuffInfo &buffInfo = tempAlgParams.buffInfo;
    if (myRank_ != root_) {
        HCCL_DEBUG("[InsTempReduceNHR][PostCopy] not root, skip postcopy rank[%d]", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_INFO("[InsTempReduceNHR][PostCopy] Opbase root copy from scratchBuffer to userOut");
    void *const srcBasePtr =
        (reduceInBuffType_ == BufferType::HCCL_BUFFER) ? buffInfo.hcclBuff.addr : buffInfo.inputPtr;
    void *const dstBasePtr = buffInfo.outputPtr;
    void *const src = static_cast<void *>(static_cast<u8 *>(srcBasePtr) + reduceInBuffBaseOff_);
    void *const dst = static_cast<void *>(static_cast<u8 *>(dstBasePtr) + reduceOutBuffBaseOff_);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, dst, src, tempAlgParams.sliceSize)));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo)
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

        HCCL_DEBUG("[InsTempReduceNHR][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

//  计算每轮收发的对端以及slice编号
HcclResult InsTempReduceNHR::GetStepInfoList(std::vector<AicpuNHRStepInfo> &stepInfoList)
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

        AicpuNHRStepInfo &currStepInfo = stepInfoList[step];
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
                "[InsTempReduceNHR][GetStepInfoList] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

std::pair<std::vector<DataSlice>, std::vector<DataSlice>> InsTempReduceNHR::getTxRxSlices(
    const AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels)
{
    std::vector<DataSlice> txSlices;
    std::vector<DataSlice> rxSlices;

    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (u32 i = 0; i < stepInfo.nSlices; i++) {
        u32 txRank = stepInfo.txSliceIdxs[i];
        u32 rxRank = stepInfo.rxSliceIdxs[i];
        u64 txOffset = sliceInfoVec_.at(txRank).at(0).offset + buffInfo_.hcclBuffBaseOff;
        u64 txSize = sliceInfoVec_.at(txRank).at(0).size;
        u64 rxOffset = sliceInfoVec_.at(rxRank).at(0).offset + buffInfo_.hcclBuffBaseOff;
        u64 rxSize = sliceInfoVec_.at(rxRank).at(0).size;
        void *const txAddr =
            (txRank == myRank_) ? buffInfo_.hcclBuff.addr : channels.at(txRank).at(0).remoteCclMem.addr;
        void *const rxAddr =
            (rxRank == myRank_) ? buffInfo_.hcclBuff.addr : channels.at(rxRank).at(0).remoteCclMem.addr;
        HCCL_DEBUG("[getTxRxSlices] myRank:%u, i:%u, txRank:%u, txOffset:%u, txSize:%u, rxRank:%u,rxOffset:%u, "
                   "rxSize:%u, txAddr:[%#llx], rxAddr:[%#llx]",
            myRank_,
            i,
            txRank,
            txOffset,
            txSize,
            rxRank,
            rxOffset,
            rxSize,
            txAddr,
            rxAddr);

        // 默认reduceInBuffType_为SCRATCH的场景
        txSlices.emplace_back(txAddr, txOffset, txSize, txSize / dataTypeSize);
        rxSlices.emplace_back(rxAddr, rxOffset, rxSize, rxSize / dataTypeSize);
    }
    return std::make_pair(txSlices, rxSlices);
}

u32 InsTempReduceNHR::getMyAlgRank() const
{
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter == subCommRanks_[0].end()) {
        throw std::runtime_error("Cannot find myRank = " + std::to_string(myRank_) + " in subCommRanks_[0]");
    }
    return static_cast<u32>(std::distance(subCommRanks_[0].begin(), iter));
}

void InsTempReduceNHR::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{}

void InsTempReduceNHR::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{}

u64 InsTempReduceNHR::GetThreadNum()
{
    return 1;
}

}  // namespace ops_hccl
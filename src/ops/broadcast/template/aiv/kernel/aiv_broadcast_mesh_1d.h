/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "aiv_communication_base_v2.h"
 
using namespace AscendC;
 
// todo 简化参数
 
class AivBroadcastMesh1D : public AivCommBase {
public:
    __aicore__ inline AivBroadcastMesh1D() {}

    template<typename T>
    __aicore__ inline void Process(uint64_t curCount, uint64_t sliceId, uint64_t stride);

    template<typename T>
    __aicore__ inline void ProcessBigData(uint64_t curCount, uint64_t sliceId);
private:
    __aicore__ inline void CalculateOffsetAndCount(uint64_t totalData, uint64_t index, 
                                                   uint64_t totalParts, uint64_t &offset, uint64_t &count);
};

__aicore__ inline void AivBroadcastMesh1D::CalculateOffsetAndCount(uint64_t totalData, uint64_t index, 
                                               uint64_t totalParts, uint64_t &offset, uint64_t &count)
{
    if (totalParts == 0) {
        offset = 0;
        count = 0;
        return;
    }
    uint64_t dataPerPart = totalData / totalParts;
    uint64_t remainder = totalData % totalParts;
    if (index < remainder) {
        offset = index * dataPerPart + index;
        count = dataPerPart + 1;
    } else {
        offset = index * dataPerPart + remainder;
        count = dataPerPart;
    }
}
 
template<typename T>
__aicore__ inline void AivBroadcastMesh1D::Process(uint64_t curCount, uint64_t sliceId, uint64_t stride)
{
    curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    uint64_t dataTypeSize = sizeof(T);
    uint64_t coreNumPerRank = rankSize_ == 8 ? 2 : 1;
    uint64_t curStageCoreNum = coreNumPerRank * rankSize_;
    if (blockIdx_ >= curStageCoreNum) {
        return;
    }

    uint64_t peerRank = blockIdx_ / coreNumPerRank;
    uint64_t offsetPerCore = curCount / curStageCoreNum * dataTypeSize;
    uint64_t dataOffset = offsetPerCore * blockIdx_;
    uint64_t countPerCore = blockIdx_ == curStageCoreNum - 1 ? curCount - (curStageCoreNum - 1) * (curCount / curStageCoreNum)
                                    : curCount / curStageCoreNum;
    uint64_t flag_offset = blockIdx_;
    __gm__ T *inputGM = (__gm__ T *)(input_ + dataOffset);
    __gm__ T *cclGM = (__gm__ T *)(GM_IN[peerRank] + dataOffset);
    // scatter
    if (rank_ == root_) {
        CpGM2GM(cclGM, inputGM, countPerCore);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < rankSize_; i++) {
            Record(peerRank, blockIdx_ % coreNumPerRank + i * coreNumPerRank, curTag_);
        }
    }
    // allgather
    WaitFlag(rank_, blockIdx_, curTag_);
    Record(peerRank, curStageCoreNum + blockIdx_ % coreNumPerRank + rank_ * coreNumPerRank, curTag_);
    WaitFlag(rank_, curStageCoreNum + blockIdx_ % coreNumPerRank + peerRank * coreNumPerRank, curTag_);
    CpGM2GM(inputGM, cclGM, countPerCore);
    PipeBarrier<PIPE_ALL>();
}

template<typename T>
__aicore__ inline void AivBroadcastMesh1D::ProcessBigData(uint64_t curCount, uint64_t sliceId)
{
    curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    // root节点先用全量核去写本卡cclBuffer，这里的每个核要对应多个flag，让其他卡可以用全量核来读 
    // 然后其他卡用全量核去读数据 
    // 最后做allgather
    uint64_t curStageCoreNum = numBlocks_ / rankSize_ * rankSize_;
    if (blockIdx_ >= curStageCoreNum) {
        return;
    }

    uint64_t coreNumPerRank = curStageCoreNum / rankSize_;
    uint64_t targetRank = blockIdx_ / coreNumPerRank;
    uint64_t coreIndex = (blockIdx_ - (targetRank * coreNumPerRank)) % coreNumPerRank;
    uint64_t flag_offset = 0;

    // 先把数据按照rankSize 切分
    uint64_t rankInnerDispls = 0;
    uint64_t targetRankCurCount = 0;
    CalculateOffsetAndCount(curCount, targetRank, rankSize_, rankInnerDispls, targetRankCurCount);

    // 给每个核划分数据
    uint64_t innerDispls = 0;
    uint64_t sendCurCount = 0;
    CalculateOffsetAndCount(targetRankCurCount, coreIndex, coreNumPerRank, innerDispls, sendCurCount);

    // root 开始本卡搬运数据:这里是全量卡都去搬比较好，还是就用rankSize的卡去搬
    uint64_t sendInputOffset = input_ + (rankInnerDispls + innerDispls) * sizeof(T);
    uint64_t sendCclInOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (rankInnerDispls + innerDispls) * sizeof(T);
    if (rank_ == root_) {
        CpGM2GM((__gm__ T *)sendCclInOffset, (__gm__ T *)sendInputOffset, sendCurCount);
        PipeBarrier<PIPE_ALL>();
        // targetRankCurCount这么多的数据量，有coreNumPerRank去写，但是有coreNumPerRank * rankSize的核去读，所以一个核要写rankSize个flag
        for (uint64_t i = 0; i < rankSize_; i++) {
            Record(root_, blockIdx_ * rankSize_ + i, curTag_);
        }
    }

    // 现在除了root节点，其他卡要用全量核去拿root卡上的数据
    uint64_t rankInnerDisplsStage1 = 0;
    uint64_t targetRankCurCountStage1 = 0;
    CalculateOffsetAndCount(curCount, rank_, rankSize_, rankInnerDisplsStage1, targetRankCurCountStage1);

    // 给每rankSize个核划分数据
    uint64_t rankSizeCoreDataIndex = blockIdx_ / rankSize_;
    uint64_t rankSizeCoreInnerDispls = 0;
    uint64_t rankSizeCoreSendCurCount = 0;
    CalculateOffsetAndCount(targetRankCurCountStage1, rankSizeCoreDataIndex, 
                            coreNumPerRank, rankSizeCoreInnerDispls, rankSizeCoreSendCurCount);

    // 给每个核划分数据
    uint64_t coreIndexStage1 = (blockIdx_ - (rankSizeCoreDataIndex * rankSize_)) % rankSize_;
    uint64_t innerDisplsStage1 = 0;
    uint64_t sendCurCountStage1 = 0;
    CalculateOffsetAndCount(rankSizeCoreSendCurCount, coreIndexStage1, 
                            rankSize_, innerDisplsStage1, sendCurCountStage1);

    // 每个核开始去读数据
    uint64_t recvCclInOffset = reinterpret_cast<uint64_t>(GM_IN[root_]) + (rankInnerDisplsStage1 + rankSizeCoreInnerDispls + innerDisplsStage1) * sizeof(T);
    uint64_t recvCclOutOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (rankInnerDisplsStage1 + rankSizeCoreInnerDispls + innerDisplsStage1) * sizeof(T);
    uint64_t flagTotal = rankSize_ * curStageCoreNum;
    flag_offset = rank_ * coreNumPerRank * rankSize_ + rankSizeCoreDataIndex * rankSize_ + coreIndexStage1;
    WaitFlag(root_, flag_offset, curTag_);
    if (rank_ != root_) {
        CpGM2GM((__gm__ T *)recvCclOutOffset, (__gm__ T *)recvCclInOffset, sendCurCountStage1);
        PipeBarrier<PIPE_ALL>();
        Record(rank_, flag_offset, curTag_);
    }
    if (coreIndexStage1 == 0) {
        for (uint64_t i = 0; i < rankSize_; i++) {
            uint64_t flag_offset_w = rank_ * coreNumPerRank * rankSize_ + rankSizeCoreDataIndex * rankSize_ + i;
            WaitFlag(rank_, flag_offset_w, curTag_);
        }
        for (uint64_t i = 0; i < rankSize_; i++) {
            Record(i, flagTotal + rank_ + rankSizeCoreDataIndex * rankSize_, curTag_);
        }
    }

    // 最后所有的卡去做allgather,每个卡要读对面rankSize个flag
    uint64_t gatherSrcOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (rankInnerDispls + innerDispls) * sizeof(T);
    uint64_t ouputOffset = input_ + (rankInnerDispls + innerDispls) * sizeof(T);
    if ((rank_ != root_) && (sendCurCount > 0)) {
        // 每块数据要去等rankSize个flag
        WaitFlag(rank_, flagTotal + targetRank + coreIndex * rankSize_, curTag_);
        CpGM2GM((__gm__ T *)ouputOffset, (__gm__ T *)gatherSrcOffset, sendCurCount);
        PipeBarrier<PIPE_ALL>();
    }
}
 
template<typename T>
__aicore__ inline void AivBroadcastV2Mesh1D(KERNEL_ARGS_DEF)
{
    AivBroadcastMesh1D op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    if (len * sizeof(T) >= DATA_LIMIT && rankSize != 8) {
        op.ProcessBigData<T>(len, sliceId);
    } else {
        op.Process<T>(len, sliceId, inputSliceStride);
    }
    op.BarrierAll();
}
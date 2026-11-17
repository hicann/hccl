/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_GATHER_MESH_1D_WRITE_H
#define AIV_ALL_GATHER_MESH_1D_WRITE_H

#include "aiv_communication_base_v2.h"

using namespace AscendC;

// AllGather Mesh 1D写语义：发送到对端按源rank划分的CCL槽，再由本端consumer回拷output。
// 分片数量与边界沿用读语义的原编排；核预算充足时按rank加倍producer（每rank最多两个）。
// producer按逻辑片序循环领取任务，consumer按目标发送轮次反向遍历大片序。
// coreNumStage2表示原编排的逻辑分片数，同时是stage2的consumer核数与领取步长。
// 数据flag按(源rank, 大片序, 小片序)编址，不随实际分核改变。
// 单片不超过UB_DB_DATA_BATCH_SIZE时只读input一次，持有UB直到所有目标发送完成。
// READ_DONE位于数据flag之后，仅用于input==output且rank_非0时保护原input不被提前覆盖。
// host模板的CalcScratchMultiple=rankSize、目标发送轮转、ready/data/consumed与ping/pong协议保持不变。
// 与读语义共享分片和协议约束，producer/consumer调度不再要求镜像；SK/图模式仍走原路由。
template <typename T>
class AivAllGatherMesh1DWrite : public AivCommBase {
public:
    __aicore__ inline AivAllGatherMesh1DWrite() {}

    __aicore__ inline void InitCoreInfo(uint64_t len) // len is input length
    {
        // 不逐对端presync：数据flag按curTag(含sliceId)限定，非pingpong路径首尾有BarrierAll，
        // pingpong路径使用交替缓冲，cclBuff的跨算子归属交接由上述机制保证。
        // 逻辑片序按“小片序优先、rank片序次之”领取，每rank单producer时与读语义原分工一致。
        for (uint64_t piece = blockIdx_; piece < coreNumStage2; piece += coreNumStage1) {
            uint64_t pieceRank = piece % rankSize_;
            uint64_t coreIndex = piece / rankSize_;
            uint64_t dataPerRank = len / rankSize_;
            uint64_t remainderPerRank = len % rankSize_;
            // 数据对不齐的情况
            uint64_t innerDisplsPerRank = 0;
            uint64_t sendCurCountPerRank = 0;
            if (pieceRank < remainderPerRank) { // 这部分核需要多处理一个数据
                innerDisplsPerRank = pieceRank * dataPerRank + pieceRank;
                sendCurCountPerRank = dataPerRank + 1;
            } else {
                innerDisplsPerRank = pieceRank * dataPerRank + remainderPerRank;
                sendCurCountPerRank = dataPerRank;
            }

            uint64_t dataPerCore = sendCurCountPerRank / cutNum;
            uint64_t remainder = sendCurCountPerRank % cutNum;
            // 数据对不齐的情况
            uint64_t innerDispls = 0;
            uint64_t sendCurCount = 0;
            if (coreIndex < remainder) { // 这部分核需要多处理一个数据
                innerDispls = coreIndex * dataPerCore + coreIndex;
                sendCurCount = dataPerCore + 1;
            } else {
                innerDispls = coreIndex * dataPerCore + remainder;
                sendCurCount = dataPerCore;
            }

            if (sendCurCount > 0) {
                SendPiece(len, innerDisplsPerRank + innerDispls, sendCurCount, pieceRank * cutNum + coreIndex);
            }
        }
    }

    __aicore__ inline void Run(uint64_t len, uint64_t stride)
    {
        uint32_t coreNumPerRank = coreNumStage2 / rankSize_;
        uint64_t taskCount = rankSize_ * coreNumStage2;
        bool inputInOwnOutput = input_ == output_ + rank_ * stride;
        for (uint64_t task = blockIdx_ - coreNumStage1; task < taskCount; task += coreNumStage2) {
            // 实际consumer轮流领取任务，仅调整大片消费顺序，保持每个片的边界和flag。
            uint64_t logicalCore = task % coreNumStage2;
            uint32_t targetRank = logicalCore / coreNumPerRank;
            uint32_t coreIndex = logicalCore % coreNumPerRank;
            // 源rank S在第idx轮发往本rank T的片序为(T - S - idx) % rankSize_。
            // 按发送轮次反向消费片序，加两倍rankSize_避免无符号减法下溢。
            uint64_t idx = task / coreNumStage2;
            uint64_t innerIndex = (rank_ + 2 * rankSize_ - targetRank - idx) % rankSize_;
            uint64_t dataPerRank = len / rankSize_;
            uint64_t remainderPerRank = len % rankSize_;
            // 数据对不齐的情况
            uint64_t innerDisplsPerRank = 0;
            uint64_t sendCurCountPerRank = 0;
            if (innerIndex < remainderPerRank) { // 这部分核需要多处理一个数据
                innerDisplsPerRank = innerIndex * dataPerRank + innerIndex;
                sendCurCountPerRank = dataPerRank + 1;
            } else {
                innerDisplsPerRank = innerIndex * dataPerRank + remainderPerRank;
                sendCurCountPerRank = dataPerRank;
            }

            // 然后每个核分数据
            uint64_t dataPerCore = sendCurCountPerRank / coreNumPerRank;
            uint64_t remainder = sendCurCountPerRank % coreNumPerRank;
            // 数据对不齐的情况
            uint64_t innerDispls = 0;
            uint64_t sendCurCount = 0;
            if (coreIndex < remainder) { // 这部分核需要多处理一个数据
                innerDispls = coreIndex * dataPerCore + coreIndex;
                sendCurCount = dataPerCore + 1;
            } else {
                innerDispls = coreIndex * dataPerCore + remainder;
                sendCurCount = dataPerCore;
            }

            if (sendCurCount > 0) {
                // 标准in-place的本rank结果已经就位，避免回拷与producer读取同一input槽。
                if (inputInOwnOutput && targetRank == rank_) {
                    continue;
                }
                uint64_t pieceFlag = innerIndex * coreNumPerRank + coreIndex;
                // 先去wait flag（写语义flag由源rank直写时记在本端），然后把数据拷贝过来
                WaitFlag(rank_, targetRank * coreNumStage2 + pieceFlag, curTag_);
                // input==output时只有rank0结果覆盖原input；等待本地同一片已缓存或全部发送完成。
                if (needReadDone_ && targetRank == 0) {
                    WaitFlag(rank_, GetReadDoneFlag(pieceFlag), curTag_);
                }

                // 读语义为跨卡远读对端cclBuff；写语义对端已直写本端cclBuff的targetRank槽位，纯本地读
                uint64_t srcCclInOffset = reinterpret_cast<uint64_t>(myGmIn_)
                                          + (targetRank * len + innerDisplsPerRank + innerDispls) * sizeof(T);
                uint64_t usrOutOffset = output_ + targetRank * stride + (innerDisplsPerRank + innerDispls) * sizeof(T);
                CpGM2GM((__gm__ T*)usrOutOffset, (__gm__ T*)srcCclInOffset, sendCurCount);
            }
        }
        // 任务间不发布flag，UB复用由TQueBind的EnQue/DeQue/Free事件保证，无需逐个task冲刷流水；
        // 循环末统一屏障一次，保证本核最后一批UB2GM在返回前落地（pingpong路径无BarrierAll，尤其必要）。
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(uint64_t count, uint64_t sliceId, uint64_t stride)
    {
        uint32_t rankMultiple = 2;
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        needReadDone_ = input_ == output_ && rank_ != 0;
        if (rankSize_ < CTRL_CORE_MIN_RANK_SIZE && count * sizeof(T) >= DATA_LIMIT
            && numBlocks_ >= rankMultiple * rankSize_) {
            // 核数大于等于2倍ranksize
            curStageCoreNum = numBlocks_ / rankSize_ * rankSize_; // 总的核数
            coreNumStage1 = rankSize_;
            // 借鉴ReduceScatter写语义：至少保留一组consumer，且每个producer都能分到非空分片时，
            // 每rank再增加一个发送核（核预算充足的少卡场景才会命中）。
            uint64_t candidateProducerNum = MAX_PRODUCER_NUM_PER_RANK * rankSize_;
            if (curStageCoreNum >= candidateProducerNum + rankSize_) {
                uint64_t candidateCutNum = (curStageCoreNum - candidateProducerNum) / rankSize_;
                uint64_t nonEmptyPieceNum = count < candidateCutNum ? count : candidateCutNum;
                if (nonEmptyPieceNum >= MAX_PRODUCER_NUM_PER_RANK) {
                    coreNumStage1 = candidateProducerNum;
                }
            }
            coreNumStage2 = curStageCoreNum - coreNumStage1;
            cutNum = coreNumStage2 / rankSize_;

            if (blockIdx_ < coreNumStage1) {
                InitCoreInfo(count);
            } else if (blockIdx_ < curStageCoreNum) {
                Run(count, stride);
            }
        } else {
            // 8卡及以上、小数据或核数不足时使用控核写路径。
            RunCtrlCore(count, stride);
        }
    }

    __aicore__ inline void RunCtrlCore(uint64_t count, uint64_t stride)
    {
        // inOutQue只有一个UB缓冲区，控核搬运使用其完整容量。
        const bool savedUseDoubleBuffer = useDoubleBuffer_;
        useDoubleBuffer_ = false;
        const uint64_t bytes = count * sizeof(T);
        const uint64_t ownOutput = output_ + rank_ * stride;
        const uint64_t outputSpan = (rankSize_ - 1) * stride + bytes;
        const bool inputInOwnOutput = input_ == ownOutput;
        const bool inputOutsideOutput = input_ < output_ ? output_ - input_ >= bytes : input_ - output_ >= outputSpan;
        // 按整个输出跨度保守判断。仅在各输出片互不重叠时允许自身旁路和提前回拷。
        const bool bypassSelf = stride >= bytes && (inputInOwnOutput || inputOutsideOutput);
        // 仅8卡大包且核数严格大于2R时，前R核分段发送，随后R核回拷。
        const bool splitTxRx
            = rankSize_ == CTRL_CORE_MIN_RANK_SIZE && bytes >= DATA_LIMIT && numBlocks_ > 2 * rankSize_;
        const uint32_t peerCoreNum = numBlocks_ < rankSize_ ? numBlocks_ : rankSize_;
        const uint32_t copyCoreBegin = splitTxRx ? rankSize_ : 0;
        const bool sendCore = blockIdx_ < peerCoreNum;
        const bool copyCore = blockIdx_ >= copyCoreBegin && blockIdx_ < copyCoreBegin + peerCoreNum;
        const uint32_t peerCoreIndex = copyCore ? blockIdx_ - copyCoreBegin : blockIdx_;

        // 收发组使用同一组内索引分配对端；少核时每核负责连续多个rank。
        const uint32_t perCoreRankNum = rankSize_ / peerCoreNum;
        const uint32_t remainRankNum = rankSize_ % peerCoreNum;
        const uint32_t rankNum = perCoreRankNum + (peerCoreIndex < remainRankNum ? 1 : 0);
        const uint32_t rankBegin
            = peerCoreIndex * perCoreRankNum + (peerCoreIndex < remainRankNum ? peerCoreIndex : remainRankNum);
        const uint64_t firstCount = splitTxRx ? count / 2 : count;
        const CtrlCoreTask task{
            rankBegin, rankBegin + rankNum, splitTxRx ? 2u : 1u, {firstCount, count - firstCount}, bypassSelf};

        SyncAll<true>();
        if (copyCore) {
            PublishCtrlCoreReady(task);
        }
        if (sendCore) {
            SendCtrlCoreData(task, count);
        }

        // 重叠布局下所有核先完成input读取，再从CCL回拷，防止覆盖其他核尚未读取的input。
        if (!bypassSelf) {
            SyncAll<true>();
        }
        if (copyCore) {
            CopyCtrlCoreOutput(task, count, stride);
        }
        if (sendCore) {
            WaitCtrlCoreConsumed(task);
        }
        useDoubleBuffer_ = savedUseDoubleBuffer;
    }

    uint64_t coreOffset;
    uint64_t curCount;
    uint64_t coreNumStage1;
    uint64_t coreNumStage2;
    uint64_t cutNum;
    uint64_t curStageCoreNum;

private:
    static constexpr uint32_t CTRL_CORE_MIN_RANK_SIZE = 8;
    // 每个源rank最多两个发送核；仅在保留一组consumer且分片足够时启用。
    static constexpr uint64_t MAX_PRODUCER_NUM_PER_RANK = 2;
    bool needReadDone_ = false;

    struct CtrlCoreTask {
        uint32_t rankBegin;
        uint32_t rankEnd;
        uint32_t partCount;
        uint64_t partCounts[2];
        bool bypassSelf;
    };

    __aicore__ inline void PublishCtrlCoreReady(const CtrlCoreTask& task)
    {
        // data每段占R个flag，随后依次是consumed和ready。
        const uint64_t readyFlagOffset = (task.partCount + 1) * rankSize_;
        for (uint32_t rank = task.rankBegin; rank < task.rankEnd; rank++) {
            if (task.bypassSelf && rank == rank_) {
                continue;
            }
            Record(rank, readyFlagOffset + rank_, curTag_);
        }
    }

    __aicore__ inline void SendCtrlCoreData(const CtrlCoreTask& task, uint64_t count)
    {
        const uint64_t readyFlagOffset = (task.partCount + 1) * rankSize_;
        auto input = reinterpret_cast<__gm__ T*>(input_);
        for (uint32_t rank = task.rankBegin; rank < task.rankEnd; rank++) {
            if (task.bypassSelf && rank == rank_) {
                continue;
            }
            WaitFlag(rank_, readyFlagOffset + rank, curTag_);
            auto remoteSlot = reinterpret_cast<__gm__ T*>(GetGmIn(rank)) + rank_ * count;
            uint64_t offset = 0;
            for (uint32_t part = 0; part < task.partCount; part++) {
                CpGM2GM(remoteSlot + offset, input + offset, task.partCounts[part]);
                Record(rank, rank_ + part * rankSize_, curTag_);
                offset += task.partCounts[part];
            }
        }
    }

    __aicore__ inline void CopyCtrlCoreOutput(const CtrlCoreTask& task, uint64_t count, uint64_t stride)
    {
        const uint64_t consumedFlagOffset = task.partCount * rankSize_;
        const uint64_t ownOutput = output_ + rank_ * stride;
        for (uint32_t rank = task.rankBegin; rank < task.rankEnd; rank++) {
            if (task.bypassSelf && rank == rank_) {
                if (input_ != ownOutput) {
                    CpGM2GM(reinterpret_cast<__gm__ T*>(ownOutput), reinterpret_cast<__gm__ T*>(input_), count);
                    // 直拷没有Record，确保本核返回前数据已写到output。
                    PipeBarrier<PIPE_ALL>();
                }
                continue;
            }
            auto localSlot = reinterpret_cast<__gm__ T*>(myGmIn_) + rank * count;
            auto output = reinterpret_cast<__gm__ T*>(output_ + rank * stride);
            uint64_t offset = 0;
            for (uint32_t part = 0; part < task.partCount; part++) {
                WaitFlag(rank_, rank + part * rankSize_, curTag_);
                CpGM2GM(output + offset, localSlot + offset, task.partCounts[part]);
                offset += task.partCounts[part];
            }
            // Record内部的屏障确保所有分片回拷完成后才确认consumed。
            Record(rank, rank_ + consumedFlagOffset, curTag_);
        }
    }

    __aicore__ inline void WaitCtrlCoreConsumed(const CtrlCoreTask& task)
    {
        const uint64_t consumedFlagOffset = task.partCount * rankSize_;
        for (uint32_t rank = task.rankBegin; rank < task.rankEnd; rank++) {
            if (task.bypassSelf && rank == rank_) {
                continue;
            }
            WaitFlag(rank_, rank + consumedFlagOffset, curTag_);
        }
    }

    __aicore__ inline uint64_t GetReadDoneFlag(uint64_t pieceFlag) const
    {
        // 数据flag占[0, rankSize_ * coreNumStage2)，追加coreNumStage2个本地READ_DONE槽。
        return rankSize_ * coreNumStage2 + pieceFlag;
    }

    __aicore__ inline void SendCachedPiece(uint64_t len, uint64_t elemOffset, uint64_t pieceCount, uint64_t pieceFlag)
    {
        uint32_t copyCount = static_cast<uint32_t>(pieceCount);
        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input_) + elemOffset, pieceCount);
        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT, copyCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> payload = inOutQue.DeQue<T>();

        if (needReadDone_) {
            // Record的前置屏障保证GM2UB完成；之后原input可被覆盖，payload仍由本核持有。
            Record(rank_, GetReadDoneFlag(pieceFlag), curTag_);
        }
        for (uint32_t step = 0; step < rankSize_; step++) {
            uint32_t target = (rank_ + blockIdx_ + step) % rankSize_;
            GlobalTensor<T> outputGT;
            outputGT.SetGlobalBuffer(
                reinterpret_cast<__gm__ T*>(GetGmIn(target)) + rank_ * len + elemOffset, pieceCount);
            DataCopyUB2GM(outputGT, payload, copyCount);
            // 保持逐目标发布；Record内部保证数据搬出完成后才发布flag，无需重复PIPE_ALL。
            Record(target, rank_ * coreNumStage2 + pieceFlag, curTag_);
        }
        // 最后一次Record已等待搬出完成；所有目标发送结束后才释放，禁止提前复用payload。
        inOutQue.FreeTensor(payload);
    }

    __aicore__ inline void SendPiece(uint64_t len, uint64_t elemOffset, uint64_t pieceCount, uint64_t pieceFlag)
    {
        if (pieceCount <= UB_DB_DATA_BATCH_SIZE / sizeof(T)) {
            SendCachedPiece(len, elemOffset, pieceCount, pieceFlag);
            return;
        }

        // 多chunk分片保留原搬运，每个目标重新读取input。
        auto input = reinterpret_cast<__gm__ T*>(input_) + elemOffset;
        for (uint32_t step = 0; step < rankSize_; step++) {
            uint32_t target = (rank_ + blockIdx_ + step) % rankSize_;
            auto output = reinterpret_cast<__gm__ T*>(GetGmIn(target)) + rank_ * len + elemOffset;
            CpGM2GM(output, input, pieceCount);
            Record(target, rank_ * coreNumStage2 + pieceFlag, curTag_);
        }
        if (needReadDone_) {
            // 此分支未整体缓存，必须等所有目标发送结束才允许consumer覆盖原input。
            Record(rank_, GetReadDoneFlag(pieceFlag), curTag_);
        }
    }
};

template <typename T>
__aicore__ inline void AivAllGatherV2Mesh1DWrite(KERNEL_ARGS_DEF)
{
    AivAllGatherMesh1DWrite<T> op;
    bool pingpong = false;
    if (len * sizeof(T) <= DATA_LIMIT && rankSize * len * sizeof(T) <= PINGPONG_TOTAL_DATA_LIMIT) {
        pingpong = true;
    }
    op.Init(KERNEL_CLASS_INIT, true, pingpong);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    op.Process(len, sliceId, outputSliceStride);
    if (!pingpong) {
        op.BarrierAll();
    }
}

#endif // AIV_ALL_GATHER_MESH_1D_WRITE_H

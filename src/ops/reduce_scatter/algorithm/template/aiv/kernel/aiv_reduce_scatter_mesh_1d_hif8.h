/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base_v2.h"
using namespace AscendC;

template <typename T>
class AivReduceScatterMesh1DHif8 : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterMesh1DHif8() {}

    __aicore__ inline void Init(
        GM_ADDR buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank, uint32_t rankSize,
        uint64_t len, uint32_t dataType, uint32_t reduceOp, uint32_t root, uint64_t inputSliceStride,
        uint64_t outputSliceStride, uint64_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
        GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter,
        uint32_t numBlocks, bool useDoubleBuffer, bool pingpong = false)
    {
        rank_ = rank;
        sendRecvRemoteRank_ = sendRecvRemoteRank;
        root_ = root;
        rankSize_ = rankSize;
        reduceOp_ = reduceOp;
        len_ = len;
        input_ = input;
        output_ = output;
        dataType_ = dataType;
        useDoubleBuffer_ = useDoubleBuffer;
        numBlocks_ = numBlocks;

        inputSliceStride_ = inputSliceStride;
        outputSliceStride_ = outputSliceStride;
        repeatNum_ = repeatNum;
        inputRepeatStride_ = inputRepeatStride;
        outputRepeatStride_ = outputRepeatStride;

        localOffset = (rankSize_ * NUM_BLOCKS_FOUR_PER_RANK_A3 * FLAG_BUF_NUM) * FLAG_SIZE;
        multiOffset = MAX_NUM_BLOCKS * DOUBLE * FLAG_SIZE + localOffset;
        pingpongOffset = multiOffset + DOUBLE * DOUBLE * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * DOUBLE;
        countOffset = DOUBLE * pingpongOffset;
        separateOffset = countOffset + NUM_BLOCKS_FOUR_PER_RANK_A3 * rankSize_ * FLAG_SIZE;

        pipe.InitBuffer(localFlagBuf, LOCAL_FLAG_BUF_LEN);
        localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_ONE_OFFSET);
        localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_TWO_OFFSET);
        localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_THREE_OFFSET);
        localGetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FOUR_OFFSET);
        localTagTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FIVE_OFFSET);
        pipe.InitBufPool(inOutPool, UB_MAX_DATA_SIZE);
        pipe.InitBufPool(reducePool, UB_MAX_DATA_SIZE, inOutPool);
        if (useDoubleBuffer_) {
            inOutPool.InitBuffer(inOutQue, 2, UB_DB_DATA_BATCH_SIZE);
        } else {
            inOutPool.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);
        }
        GetTag(buffIn);
        InitBuffArray(buffIn, pingpong);
    }

    __aicore__ inline void Init(GM_ADDR hiddenInput, GM_ADDR input, GM_ADDR output, bool pingpong = false)
    {
        __gm__ AivSuperKernelArgs* args = reinterpret_cast<__gm__ AivSuperKernelArgs*>(hiddenInput);

        rank_ = args->rank;
        rankSize_ = args->rankSize;
        reduceOp_ = args->reduceOp;
        len_ = args->len;
        tag_ = args->tag;
        dataType_ = args->dataType;
        unitSize_ = args->unitSize;
        numBlocks_ = args->numBlocks;

        input_ = reinterpret_cast<uint64_t>(input);
        output_ = reinterpret_cast<uint64_t>(output);
        cclBufferSize_ = args->cclBufferSize;

        inputSliceStride_ = len_ * unitSize_;
        outputSliceStride_ = len_ * unitSize_;
        repeatNum_ = args->repeatNum;
        inputRepeatStride_ = args->inputRepeatStride;
        outputRepeatStride_ = args->outputRepeatStride;

        localOffset = (rankSize_ * NUM_BLOCKS_FOUR_PER_RANK_A3 * FLAG_BUF_NUM) * FLAG_SIZE;
        multiOffset = MAX_NUM_BLOCKS * DOUBLE * FLAG_SIZE + localOffset;
        pingpongOffset = multiOffset + DOUBLE * DOUBLE * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * DOUBLE;
        countOffset = DOUBLE * pingpongOffset;
        separateOffset = countOffset + NUM_BLOCKS_FOUR_PER_RANK_A3 * rankSize_ * FLAG_SIZE;

        pipe.InitBuffer(localFlagBuf, LOCAL_FLAG_BUF_LEN);
        localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_ONE_OFFSET);
        localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_TWO_OFFSET);
        localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_THREE_OFFSET);
        localGetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FOUR_OFFSET);
        localTagTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FIVE_OFFSET);
        pipe.InitBufPool(inOutPool, UB_MAX_DATA_SIZE);
        pipe.InitBufPool(reducePool, UB_MAX_DATA_SIZE, inOutPool);
        if (useDoubleBuffer_) {
            inOutPool.InitBuffer(inOutQue, 2, UB_DB_DATA_BATCH_SIZE);
        } else {
            inOutPool.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);
        }
        GetTag(args->buffersIn);
        InitBuffArray(args->buffersIn, pingpong);
        if (args->clearEnable == 1) {
            ClearSyncBuf();
        }
    }

    __aicore__ inline void InitCoreInfo(uint64_t totalLen, uint64_t inputStride)
    {
        totalLen_ = totalLen;
        recvCount_ = totalLen / numBlocks_ / ubAlignCount_ * ubAlignCount_;
        reduceCount_ = blockIdx_ < numBlocks_ - 1 ? recvCount_ : totalLen_ - blockIdx_ * recvCount_;
        inputStride_ = inputStride;
        rankSizeU32_ = static_cast<uint32_t>(rankSize_);
        SplitLogicalRange(rankSizeU32_, publishBegin_, publishEnd_);
    }

    __aicore__ inline void AllToAll()
    {
        for (uint32_t targetRank = publishBegin_; targetRank < publishEnd_; ++targetRank) {
            uint64_t srcOffset = input_ + targetRank * inputStride_;
            uint64_t dstOffset = reinterpret_cast<uint64_t>(GetGmIn(targetRank)) + rank_ * totalLen_ * sizeof(T);
            CpGM2GM((__gm__ T*)dstOffset, (__gm__ T*)srcOffset, totalLen_);
            pipe_barrier(PIPE_ALL);
            Record(targetRank, rank_, curTag_);
        }
        inOutPool.Reset();
    }

    __aicore__ inline void LocalReduce()
    {
        reducePool.InitBuffer(inQueueX, 2, maxCountPerLoop_ * sizeof(T));
        reducePool.InitBuffer(castBuf, maxCountPerLoop_ * sizeof(float));
        reducePool.InitBuffer(accBuf, maxCountPerLoop_ * sizeof(float));
        reducePool.InitBuffer(outQueueZ, 1, maxCountPerLoop_ * sizeof(T));
        LocalTensor<float> castLocal = castBuf.Get<float>();
        LocalTensor<float> accLocal = accBuf.Get<float>();

        uint64_t outSegOffset = blockIdx_ * recvCount_ * sizeof(T);
        uint64_t countLeft = reduceCount_;
        uint64_t curOffset = 0;
        while (countLeft > 0) {
            uint64_t curCount = countLeft > maxCountPerLoop_ ? maxCountPerLoop_ : countLeft;
            for (uint32_t r = 0; r < rankSizeU32_; ++r) {
                WaitFlag(rank_, r, curTag_);
                LocalTensor<T> inLocal = inQueueX.AllocTensor<T>();
                GlobalTensor<T> srcGm;
                uint64_t srcOffset = reinterpret_cast<uint64_t>(myGmIn_) + r * totalLen_ * sizeof(T) + outSegOffset
                                     + curOffset * sizeof(T);
                srcGm.SetGlobalBuffer((__gm__ T*)srcOffset, curCount);
                DataCopyGM2UB(inLocal, srcGm, curCount);
                pipe_barrier(PIPE_ALL);
                inQueueX.EnQue(inLocal);
                inLocal = inQueueX.DeQue<T>();
                if (r == 0) {
                    Cast<float, T>(accLocal, inLocal, RoundMode::CAST_NONE, curCount);
                } else {
                    Cast<float, T>(castLocal, inLocal, RoundMode::CAST_NONE, curCount);
                    if (reduceOp_ == HcclReduceOp::HCCL_REDUCE_SUM) {
                        Add<float>(accLocal, castLocal, accLocal, curCount);
                    } else if (reduceOp_ == HcclReduceOp::HCCL_REDUCE_MAX) {
                        Max<float>(accLocal, castLocal, accLocal, curCount);
                    } else if (reduceOp_ == HcclReduceOp::HCCL_REDUCE_MIN) {
                        Min<float>(accLocal, castLocal, accLocal, curCount);
                    }
                }
                inQueueX.FreeTensor(inLocal);
            }
            LocalTensor<T> outLocal = outQueueZ.AllocTensor<T>();
            Cast<T, float>(outLocal, accLocal, RoundMode::CAST_ROUND, curCount);
            pipe_barrier(PIPE_ALL);
            outQueueZ.EnQue(outLocal);
            outLocal = outQueueZ.DeQue<T>();
            GlobalTensor<T> outGm;
            outGm.SetGlobalBuffer((__gm__ T*)(output_ + outSegOffset + curOffset * sizeof(T)), curCount);
            DataCopyUB2GM(outGm, outLocal, curCount);
            pipe_barrier(PIPE_ALL);
            outQueueZ.FreeTensor(outLocal);
            curOffset += curCount;
            countLeft -= curCount;
        }
        reducePool.Reset();
    }

    __aicore__ inline void Process(uint32_t sliceId)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        AllToAll();
        SyncAll<true>();
        LocalReduce();
    }

private:
    __aicore__ inline void SplitLogicalRange(uint32_t total, uint32_t& begin, uint32_t& end)
    {
        const uint32_t baseCnt = total / numBlocks_;
        const uint32_t extra = total % numBlocks_;
        const uint32_t myCnt = baseCnt + (blockIdx_ < extra ? 1u : 0u);
        begin = baseCnt * blockIdx_ + (blockIdx_ < extra ? blockIdx_ : extra);
        end = begin + myCnt;
    }
    uint64_t totalLen_ = 0;
    uint64_t recvCount_ = 0;
    uint64_t reduceCount_ = 0;
    uint64_t inputStride_ = 0;
    uint32_t rankSizeU32_ = 0;
    uint32_t publishBegin_ = 0;
    uint32_t publishEnd_ = 0;
    static constexpr uint64_t ubAlignCount_ = UB_ALIGN_SIZE / sizeof(T);
    static constexpr uint64_t maxCountPerLoop_
        = UB_MAX_DATA_SIZE / (sizeof(T) * 3 + sizeof(float) * 2) / ubAlignCount_ * ubAlignCount_;

    TBufPool<TPosition::VECCALC> inOutPool;
    TBufPool<TPosition::VECCALC> reducePool;
    TBuf<QuePosition::VECCALC> accBuf;
    TBuf<QuePosition::VECCALC> castBuf;
};

template <typename T>
__aicore__ inline void AivReduceScatterV2Mesh1DHif8(KERNEL_ARGS_DEF)
{
    AivReduceScatterMesh1DHif8<T> op;
    op.Init(KERNEL_CLASS_INIT, false);
    op.InitCoreInfo(len, inputSliceStride);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    op.Process(sliceId);
    op.BarrierAll();
}

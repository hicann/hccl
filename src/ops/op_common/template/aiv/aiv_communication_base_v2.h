/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_COMMUNICATION_BASE_V2_H
#define AIV_COMMUNICATION_BASE_V2_H

#include "kernel_operator.h"
#include "sync_interface.h"

using namespace AscendC;

#define EXPORT_AIV_META_INFO(kernel_name) \
static const struct FunLevelKType kernel_name##_kernel_type_section __attribute__ \
((used, section (".ascend.meta." #kernel_name))) \
= {{F_TYPE_KTYPE, sizeof(unsigned int), K_TYPE_AIV}}

constexpr uint32_t MAX_RANK_SIZE = 128; // server内最大卡数
constexpr uint64_t BUFFER_OUT_ADDR_OFFSET = 16 * 1024;
constexpr uint64_t TOPO_ADDR_OFFSET = 32 * 1024;
constexpr uint64_t FLAG_ADDR_OFFSET = 40 * 1024;
constexpr uint64_t TOPO_LEN = 64;
constexpr uint64_t TOPO_LEN_Y_OFFSET = 8;
constexpr uint64_t TOPO_LEN_Z_OFFSET = 16;
constexpr uint64_t LOCAL_FLAG_BUF_LEN = 2560;
constexpr uint64_t AIV_TAG_MOVE_RIGHT_BITS = 16;
constexpr uint64_t LOW_16_BITS = 0xFFFF;
constexpr uint64_t DATA_LIMIT = 512 * 1024;

struct ExtraArgs {
    uint64_t sendCounts[MAX_RANK_SIZE] = {};
    uint64_t sendDispls[MAX_RANK_SIZE] = {};
    uint64_t recvCounts[MAX_RANK_SIZE] = {};
    uint64_t recvDispls[MAX_RANK_SIZE] = {};
};

enum class AivNotifyType {
    ACK,
    DataSignal,
    Done
};

enum class CommPattern {
    //server间
    interRank,
    //server内
    intraRank
};

#define KERNEL_ARGS_DEF \
GM_ADDR buffIn, \
uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank, uint32_t rankSize, uint64_t xRankSize,  uint64_t yRankSize, uint64_t zRankSize, uint64_t len, \
uint32_t dataType, uint32_t reduceOp, uint32_t root, uint32_t sliceId, \
uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride, \
bool isOpBase, \
GM_ADDR headCountMem, \
GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter

#define EXTERN_KERNEL_ARGS_DEF_V2 \
KERNEL_ARGS_DEF, ExtraArgs extraArgs

#define KERNEL_ARGS_CALL \
buffIn, \
input, output, rank, sendRecvRemoteRank, rankSize, xRankSize, yRankSize, zRankSize, len, dataType, reduceOp, root, sliceId, \
inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride, outputRepeatStride, \
isOpBase, \
headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter

#define EXTERN_KERNEL_ARGS_CALL \
KERNEL_ARGS_CALL, extraArgs

#define KERNEL_CLASS_INIT \
buffIn, input, output,\
rank, sendRecvRemoteRank, rankSize, xRankSize, yRankSize, zRankSize, len, dataType, reduceOp, root, \
inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride, outputRepeatStride, \
headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter

constexpr uint64_t AIV_PING_PONG_FACTOR_TWO = 2;
constexpr uint32_t NUM_BLOCKS_FOUR_PER_RANK_A3 = 4;
constexpr uint32_t MAX_NUM_BLOCKS = 48;

constexpr uint64_t FLAG_SIZE = 128;
constexpr uint64_t UB_ALIGN_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE_4 = UB_FLAG_SIZE * 4;
constexpr uint64_t UB_FLAG_SIZE_8 = UB_FLAG_SIZE * 8;
constexpr uint64_t UB_MAX_DATA_SIZE = 190 * 1024;
constexpr uint64_t UB_DB_DATA_BATCH_SIZE = UB_MAX_DATA_SIZE / 2;
constexpr uint32_t MaxBufferSize = 200 * 1024 * 1024;

constexpr uint64_t ATOMIC_FLAG_SIZE = 512;
constexpr uint64_t FLAG_ONE_OFFSET = 0;
constexpr uint64_t FLAG_TWO_OFFSET = FLAG_SIZE;
constexpr uint64_t FLAG_THREE_OFFSET = FLAG_SIZE * 2;
constexpr uint64_t FLAG_FOUR_OFFSET = FLAG_SIZE * 3;
constexpr uint64_t FLAG_FIVE_OFFSET = FLAG_SIZE * 4;

constexpr uint64_t DOUBLE = 2;
constexpr uint64_t FLAG_BUF_NUM = 3;
constexpr uint64_t TILING_NUM = 4;
constexpr uint64_t CHUNK_SIZE = 2048;

constexpr int32_t TAG_INIT_VALUE = 1;
constexpr int32_t TAG_RESET_COUNT = 1000;
constexpr uint32_t AIV_FLAG_CLEAR_OFFSET = 16 * 1024 * 1024;

// 当前每个kernel最多使用4组同步标记，这里预留6组
constexpr uint32_t MAX_FLAG_SIZE_PER_KERNEL = AIV_FLAG_CLEAR_OFFSET - MAX_RANK_SIZE * FLAG_SIZE;

#define BASE_FLAG_OFFSET (MAX_FLAG_SIZE_PER_KERNEL)

class AivCommBase {
public:
    __aicore__ inline AivCommBase() {
    }

    __aicore__ inline void Init(GM_ADDR buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank, uint32_t rankSize, uint64_t xRankSize,  uint64_t yRankSize, uint64_t zRankSize,
                                uint64_t len,
                                uint32_t dataType, uint32_t reduceOp, uint32_t root,
                                uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
                                GM_ADDR headCountMem,
                                GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter,
                                bool useDoubleBuffer)
    {
        rank_ = rank;
        sendRecvRemoteRank_  = sendRecvRemoteRank;
        root_ = root;
        rankSize_ = rankSize;
        xRankSize_ = xRankSize;
        yRankSize_ = yRankSize;
        zRankSize_ = zRankSize;
        reduceOp_ = reduceOp;
        len_ = len;
        input_ = input;
        output_ = output;
        dataType_ = dataType;
        useDoubleBuffer_ = useDoubleBuffer;
        numBlocks_ = block_num;

        inputSliceStride_ = inputSliceStride;
        outputSliceStride_ = outputSliceStride;
        repeatNum_ = repeatNum;
        inputRepeatStride_ = inputRepeatStride;
        outputRepeatStride_ = outputRepeatStride;

        InitBuffArray(buffIn);

        localOffset = (rankSize_ * NUM_BLOCKS_FOUR_PER_RANK_A3 * FLAG_BUF_NUM) * FLAG_SIZE;
        multiOffset = MAX_NUM_BLOCKS * DOUBLE * FLAG_SIZE+ localOffset;
        pingpongOffset = multiOffset + DOUBLE * DOUBLE * NUM_BLOCKS_FOUR_PER_RANK_A3 * ATOMIC_FLAG_SIZE * DOUBLE;
        countOffset = DOUBLE * pingpongOffset;
        seperateOffset = countOffset + NUM_BLOCKS_FOUR_PER_RANK_A3 * rankSize_ * FLAG_SIZE;

        pipe.InitBuffer(localFlagBuf, LOCAL_FLAG_BUF_LEN);
        localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_ONE_OFFSET);
        localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_TWO_OFFSET);
        localCheckGETensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_THREE_OFFSET);
        localGetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FOUR_OFFSET);
        localTagTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, FLAG_FIVE_OFFSET);
        pipe.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);

        uint64_t chunkSize = UB_MAX_DATA_SIZE / TILING_NUM / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
        pipe.InitBuffer(inQueueX, 1, chunkSize);
        pipe.InitBuffer(inQueueY, 1, chunkSize);
        pipe.InitBuffer(outQueueZ, 1, chunkSize);

        GetTag(buffIn);
    }

    __aicore__ inline void InitBuffArray(GM_ADDR buffIn)
    {
        GlobalTensor<uint64_t> ipcBufferGlobal;
        ipcBufferGlobal.SetGlobalBuffer((__gm__ uint64_t*)(buffIn));
        for(int i=0; i<rankSize_;i++){
            GM_IN[i] = (GM_ADDR)ipcBufferGlobal.GetValue(i);
            GM_OUT[i] = (GM_ADDR)ipcBufferGlobal.GetValue(BUFFER_OUT_ADDR_OFFSET / sizeof(uint64_t) + i) + FLAG_ADDR_OFFSET;
        }
        for(int i=0; i< TOPO_LEN ;i++){
            TOPO_[i] = (uint64_t)ipcBufferGlobal.GetValue(TOPO_ADDR_OFFSET / sizeof(uint64_t) + i);
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void GetTag(GM_ADDR buffIn)
    {
        uint64_t blockIdx = GetBlockIdx();
        LocalTensor<uint32_t> localIn = inOutQue.AllocTensor<uint32_t>();
        GlobalTensor<uint32_t> ipcBufferGlobal;
        ipcBufferGlobal.SetGlobalBuffer((__gm__ uint32_t*)(buffIn));
        DataCopyGM2UB(localIn, ipcBufferGlobal[AIV_FLAG_CLEAR_OFFSET / sizeof(uint32_t)], 1);
        pipe_barrier(PIPE_ALL);
        tag_ = localIn.GetValue(0);
        SyncAll<true>();
        tag_ = (tag_ == TAG_RESET_COUNT) ? TAG_INIT_VALUE : tag_ + 1;
        if (blockIdx == 0) {
            localIn.SetValue(0, tag_);
            pipe_barrier(PIPE_ALL);
            DataCopyUB2GM(ipcBufferGlobal[AIV_FLAG_CLEAR_OFFSET / sizeof(uint32_t)], localIn, 1);
        }
        inOutQue.FreeTensor(localIn);
    }

    __aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);

    template<typename T>
    __aicore__ inline void SetAtomicOp(uint32_t atomicOp);

    template<typename T>
    __aicore__ inline void DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                         const uint32_t calCount);

    template<typename T>
    __aicore__ inline void DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                         const uint32_t calCount);

    template<typename T>
    __aicore__ inline void CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, uint32_t atomicOp);

    template<typename T>
    __aicore__ inline void CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count);

    template<typename T>
    __aicore__ inline void Reduce64(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, uint32_t reduceOp);

    __aicore__ inline void BarrierAll();

    __aicore__ inline void SendRecvBarrierAll(uint32_t myRank, uint32_t remoteRank);

    __aicore__ inline bool IsFirstOP(int32_t sliceId);

    __aicore__ inline void BarrierForFirstOP();

    __aicore__ inline void SendRecvBarrierForFirstOP(uint32_t myRank, uint32_t remoteRank);

    __aicore__ inline void WaitFlag(uint32_t targetRank, uint64_t flag_offset, int32_t curTag);

    __aicore__ inline void Record(uint32_t targetRank, uint64_t flag_offset, int32_t curTag);

    GM_ADDR GM_IN[MAX_RANK_SIZE];
    GM_ADDR GM_OUT[MAX_RANK_SIZE];
    uint64_t TOPO_[TOPO_LEN];
    uint32_t rank_;
    uint32_t sendRecvRemoteRank_;
    uint32_t root_;
    uint32_t rankSize_;
    uint64_t xRankSize_;
    uint64_t yRankSize_;
    uint64_t zRankSize_;
    uint32_t reduceOp_;
    uint32_t dataType_;

    uint64_t input_;
    uint64_t output_;

    uint64_t len_;
    uint32_t tag_;
    uint32_t curTag_{0};
    int32_t numBlocks_;

    uint64_t inputSliceStride_;
    uint64_t outputSliceStride_;
    uint64_t repeatNum_;
    uint64_t inputRepeatStride_;
    uint64_t outputRepeatStride_;

    bool useDoubleBuffer_;

    TPipe pipe;
    TBuf<> localFlagBuf;
    LocalTensor<int32_t> localSetTensor;
    LocalTensor<int32_t> localCheckTensor;
    LocalTensor<int32_t> localCheckGETensor;
    LocalTensor<int32_t> localGetTensor;
    LocalTensor<int32_t> localTagTensor;
    GlobalTensor<int32_t> d2hGlobal;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inQueueX;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inQueueY;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> outQueueZ;

    uint32_t localOffset;
    uint32_t multiOffset;
    uint32_t pingpongOffset;
    uint32_t countOffset;
    uint32_t seperateOffset;
};


__aicore__ inline void AivCommBase::Record(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
{
    d2hGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(GM_OUT[targetRank] + flag_offset * FLAG_SIZE));
    localTagTensor.SetValue(0, curTag);
    pipe_barrier(PIPE_ALL);
    DataCopyUB2GM(d2hGlobal, localTagTensor, FLAG_SIZE / sizeof(int32_t));
    pipe_barrier(PIPE_ALL);
}


__aicore__ inline void AivCommBase::WaitFlag(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
{
    d2hGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(GM_OUT[targetRank] + flag_offset * FLAG_SIZE));
    while (true) {
        DataCopyGM2UB(localTagTensor, d2hGlobal, FLAG_SIZE / sizeof(int32_t));
        pipe_barrier(PIPE_ALL);
        if (localTagTensor.GetValue(0) == curTag) {
            break;
        }
    }
}

__aicore__ inline bool AivCommBase::IsFirstOP(int32_t sliceId)
{
    return sliceId == 1 && tag_ == 1;
}

__aicore__ inline void AivCommBase::BarrierForFirstOP()
{
    // 每个核分配多个rank
    uint32_t perCoreRankNum = rankSize_ / numBlocks_;
    uint32_t remainRankNum = rankSize_ % numBlocks_;
    uint32_t curCoreRankNum = block_idx < remainRankNum ? perCoreRankNum + 1 : perCoreRankNum;
    uint32_t startRank = block_idx < remainRankNum
                        ? (perCoreRankNum + 1) * block_idx
                        : perCoreRankNum * block_idx + remainRankNum;
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        uint64_t flag_offset = BASE_FLAG_OFFSET + rank * FLAG_SIZE;
        Record(rank_, flag_offset / FLAG_SIZE, DOUBLE);
    }
    PipeBarrier<PIPE_ALL>();
    uint64_t flag_offset = BASE_FLAG_OFFSET + rank_ * FLAG_SIZE;
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        WaitFlag(rank, flag_offset / FLAG_SIZE, DOUBLE);
    }

    SyncAll<true>();
}

// 为sendRecv单独设计
__aicore__ inline void AivCommBase::SendRecvBarrierForFirstOP(uint32_t myRank, uint32_t remoteRank)
{
    if (GetBlockIdx() == 0) {
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            if (i == myRank || i == remoteRank) {
                uint64_t flag_offset = BASE_FLAG_OFFSET + i * FLAG_SIZE;
                Record(rank_, flag_offset / FLAG_SIZE, DOUBLE);
            }
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            if (i == myRank || i == remoteRank) {
                uint64_t flag_offset = BASE_FLAG_OFFSET + rank_ * FLAG_SIZE;
                WaitFlag(i, flag_offset / FLAG_SIZE, DOUBLE);
            }
        }
    }

    SyncAll<true>();
}

__aicore__ inline void AivCommBase::BarrierAll()
{
    SyncAll<true>();

    // 每个核分配多个rank
    uint32_t perCoreRankNum = rankSize_ / numBlocks_;
    uint32_t remainRankNum = rankSize_ % numBlocks_;
    uint32_t curCoreRankNum = block_idx < remainRankNum ? perCoreRankNum + 1 : perCoreRankNum;
    uint32_t startRank = block_idx < remainRankNum
                        ? (perCoreRankNum + 1) * block_idx
                        : perCoreRankNum * block_idx + remainRankNum;
    uint64_t flag_offset = BASE_FLAG_OFFSET + rank_ * FLAG_SIZE;
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        Record(rank, flag_offset / FLAG_SIZE, 1);
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
        uint64_t flag_offset = BASE_FLAG_OFFSET + rank * FLAG_SIZE;
        WaitFlag(rank_, flag_offset / FLAG_SIZE, 1);
        Record(rank_, flag_offset / FLAG_SIZE, 0);
    }
}

// 为sendRecv单独设计
__aicore__ inline void AivCommBase::SendRecvBarrierAll(uint32_t myRank, uint32_t remoteRank)
{
    SyncAll<true>();
    if (GetBlockIdx() == 0) {
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            if (i == myRank || i == remoteRank) {
                uint64_t flag_offset = BASE_FLAG_OFFSET + rank_ * FLAG_SIZE;
                Record(i, flag_offset / FLAG_SIZE, 1);
            }
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            if (i == myRank || i == remoteRank) {
                uint64_t flag_offset = BASE_FLAG_OFFSET + i * FLAG_SIZE;
                WaitFlag(rank_, flag_offset / FLAG_SIZE, 1);
                Record(rank_, flag_offset / FLAG_SIZE, 0);
            }
        }
    }
}

__aicore__ inline uint64_t AivCommBase::CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template<typename T>
__aicore__ inline void AivCommBase::SetAtomicOp(uint32_t atomicOp)
{
    switch (atomicOp) {
        case HcclReduceOp::HCCL_REDUCE_SUM:
        SetAtomicAdd<T>(); break;
        case HcclReduceOp::HCCL_REDUCE_MAX:
        SetAtomicMax<T>(); break;
        case HcclReduceOp::HCCL_REDUCE_MIN:
        SetAtomicMin<T>(); break;
        default:
        SetAtomicNone(); break;
    }
}

template<typename T>
__aicore__ inline void AivCommBase::DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                                  const uint32_t calCount)
{
    copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)dstLocal.GetPhyAddr(), (__gm__ uint8_t*)srcGlobal.GetPhyAddr(), 0, 1, calCount * sizeof(T), 0,0,0, 0,0,0);
}

template<typename T>
__aicore__ inline void AivCommBase::DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                                  const uint32_t calCount)
{
    copy_ubuf_to_gm_align_v2(reinterpret_cast<const __gm__ uint8_t*>(dstGlobal.GetPhyAddr()), reinterpret_cast<__ubuf__ uint8_t*>(srcLocal.GetPhyAddr()), 0, 1, calCount * sizeof(T), 0,0,0);
}

template<typename T>
__aicore__ inline void AivCommBase::CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count)
{
    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);
    uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
    if (useDoubleBuffer_) {
        maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
    }
    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;
    }
    return;
}

template<typename T>
__aicore__ inline void AivCommBase::CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, uint32_t atomicOp)
{
    if constexpr (Std::is_same<T, int64_t>::value) {
        Reduce64(outputGM, inputGM, count, atomicOp);
        return;
    } else {
        GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer(inputGM, count);
        GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer(outputGM, count);

        SetAtomicOp<T>(atomicOp);

        uint64_t maxCountPerLoop = UB_MAX_DATA_SIZE / sizeof(T);
        if (useDoubleBuffer_) {
            maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
        }
        uint64_t curOffset = 0;
        while (count > 0) {
            uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

            LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
            DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
            inOutQue.EnQue(localIn);
            LocalTensor<T> localOut = inOutQue.DeQue<T>();
            DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
            inOutQue.FreeTensor(localOut);

            count -= curCount;
            curOffset += curCount;
        }

        SetAtomicNone();
        return;
    }
}

template<typename T>
__aicore__ inline void AivCommBase::Reduce64(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, uint32_t reduceOp)
{
    GlobalTensor<T> xGm;  // xGm, yGm为输入，zGm为输出
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    xGm.SetGlobalBuffer(inputGM, count);
    yGm.SetGlobalBuffer(outputGM, count);
    zGm.SetGlobalBuffer(outputGM, count);

    // 单核Add/Max/Min数据量限制
    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > CHUNK_SIZE ? CHUNK_SIZE : count;

        xGm.SetGlobalBuffer(inputGM + curOffset, curCount);
        yGm.SetGlobalBuffer(outputGM + curOffset, curCount);
        zGm.SetGlobalBuffer(outputGM + curOffset, curCount);

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyGM2UB(xLocal, xGm, curCount);
        pipe_barrier(PIPE_ALL);
        inQueueX.EnQue(xLocal);

        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        DataCopyGM2UB(yLocal, yGm, curCount);
        inQueueY.EnQue(yLocal);

        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();

        pipe_barrier(PIPE_ALL);
        if (reduceOp == HcclReduceOp::HCCL_REDUCE_SUM) {
            Add<T>(yLocal, xLocal, yLocal, curCount);
        } else if (reduceOp == HcclReduceOp::HCCL_REDUCE_MAX) {
            Max<T>(yLocal, xLocal, yLocal, curCount);
        } else if (reduceOp == HcclReduceOp::HCCL_REDUCE_MIN) {
            Min<T>(yLocal, xLocal, yLocal, curCount);
        }
        pipe_barrier(PIPE_ALL);

        DataCopyUB2GM(zGm, yLocal, curCount);
        pipe_barrier(PIPE_ALL);
        // 释放localTensor
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);

        count -= curCount;
        curOffset += curCount;
    }
}

// 910B支持的Atomic数据类型
#define AIV_ATOMIC_DATA_TYPE_DEF(func) \
    func(float); \
    func(half); \
    func(int16_t); \
    func(int32_t); \
    func(int8_t); \
    func(bfloat16_t); \
    func(int64_t)

// 910B支持的DataCopy数据类型
#define AIV_COPY_DATA_TYPE_DEF(func) \
    func(half); \
    func(int16_t); \
    func(uint16_t); \
    func(float); \
    func(int32_t); \
    func(uint32_t); \
    func(int8_t); \
    func(uint8_t); \
    func(bfloat16_t); \
    func(uint64_t); \
    func(int64_t); \
    func(fp8_e4m3fn_t); \
    func(fp8_e5m2_t); \
    func(fp8_e8m0_t); \
    func(hifloat8_t)

#endif  /* AIV_COMMUNICATION_BASE_V2_H */
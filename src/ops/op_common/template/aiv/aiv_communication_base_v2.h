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

constexpr uint32_t MAX_RANK_SIZE = 8; // server内最大卡数
constexpr uint64_t BUFFER_OUT_ADDR_OFFSET = 16 * 1024;
constexpr uint64_t TOPO_ADDR_OFFSET = 32 * 1024;
constexpr uint64_t FLAG_ADDR_OFFSET = 40 * 1024;
constexpr uint64_t TOPO_LEN = 32;
constexpr uint64_t TOPO_LEN_Y_OFFSET = 8;
constexpr uint64_t TOPO_LEN_Z_OFFSET = 16;
constexpr uint64_t IPC_SYNC_OFFSET = 500 * 1024;
constexpr uint64_t BARRIER_OFFSET = 900 * 1024;
constexpr uint64_t SYNC_CORE_OFFSET = 950 * 1024;
constexpr uint64_t LOCAL_FLAG_BUF_LEN = 1024;
constexpr uint64_t AIV_TAG_MOVE_RIGHT_BITS = 16;
constexpr uint64_t LOW_16_BITS = 0xFFFF;

struct ExtraArgsv2 {
    uint64_t sendCountMatrix[MAX_RANK_SIZE * MAX_RANK_SIZE] = {};
    uint64_t sendCounts[MAX_RANK_SIZE] = {};
    uint64_t sendDispls[MAX_RANK_SIZE] = {};
    uint64_t recvCounts[MAX_RANK_SIZE] = {};
    uint64_t recvDispls[MAX_RANK_SIZE] = {};
    uint64_t maxCount = 0;
};

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
uint64_t input, uint64_t output, uint32_t rank, uint32_t rankSize, uint64_t xRankSize,  uint64_t yRankSize, uint64_t zRankSize, uint64_t len, \
uint32_t dataType, uint32_t reduceOp, uint32_t root, uint32_t tag, \
uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride, \
bool isOpBase, \
GM_ADDR headCountMem, \
GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter

#define EXTERN_KERNEL_ARGS_DEF_V2 \
KERNEL_ARGS_DEF, ExtraArgs extraArgs

#define KERNEL_ARGS_CALL \
buffIn, \
input, output, rank, rankSize, xRankSize, yRankSize, zRankSize, len, dataType, reduceOp, root, tag, \
inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride, outputRepeatStride, \
isOpBase, \
headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter

#define EXTERN_KERNEL_ARGS_CALL \
KERNEL_ARGS_CALL, extraArgs

#define KERNEL_CLASS_INIT \
buffIn, input, output,\
rank, rankSize, xRankSize, yRankSize, zRankSize, len, dataType, reduceOp, root, \
inputSliceStride, outputSliceStride, repeatNum, inputRepeatStride, outputRepeatStride, \
headCountMem, tailCountMem, addOneMem, counterMemSize, isEnableCounter


constexpr uint64_t AIV_FLAG_BUFFER_SIZE = 3 * 1024 * 1024; // aiv算子的flag区域大小
constexpr uint64_t CLEAR_BUFFER_OFFSET = 1024 * 1024; // 用于清空的aiv buffer的偏移
constexpr uint64_t SYNC_BUFFER_OFFSET = 2 * 1024 * 1024; // 用于sync的aiv buffer的偏移
constexpr uint64_t BUFFER_AREA = 1024 * 1024; // aiv算子的单独功能flag区域大小

constexpr uint64_t AIV_PING_PONG_FACTOR_TWO = 2;

constexpr uint32_t NUM_BLOCKS_FOUR_PER_RANK_A3 = 4;
constexpr uint32_t MAX_NUM_BLOCKS = 48;

constexpr uint64_t UB_ALIGN_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE = 32;
constexpr uint64_t UB_FLAG_SIZE_4 = UB_FLAG_SIZE * 4;
constexpr uint64_t UB_FLAG_SIZE_8 = UB_FLAG_SIZE * 8;
constexpr uint64_t UB_MAX_DATA_SIZE = 190 * 1024;
constexpr uint64_t UB_DB_DATA_BATCH_SIZE = UB_MAX_DATA_SIZE / 2;
constexpr uint32_t MaxBufferSize = 200 * 1024 * 1024;

constexpr uint64_t FLAG_SIZE = 32;
constexpr uint64_t ATOMIC_FLAG_SIZE = 512;
constexpr uint64_t FLAG_ONE_OFFSET = 0;
constexpr uint64_t FLAG_TWO_OFFSET = FLAG_SIZE;
constexpr uint64_t FLAG_THREE_OFFSET = FLAG_SIZE * 2;
constexpr uint64_t FLAG_FOUR_OFFSET = FLAG_SIZE * 3;
constexpr uint64_t FLAG_FIVE_OFFSET = FLAG_SIZE * 4;

constexpr uint64_t DOUBLE = 2;
constexpr uint64_t FLAG_BUF_NUM = 3;

// 当前每个kernel最多使用4组同步标记，这里预留6组
constexpr uint32_t MAX_FLAG_SIZE_PER_KERNEL = 6 * MAX_RANK_SIZE * FLAG_SIZE;

#define BASE_FLAG_OFFSET (MAX_FLAG_SIZE_PER_KERNEL)

class AivCommBase {
public:
    __aicore__ inline AivCommBase() {
    }

    __aicore__ inline void Init(GM_ADDR buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t rankSize, uint64_t xRankSize,  uint64_t yRankSize, uint64_t zRankSize,
                                uint64_t len,
                                uint32_t dataType, uint32_t reduceOp, uint32_t root,
                                uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
                                GM_ADDR headCountMem,
                                GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter,
                                bool useDoubleBuffer)
    {
        rank_ = rank;
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

    __aicore__ inline void BarrierAll();

    __aicore__ inline void BarrierForFirstOP();

    __aicore__ inline void SyncCoreAll(int32_t curTag);

    __aicore__ inline void WaitFlag(uint32_t targetRank, uint64_t flag_offset, int32_t curTag);

    __aicore__ inline void Record(uint32_t targetRank, uint64_t flag_offset, int32_t curTag);

    GM_ADDR GM_IN[MAX_RANK_SIZE];
    GM_ADDR GM_OUT[MAX_RANK_SIZE];
    uint64_t TOPO_[TOPO_LEN];
    uint32_t rank_;
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
    int32_t tag_;
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

    uint32_t localOffset;
    uint32_t multiOffset;
    uint32_t pingpongOffset;
    uint32_t countOffset;
    uint32_t seperateOffset;
};


__aicore__ inline void AivCommBase::Record(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
{
    d2hGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(GM_OUT[targetRank] + flag_offset * UB_ALIGN_SIZE));
    localTagTensor.SetValue(0, curTag);
    pipe_barrier(PIPE_ALL);
    DataCopyUB2GM(d2hGlobal, localTagTensor, UB_ALIGN_SIZE / sizeof(int32_t));
    pipe_barrier(PIPE_ALL);
}


__aicore__ inline void AivCommBase::WaitFlag(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
{
    d2hGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(GM_OUT[targetRank] + flag_offset * UB_ALIGN_SIZE));
    while (true) {
        DataCopyGM2UB(localTagTensor, d2hGlobal, UB_ALIGN_SIZE / sizeof(int32_t));
        pipe_barrier(PIPE_ALL);
        if (localTagTensor.GetValue(0) == curTag) {
            break;
        }
    }
}

__aicore__ inline void AivCommBase::BarrierForFirstOP()
{
    if (GetBlockIdx() == 0) {
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
			uint64_t flag_offset = BASE_FLAG_OFFSET + i * FLAG_SIZE;
            Record(rank_, flag_offset / UB_ALIGN_SIZE, DOUBLE);
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            uint64_t flag_offset = BASE_FLAG_OFFSET + rank_ * FLAG_SIZE;
            WaitFlag(i, flag_offset / UB_ALIGN_SIZE, DOUBLE);
        }
    }
}

__aicore__ inline void AivCommBase::SyncCoreAll(int32_t curTag)
{
    pipe_barrier(PIPE_ALL);
    uint64_t flag_offset = SYNC_CORE_OFFSET + GetBlockIdx() * FLAG_SIZE;
    Record(rank_, flag_offset / UB_ALIGN_SIZE, curTag);

    pipe_barrier(PIPE_ALL);
    for (int i = 0; i < MAX_NUM_BLOCKS; i++) {
        uint64_t flag_offset = SYNC_CORE_OFFSET + i * FLAG_SIZE;
        WaitFlag(rank_, flag_offset / UB_ALIGN_SIZE, curTag);
    }
}

__aicore__ inline void AivCommBase::BarrierAll()
{
    SyncAll<true>();
    if (GetBlockIdx() == 0) {
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            uint64_t flag_offset = BASE_FLAG_OFFSET + rank_ * FLAG_SIZE;
            Record(i, flag_offset / UB_ALIGN_SIZE, 1);
        }
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < rankSize_; i++) {
            uint64_t flag_offset = BASE_FLAG_OFFSET + i * FLAG_SIZE;
            WaitFlag(rank_, flag_offset / UB_ALIGN_SIZE, 1);
            Record(rank_, flag_offset / UB_ALIGN_SIZE, 0);
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

// 910B支持的Atomic数据类型
#define AIV_ATOMIC_DATA_TYPE_DEF(func) \
    func(float); \
    func(half); \
    func(int16_t); \
    func(int32_t); \
    func(int8_t); \
    func(bfloat16_t)

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
    func(int64_t)

#endif  /* AIV_COMMUNICATION_BASE_V2_H */
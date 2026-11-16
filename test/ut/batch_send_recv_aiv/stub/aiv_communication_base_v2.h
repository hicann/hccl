/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_AIV_TEST_COMMUNICATION_BASE_H
#define BSR_AIV_TEST_COMMUNICATION_BASE_H

#include "aiv_defines.h"

// AivCommBase 的 CPU 版：只保留 kernel 真正用到的语义，但 Record / WaitFlag / GetGmIn / GetGmOut /
// InitBuffArray 与生产实现逐句同构 —— 被测的正是"flag 落在哪个地址、值是什么、tag 奇偶选哪个半区"，
// 这些表达式一旦走样，断言就失去意义。
// 单线程模型里跨 rank 的先后由测试自己排列，所以 WaitFlag 不做真阻塞：读一次，记录命中与否即返回。
class AivCommBase {
public:
    __aicore__ inline AivCommBase() {}

    // 与生产同构：buffersIn 表在块首，buffersOut 表在 BUFFER_OUT_ADDR_OFFSET 处，两表都存各 rank 的块地址。
    __aicore__ inline void InitBuffArray(GM_ADDR buffIn, bool pingpong = false)
    {
        __gm__ uint64_t* base = (__gm__ uint64_t*)buffIn;
        __gm__ uint64_t* outBase = base + BUFFER_OUT_ADDR_OFFSET / sizeof(uint64_t);
        if (!pingpong) {
            gmInBase_ = base;
            gmInOffset_ = 0;
            gmOutBase_ = outBase;
            gmOutOffset_ = tag_ % 2 == 0 ? FLAG1_OFFSET : FLAG2_OFFSET;
        } else {
            gmInBase_ = outBase;
            gmInOffset_ = tag_ % 2 == 0 ? GM_OUT_PING_OFFSET : GM_OUT_PONG_OFFSET;
            gmOutBase_ = outBase;
            gmOutOffset_ = tag_ % 2 == 0 ? FLAG1_OFFSET : FLAG2_OFFSET;
        }
        myGmIn_ = GetGmIn(rank_);
        myGmOut_ = GetGmOut(rank_);
        pipe_barrier(PIPE_ALL);
    }

    // 表项里存的是虚构的块地址，只能经伪 GM 读出来（绝不能真的解引用）。
    __aicore__ inline GM_ADDR GetGmIn(uint32_t rank)
    {
        const uint64_t cell = reinterpret_cast<uint64_t>(gmInBase_) + rank * sizeof(uint64_t);
        return reinterpret_cast<GM_ADDR>(static_cast<uintptr_t>(bsr_aiv_test::GmRead64(cell) + gmInOffset_));
    }

    __aicore__ inline GM_ADDR GetGmOut(uint32_t rank)
    {
        const uint64_t cell = reinterpret_cast<uint64_t>(gmOutBase_) + rank * sizeof(uint64_t);
        return reinterpret_cast<GM_ADDR>(static_cast<uintptr_t>(bsr_aiv_test::GmRead64(cell) + gmOutOffset_));
    }

    __aicore__ inline bool IsIdleCore() const { return blockIdx_ >= static_cast<uint32_t>(numBlocks_); }

    __aicore__ inline void Record(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
    {
        const uint64_t addr = FlagAddress(targetRank, flag_offset);
        bsr_aiv_test::GmWrite32(addr, static_cast<uint32_t>(curTag));
        bsr_aiv_test::Event event{bsr_aiv_test::EventKind::RECORD};
        event.dst = addr;
        event.target = targetRank;
        event.slot = flag_offset;
        event.tag = curTag;
        bsr_aiv_test::Push(event);
    }

    __aicore__ inline void WaitFlag(uint32_t targetRank, uint64_t flag_offset, int32_t curTag)
    {
        const uint64_t addr = FlagAddress(targetRank, flag_offset);
        const int32_t actual = static_cast<int32_t>(bsr_aiv_test::GmRead32(addr));
        bsr_aiv_test::Event event{bsr_aiv_test::EventKind::WAIT};
        event.dst = addr;
        event.target = targetRank;
        event.slot = flag_offset;
        event.tag = curTag;
        event.actual = actual;
        event.matched = (actual == curTag);
        bsr_aiv_test::Push(event);
        if (!event.matched) {
            bsr_aiv_test::store().waitMisses++;
        }
    }

    template <bool lock = false>
    __aicore__ inline void SyncAll()
    {
        bsr_aiv_test::Push({bsr_aiv_test::EventKind::SYNC});
    }

    // 卡死检测计数钩子：与生产 AivCommBase 同名同参。BSR 入口在 InitBsr 末尾发 head、kernel 末尾发
    // tail，均只由 block0 发射；stub 只统计发射次数，不模拟 GM 归约。
    __aicore__ inline void
    InitStuckCounter(__gm__ int32_t* head, __gm__ int32_t* tail, __gm__ int32_t* addOne, bool enable)
    {
        stuckHead_ = head;
        stuckTail_ = tail;
        stuckAddOne_ = addOne;
        stuckEnabled_ = enable && (head != nullptr) && (tail != nullptr) && (addOne != nullptr);
        HeadCounter();
    }

    __aicore__ inline void HeadCounter()
    {
        if (stuckEnabled_ && blockIdx_ == 0) {
            ++bsr_aiv_test::store().headCounters;
        }
    }

    __aicore__ inline void TailCounter()
    {
        if (stuckEnabled_ && blockIdx_ == 0) {
            ++bsr_aiv_test::store().tailCounters;
        }
    }

    __aicore__ inline void SyncBeforeTail()
    {
        if (stuckEnabled_) {
            ++bsr_aiv_test::store().syncBeforeTail;
        }
    }

    template <typename T>
    __aicore__ inline void CpGM2GM(__gm__ T* outputGM, __gm__ T* inputGM, uint64_t count)
    {
        const uint64_t dst = reinterpret_cast<uint64_t>(outputGM);
        const uint64_t src = reinterpret_cast<uint64_t>(inputGM);
        const uint64_t bytes = count * sizeof(T);
        bsr_aiv_test::CopyBytes(dst, src, bytes);
        bsr_aiv_test::Event event{bsr_aiv_test::EventKind::COPY};
        event.dst = dst;
        event.src = src;
        event.bytes = bytes;
        bsr_aiv_test::Push(event);
    }

    // 与生产一致：flag 地址 = GetGmOut(target) + slot * FLAG_SIZE（GetGmOut 里已含半区偏移）。
    __aicore__ inline uint64_t FlagAddress(uint32_t targetRank, uint64_t flagOffset)
    {
        return reinterpret_cast<uint64_t>(GetGmOut(targetRank)) + flagOffset * FLAG_SIZE;
    }

    __gm__ uint64_t* gmInBase_ = nullptr;
    uint64_t gmInOffset_ = 0;
    __gm__ uint64_t* gmOutBase_ = nullptr;
    uint64_t gmOutOffset_ = 0;
    GM_ADDR myGmIn_ = nullptr;
    GM_ADDR myGmOut_ = nullptr;
    uint32_t rank_ = 0;
    uint32_t sendRecvRemoteRank_ = 0;
    uint32_t root_ = 0;
    uint32_t rankSize_ = 1;
    uint32_t reduceOp_ = 0;
    uint32_t dataType_ = 0;
    uint32_t unitSize_ = 0;

    uint64_t input_ = 0;
    uint64_t output_ = 0;
    uint64_t cclBufferSize_ = 0;

    uint64_t len_ = 0;
    uint32_t tag_ = TAG_INIT_VALUE;
    uint32_t curTag_ = 0;
    int32_t numBlocks_ = 0;
    // 与生产一样在构造期取核号：Process / BatchSendRecvBarrierAll 都直接信这个值。
    uint32_t blockIdx_ = AscendC::GetBlockIdx();

    uint64_t inputSliceStride_ = 0;
    uint64_t outputSliceStride_ = 0;
    uint64_t repeatNum_ = 0;
    uint64_t inputRepeatStride_ = 0;
    uint64_t outputRepeatStride_ = 0;

    bool useDoubleBuffer_ = true;

    __gm__ int32_t* stuckHead_ = nullptr;
    __gm__ int32_t* stuckTail_ = nullptr;
    __gm__ int32_t* stuckAddOne_ = nullptr;
    bool stuckEnabled_ = false;

    AscendC::TPipe pipe;
    AscendC::TBuf<> localFlagBuf;
    AscendC::LocalTensor<int32_t> localSetTensor;
    AscendC::LocalTensor<int32_t> localCheckTensor;
    AscendC::LocalTensor<int32_t> localCheckGETensor;
    AscendC::LocalTensor<int32_t> localGetTensor;
    AscendC::LocalTensor<int32_t> localTagTensor;
    AscendC::GlobalTensor<int32_t> d2hGlobal;

    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> inOutQue;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> inQueueX;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> inQueueY;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> outQueueZ;

    uint32_t localOffset = 0;
    uint32_t multiOffset = 0;
    uint32_t pingpongOffset = 0;
    uint32_t countOffset = 0;
    uint32_t separateOffset = 0;
};

#endif // BSR_AIV_TEST_COMMUNICATION_BASE_H

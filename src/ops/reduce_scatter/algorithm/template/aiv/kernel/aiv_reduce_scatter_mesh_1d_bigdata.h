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

// 对称路径 UB 预算：只有 inOutQue(194560B) 在 UB 内真实可用 —— base Init 的 localFlagBuf(2560)
// 加 inOutQue 与 3*48640 已超出 256KB 池，inQueueY 仅前 16384B 有效，超写报 507035（MTE2 write UB out of bounds）。
constexpr uint64_t RS_SYM_UB_BYTES = 194560ULL;
// 单 tile 目标粒度（UB 预算只是硬上限）：tile 过大时 tile 内 (1+peers) 路 DMA 排队、跨 tile 读写重叠变少
constexpr uint64_t RS_SYM_TILE_TARGET = 13312ULL;
// peers<=1（np2）只有本端 + 1 路对端，无 tile 内排队，不做粒度收敛、用大 tile 摊薄同步开销
constexpr uint32_t RS_SYM_TILE_MIN_PEERS = 3U;
// 流水形态分档
constexpr uint32_t RS_SYM_DEEP3_MAX_PEERS = 3U; // np4 及更小：3 笔全下发 + 双缓冲 acc
constexpr uint32_t RS_SYM_ALLPF_PEERS = 7U;     // np8：7 笔全下发、单 acc
constexpr uint32_t RS_SYM_DEEP3_STAGES = 3U;
constexpr uint32_t RS_SYM_WIDE_STAGES = 8U; // np16 及更多：8 笔常驻 + 轮转补位
// 事件 ID 分配（AIV 只有 ID0~7，必须互不重叠）：
//   ID0..3 = 对端读分组事件（每 readGroup 笔读共用一个 MTE2_V）
//   ID4..5 = acc 块直写收口（MTE3_MTE2，accSlots<=2）
//   ID6    = 累加落地（V_MTE3）
//   ID7    = stage 槽复用前等 vector 读完（V_MTE2）
constexpr uint32_t RS_SYM_MAX_READ_EVENTS = 4U;
constexpr uint32_t RS_SYM_ACC_EVENT_BASE = 4U;
constexpr uint32_t RS_SYM_V_MTE3_EVENT = 6U;
constexpr uint32_t RS_SYM_V_MTE2_EVENT = 7U;

// 对称路径的单核流水形态
struct SymPipe {
    uint32_t accSlots;   // 累加器块数：1=单缓冲，2=双缓冲
    uint32_t stageSlots; // 对端暂存块数 = 同时在飞的 DMA 笔数
    uint32_t readGroup;  // 每几笔对端读共用一个 MTE2_V 事件（= ceil(stageSlots / 事件数上限)）
    uint64_t maxChunk;   // 单块元素数（UB 预算 / 块数，再按 tile 目标粒度收敛）
};

template <typename T>
class AivReduceScatterMesh1DBigData : public AivCommBase {
    constexpr static uint64_t stageNum = 2; // 生产者 消费者
    constexpr static uint64_t TAG_FLAG_SIZE = 8;
    uint64_t symLoopOffset_ = 0; // 对称路径：input_ 相对本 rank 对称窗口基址的偏移（host 分循环后需自己补）
    uint32_t symPeers_ = 0; // 对称路径：对端数 = rankSize_ - 1

public:
    __aicore__ inline AivReduceScatterMesh1DBigData() {}

    // 应该第一步要先用rankSize个核去切数据，一种方案是cutNum是多少就切多少份，另一个方案是切细一点，去多读几次，用足够多的流水去掩盖第一步和第三步的时间
    // 第二部是并行去读对端的数据
    // 第三步，之前把数据搬回来的那个卡，搬回来之后就没事干了，顺路就去把reduce做了

    __aicore__ inline void Producer(uint64_t len, uint64_t inputStride)
    {
        // 每个核负责len数据，一块数据切成cutNum块
        uint64_t targetRank = blockIdx_;

        uint64_t dataPerCore = len / cutNum;
        uint64_t remainder = len % cutNum;

        for (int64_t coreIndex = 0; coreIndex < cutNum; coreIndex++) {
            // 数据对不齐的情况
            uint64_t sendCurCount = 0;
            uint64_t innerDispls = 0;
            if (coreIndex < remainder) { // 这部分核需要多处理一个数据
                innerDispls = coreIndex * dataPerCore + coreIndex;
                sendCurCount = dataPerCore + 1;
            } else {
                innerDispls = coreIndex * dataPerCore + remainder;
                sendCurCount = dataPerCore;
            }

            if (sendCurCount > 0) {
                uint64_t usrInOffset = input_ + targetRank * inputStride + innerDispls * sizeof(T);
                uint64_t cclInOffset
                    = reinterpret_cast<uint64_t>(myGmIn_) + (targetRank * len + innerDispls) * sizeof(T);
                CpGM2GM((__gm__ T*)cclInOffset, (__gm__ T*)usrInOffset, sendCurCount);
                PipeBarrier<PIPE_ALL>();
                // 每个核写flag
                Record(rank_, blockIdx_ * cutNum + coreIndex, curTag);
            }
        }
    }

    __aicore__ inline void Consumer(uint64_t len)
    {
        // stage2的核分成rankSize份，每一份有cutNum，并发去读其他所有卡的数据
        uint64_t blockInedx = blockIdx_ - coreNumStage1;
        uint32_t coreNumPerRank = coreNumStage2 / rankSize_;
        uint32_t targetRank = blockInedx / coreNumPerRank;
        uint32_t coreIndex = (blockInedx - (targetRank * coreNumPerRank)) % coreNumPerRank;
        // 每个核分数据
        uint64_t dataPerCore = len / coreNumPerRank;
        uint64_t remainder = len % coreNumPerRank;
        // 数据对不齐的情况
        uint64_t innerDispls = 0;
        uint64_t recvCurCount = 0;
        if (coreIndex < remainder) { // 这部分核需要多处理一个数据
            innerDispls = coreIndex * dataPerCore + coreIndex;
            recvCurCount = dataPerCore + 1;
        } else {
            innerDispls = coreIndex * dataPerCore + remainder;
            recvCurCount = dataPerCore;
        }

        if (recvCurCount > 0) {
            // 先去wait flag，然后把数据拷贝过来
            WaitFlag(targetRank, rank_ * coreNumPerRank + coreIndex, curTag);

            uint64_t srcCclInOffset
                = reinterpret_cast<uint64_t>(GetGmIn(targetRank)) + (rank_ * len + innerDispls) * sizeof(T);
            uint64_t dstCclOutOffset
                = reinterpret_cast<uint64_t>(myGmIn_) + cclBufferStride + (targetRank * len + innerDispls) * sizeof(T);
            if (targetRank != 0) { // targetRank == 0 的那张卡，一会直接到output就可以了
                CpGM2GM((__gm__ T*)dstCclOutOffset, (__gm__ T*)srcCclInOffset, recvCurCount);
                PipeBarrier<PIPE_ALL>();
            }
            // 要写一个flag，后面做reduce需要用到
            Record(rank_, rankSize_ * coreNumPerRank + blockInedx, curTag);

            // 把所有数据放到cclBuffer之后，开始本地做reduce
            // 前面做过数据搬运的core现在是空闲状态，顺路帮忙做一下reduce
            if (targetRank == 0) {
                // 归约初值：rank0 数据（srcCclInOffset = card0 槽 rank_）已由上面的 WaitFlag(targetRank, ...)
                // 保证就位， 直接覆盖 output 作为累加器初值
                uint64_t usrOutOffset = output_ + innerDispls * sizeof(T);
                CpGM2GM((__gm__ T*)usrOutOffset, (__gm__ T*)srcCclInOffset, recvCurCount);
                PipeBarrier<PIPE_ALL>();
                // 树形折叠（与 Aicpu 保序 RunLocalReduce 同序）：slot(dst) += slot(src)，dst==0 累加进 output_，dst>=1
                // 累加进 cclBuffer 槽 零存储 flag 等待：源槽仅当 src >= k 时等待（[1,k)
                // 已在作为目的槽被折入时等待过）； 目的槽仅首次被折入时等待（新目的槽只出现在第 0/1 轮）
                uint32_t curBlocks = static_cast<uint32_t>(rankSize_);
                uint32_t m0 = GetLargestPowerOf2LessThan(curBlocks);
                uint32_t firstRoundDstNum = curBlocks - m0;
                uint32_t k = (firstRoundDstNum > (m0 >> 1)) ? firstRoundDstNum : (m0 >> 1);
                uint32_t round = 0;
                while (curBlocks > 1) {
                    uint32_t M = GetLargestPowerOf2LessThan(curBlocks);
                    if (M == 0) {
                        break;
                    }
                    for (uint32_t src = M; src < curBlocks; ++src) {
                        uint32_t dst = src % M;
                        if (src >= k) {
                            WaitFlag(rank_, rankSize_ * coreNumPerRank + src * coreNumPerRank + coreIndex, curTag);
                        }
                        if (round == 0 || (round == 1 && dst >= firstRoundDstNum)) {
                            WaitFlag(rank_, rankSize_ * coreNumPerRank + dst * coreNumPerRank + coreIndex, curTag);
                        }
                        uint64_t srcOff = reinterpret_cast<uint64_t>(myGmIn_) + cclBufferStride
                                          + (src * len + innerDispls) * sizeof(T);
                        if (dst == 0) {
                            CpGM2GM((__gm__ T*)usrOutOffset, (__gm__ T*)srcOff, recvCurCount, reduceOp_);
                        } else {
                            CpGM2GM(
                                (__gm__ T*)(reinterpret_cast<uint64_t>(myGmIn_) + cclBufferStride
                                            + (dst * len + innerDispls) * sizeof(T)),
                                (__gm__ T*)srcOff, recvCurCount, reduceOp_);
                        }
                        PipeBarrier<PIPE_ALL>();
                    }
                    curBlocks = M;
                    ++round;
                }
            }
        }
    }

    // UB 本地累加 acc = acc ⊕ tmp（⊕ 由 reduceOp_ 决定），用向量单元，避免 GM 原子加。
    // 整块下发、不分段：CHUNK_SIZE 分段只用于 Reduce64 的 int64 原子加模拟，分段反而多一笔下发开销。
    __aicore__ inline void SymReduceLocal(const LocalTensor<T>& acc, const LocalTensor<T>& tmp, uint32_t cnt)
    {
        if (reduceOp_ == HcclReduceOp::HCCL_REDUCE_SUM) {
            Add<T>(acc, acc, tmp, cnt);
        } else if (reduceOp_ == HcclReduceOp::HCCL_REDUCE_MAX) {
            Max<T>(acc, acc, tmp, cnt);
        } else if (reduceOp_ == HcclReduceOp::HCCL_REDUCE_MIN) {
            Min<T>(acc, acc, tmp, cnt);
        }
    }

    // 对称内存「全核零同步」：user in -> user out，不经 cclBuffer、不用 rank 间 ready flag（仅核内 MTE 事件
    // 流水 + 收尾一次 BarrierAll）。全部 numBlocks_ 核按输出连续区间切分，UB 内向量累加，不用 GM 原子加。
    // 语义：本 rank 输出 = 各 rank input 中属于本 rank 的那一片之和；输出片只由归属 rank 计算（单点型），
    // 累加序固定（先本端，再按环序自 rank_+1 回绕），确定性成立。
    __aicore__ inline void ConsumerSym(uint64_t len, uint64_t inputStride)
    {
        if (IsIdleCore()) {
            return;
        }
        const uint64_t chunkStride = (inputStride > 0) ? (inputStride / sizeof(T)) : len;
        const uint64_t mySymIn = reinterpret_cast<uint64_t>(GetGmSymIn(rank_));
        symLoopOffset_ = (input_ >= mySymIn) ? (input_ - mySymIn) : 0;
        symPeers_ = (rankSize_ > 0) ? static_cast<uint32_t>(rankSize_ - 1) : 0;
        const uint32_t cores = static_cast<uint32_t>(numBlocks_);
        if (cores == 0) {
            return;
        }
        uint64_t myOff = 0;
        uint64_t myCnt = 0;
        SplitCoreRange(len, cores, myOff, myCnt);
        if (myCnt == 0) {
            return;
        }
        const SymPipe pipe = MakeSymPipe(symPeers_);
        // 均衡切分：把 myCnt 均分到各 tile（向上对齐 32B），避免出现「满 tile + 近空尾 tile」——
        // 每个 tile 都是「读 → 加 → 写」的串行链，尾 tile 的搬运延迟与满 tile 同级。
        const uint64_t nTiles = (myCnt + pipe.maxChunk - 1) / pipe.maxChunk;
        const uint64_t alignElem = UB_ALIGN_SIZE / sizeof(T);
        uint64_t chunk = (myCnt + nTiles - 1) / nTiles;
        chunk = (chunk + alignElem - 1) / alignElem * alignElem;
        chunk = (chunk > pipe.maxChunk) ? pipe.maxChunk : chunk;
        LocalTensor<T> ubBuf = inOutQue.AllocTensor<T>();
        // 本 rank 负责的 input 片偏移（元素）：片索引 == rank_
        RunSymTiles(pipe, ubBuf, rank_ * chunkStride + myOff, myOff, myCnt, chunk);
        inOutQue.FreeTensor(ubBuf);
    }

    // 本核负责的输出区间 [myOff, myOff+myCnt)，按 32B 对齐切分：myOff/myCnt 都是 alignElem 的整数倍，
    // 使本端读、对端读、结果写全部命中 SymDataCopy* 的 DataCopy 快路径（DataCopyPad 每笔固定开销明显更大）。
    __aicore__ inline void SplitCoreRange(uint64_t len, uint32_t cores, uint64_t& myOff, uint64_t& myCnt) const
    {
        const uint64_t alignElem = UB_ALIGN_SIZE / sizeof(T);
        const uint64_t units = len / alignElem;
        const uint64_t perCoreUnits = units / cores;
        if (perCoreUnits == 0) {
            // 数据量太小，按元素均分（此时本来就只有 1 个 tile，对齐收益可忽略）
            const uint64_t perCore = len / cores;
            const uint64_t rem = len % cores;
            myOff = blockIdx_ * perCore + ((blockIdx_ < rem) ? blockIdx_ : rem);
            myCnt = perCore + ((blockIdx_ < rem) ? 1 : 0);
            return;
        }
        const uint64_t remUnits = units % cores;
        myOff = (blockIdx_ * perCoreUnits + ((blockIdx_ < remUnits) ? blockIdx_ : remUnits)) * alignElem;
        myCnt = (perCoreUnits + ((blockIdx_ < remUnits) ? 1 : 0)) * alignElem;
        if (static_cast<uint32_t>(blockIdx_ + 1) == cores) {
            myCnt += len - units * alignElem; // 不足 32B 的零头并到最后一个核
        }
    }

    // 按对端数选流水形态（只是参数不同，共用一套循环）：
    //   peers<=3（np4 及更小）：acc×2 + stg×3，3 笔读一次全下发；
    //   peers==7（np8）：acc×1 + stg×7，7 笔全下发、无需补位；
    //   peers>=8（np16 及更多）：acc×1 + stg×8，8 笔常驻 + k+8 轮转补位。
    static __aicore__ inline SymPipe MakeSymPipe(uint32_t peers)
    {
        SymPipe pp;
        const bool deep3 = (peers <= RS_SYM_DEEP3_MAX_PEERS);
        pp.stageSlots
            = deep3 ? RS_SYM_DEEP3_STAGES : ((peers == RS_SYM_ALLPF_PEERS) ? RS_SYM_ALLPF_PEERS : RS_SYM_WIDE_STAGES);
        pp.accSlots = deep3 ? 2U : 1U; // np8/np16 用单 acc，腾出的块给 stage
        pp.readGroup = (pp.stageSlots + RS_SYM_MAX_READ_EVENTS - 1U) / RS_SYM_MAX_READ_EVENTS;
        if (pp.readGroup == 0U) {
            pp.readGroup = 1U;
        }
        const uint64_t blocks = static_cast<uint64_t>(pp.accSlots) + pp.stageSlots;
        uint64_t maxChunk = RS_SYM_UB_BYTES / blocks / UB_ALIGN_SIZE * UB_ALIGN_SIZE / sizeof(T);
        // 单 tile 目标粒度：一个 tile 内每核会一次下发 (1+peers) 路 DMA，路数多时 tile 过大会导致
        // ① 这组 DMA 在 tile 内互相排队串行化；② tile 数变少 → 跨 tile 的「本 tile 写回 / 下一 tile 读入」
        // 重叠变少。peers<=1（np2：只有本端 + 1 路对端，无 tile 内排队）保持大 tile 摊薄同步开销。
        if (peers >= RS_SYM_TILE_MIN_PEERS) {
            const uint64_t alignElem = UB_ALIGN_SIZE / sizeof(T);
            const uint64_t target = RS_SYM_TILE_TARGET / sizeof(T) / alignElem * alignElem;
            maxChunk = (target < maxChunk) ? target : maxChunk;
        }
        pp.maxChunk = maxChunk;
        return pp;
    }

    // 逐 tile：下发读 → 累加 → 直写。accSlots 轮之前要收口上一轮的直写（单缓冲=上一轮，双缓冲=两轮前），
    // 末尾再等所有在途直写落地。pending 记录「哪些读分组事件还挂着未收口」，跨 tile 归零。
    __aicore__ inline void RunSymTiles(
        const SymPipe& pipe, const LocalTensor<T>& ubBuf, uint64_t inChunkBase, uint64_t myOff, uint64_t myCnt,
        uint64_t chunk)
    {
        uint32_t ti = 0;
        uint32_t usedMask = 0;
        uint32_t pending = 0;
        uint64_t done = 0;
        while (done < myCnt) {
            const uint64_t cur = (myCnt - done > chunk) ? chunk : (myCnt - done);
            const uint64_t srcOff = (inChunkBase + done) * sizeof(T);
            const uint64_t dstOff = (myOff + done) * sizeof(T);
            const uint32_t bi = ti % pipe.accSlots;
            LocalTensor<T> acc = ubBuf[pipe.maxChunk * bi];
            if (ti >= pipe.accSlots) {
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(RS_SYM_ACC_EVENT_BASE + bi);
            }
            IssueTileReads(pipe, ubBuf, acc, cur, srcOff, pending);
            ReduceTilePeers(pipe, ubBuf, acc, cur, srcOff, pending);
            AscendC::SetFlag<HardEvent::V_MTE3>(RS_SYM_V_MTE3_EVENT);
            AscendC::WaitFlag<HardEvent::V_MTE3>(RS_SYM_V_MTE3_EVENT); // 累加落地才能直写
            CopySymOut(output_ + dstOff, acc, cur);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(RS_SYM_ACC_EVENT_BASE + bi); // 写留在 MTE3 管道，accSlots 轮后收口
            usedMask |= (1U << bi);
            ++ti;
            done += cur;
        }
        for (uint32_t s = 0; s < pipe.accSlots; ++s) {
            if ((usedMask & (1U << s)) != 0U) {
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(RS_SYM_ACC_EVENT_BASE + s);
            }
        }
    }

    // 下发本 tile 的读：本端片进 acc，对端片进 stage 块（最多 stageSlots 笔），每 readGroup 笔共用一个事件。
    __aicore__ inline void IssueTileReads(
        const SymPipe& pipe, const LocalTensor<T>& ubBuf, const LocalTensor<T>& acc, uint64_t cur, uint64_t srcOff,
        uint32_t& pending)
    {
        CopySymIn(acc, input_ + srcOff, static_cast<uint32_t>(cur));
        const uint32_t nIssue = (pipe.stageSlots < symPeers_) ? pipe.stageSlots : symPeers_;
        const uint32_t g = pipe.readGroup;
        pending = 0;
        for (uint32_t p = 0; p < nIssue; ++p) {
            StagePeerRead(pipe, ubBuf, p, p, cur, srcOff);
            const bool groupEnd = (((p % g) == (g - 1U)) || (p + 1U == nIssue));
            if (groupEnd) {
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + (p / g));
                pending |= (1U << (p / g));
            }
        }
    }

    // 按环序等「该槽所在分组」的读齐 → 累加；peers > stageSlots 时用 k+stageSlots 号对端补位，
    // 保持 stageSlots 笔在飞。分组内任一槽补过位就要重打组事件（否则下一轮会用到没等过的数据）。
    __aicore__ inline void ReduceTilePeers(
        const SymPipe& pipe, const LocalTensor<T>& ubBuf, const LocalTensor<T>& acc, uint64_t cur, uint64_t srcOff,
        uint32_t& pending)
    {
        const uint32_t g = pipe.readGroup;
        const uint32_t s = pipe.stageSlots;
        bool groupRefilled = false;
        for (uint32_t k = 0; k < symPeers_; ++k) {
            const uint32_t hb = k % s;
            const uint32_t gi = hb / g;
            if ((hb % g) == 0U) { // 组首：等整组读齐（只有挂过事件才等）
                if ((pending & (1U << gi)) != 0U) {
                    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0 + gi);
                    pending &= ~(1U << gi);
                }
                groupRefilled = false;
            }
            SymReduceLocal(acc, ubBuf[pipe.maxChunk * (pipe.accSlots + hb)], static_cast<uint32_t>(cur));
            if (k + s < symPeers_) {
                AscendC::SetFlag<HardEvent::V_MTE2>(RS_SYM_V_MTE2_EVENT);
                AscendC::WaitFlag<HardEvent::V_MTE2>(RS_SYM_V_MTE2_EVENT); // V 读完该 stage 才能复用
                StagePeerRead(pipe, ubBuf, hb, k + s, cur, srcOff);
                groupRefilled = true;
            }
            const bool groupEnd = (((hb % g) == (g - 1U)) || (hb + 1U == s));
            if (groupEnd && groupRefilled) {
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + gi);
                pending |= (1U << gi);
            }
        }
    }

    // 把「环序第 peerIdx 个对端」（rank_+peerIdx+1）的片读进 stage[slot]
    __aicore__ inline void StagePeerRead(
        const SymPipe& pipe, const LocalTensor<T>& ubBuf, uint32_t slot, uint32_t peerIdx, uint64_t cur,
        uint64_t srcOff)
    {
        const uint32_t peer = (rank_ + peerIdx + 1U) % static_cast<uint32_t>(rankSize_);
        CopySymPeerIn(
            ubBuf[pipe.maxChunk * (pipe.accSlots + slot)], peer, symLoopOffset_ + srcOff, static_cast<uint32_t>(cur));
    }

    __aicore__ inline void CopySymIn(const LocalTensor<T>& dst, uint64_t srcAddr, uint32_t cnt)
    {
        GlobalTensor<T> g;
        g.SetGlobalBuffer((__gm__ T*)srcAddr, cnt);
        DataCopyGM2UBByAddr(dst, g, cnt, srcAddr);
    }

    // 对端 user input（对称 VA）→ UB；peerOff 已含 loopOffset
    __aicore__ inline void CopySymPeerIn(const LocalTensor<T>& dst, uint32_t peer, uint64_t peerOff, uint32_t cnt)
    {
        CopySymIn(dst, reinterpret_cast<uint64_t>(GetGmSymIn(peer)) + peerOff, cnt);
    }

    __aicore__ inline void CopySymOut(uint64_t dstAddr, const LocalTensor<T>& src, uint32_t cnt)
    {
        GlobalTensor<T> g;
        g.SetGlobalBuffer((__gm__ T*)dstAddr, cnt);
        DataCopyUB2GMByAddr(g, src, cnt, dstAddr);
    }

    // 对称路径的本地累加只实现 SUM/MAX/MIN；其它归约（如 PROD）必须落回基线两段式，
    // 否则 SymReduceLocal 不生效，输出只剩本端 input 片，静默丢对端数据。
    __aicore__ inline bool UseSymPath(uint64_t len) const
    {
        if ((symInEnable_ != 1) || (len == 0)) {
            return false;
        }
        return (reduceOp_ == HcclReduceOp::HCCL_REDUCE_SUM) || (reduceOp_ == HcclReduceOp::HCCL_REDUCE_MAX)
               || (reduceOp_ == HcclReduceOp::HCCL_REDUCE_MIN);
    }

    __aicore__ inline void Process(uint32_t sliceId, uint64_t len, uint64_t inputStride)
    {
        // 对称路径必须先于下面的两段式核数门禁：否则低核数（numBlocks <= 2*rankSize）会直接 return，
        // 一个元素都不搬（静默无输出）。
        if (UseSymPath(len)) {
            ConsumerSym(len, inputStride);
            return;
        }
        // 非对称两段式：核数大于等于2倍ranksize
        uint64_t rankMultiple = 2;
        curStageCoreNum = numBlocks_ / rankSize_ * rankSize_; // 总的核数
        if (curStageCoreNum < rankMultiple * rankSize_) {
            return;
        }
        coreNumStage1 = rankSize_;
        coreNumStage2 = curStageCoreNum - coreNumStage1;
        cutNum = coreNumStage2 / rankSize_;
        curTag = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        cclBufferStride = len * rankSize_ * sizeof(T); // cclIn和cclOut的分界
        if (blockIdx_ < coreNumStage1) {
            Producer(len, inputStride);
        } else if (blockIdx_ < curStageCoreNum) {
            Consumer(len);
        }
    }

    uint64_t curStageCoreNum;
    uint64_t coreNumStage1;
    uint64_t coreNumStage2;
    uint64_t cutNum;
    uint64_t curTag;
    uint64_t cclBufferStride;
};

template <typename T>
__aicore__ inline void AivReduceScatterV2Mesh1DBigData(KERNEL_ARGS_DEF)
{
    AivReduceScatterMesh1DBigData<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    const bool symPath = op.UseSymPath(len);
    // sliceId 是模板对象内每次 launch 自增的序号，每个算子从 1 开始，故 1 即本算子首轮：
    // 首轮已确认全 rank 输入就绪，后续轮次读同一份输入的其他区段，由上一轮出口 BarrierAll 收口即可。
    const bool needEntrySync = symPath && (sliceId == 1);
    if (needEntrySync) {
        op.BarrierAll(0); // 直读对端 input 前确认各 rank 本轮输入已提交
        SyncAll<true>();  // BarrierAll 只确认本核负责的 rank 子集，须补核间同步
    }
    op.Process(sliceId, len, inputSliceStride);
    op.BarrierAll(needEntrySync ? 1U : 0U);
}

template <typename T>
__aicore__ inline void AivReduceScatterV2Mesh1DBigDataSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    AivReduceScatterMesh1DBigData<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT);

    uint64_t maxCountPerLoop = op.cclBufferSize_ / (2 * op.rankSize_) / UB_ALIGN_SIZE * UB_ALIGN_SIZE / sizeof(T);
    uint64_t countLeft = op.len_;
    int32_t loopTag = op.tag_;
    uint32_t barrierSeq = 0;

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        op.Process(loopTag, curCount, op.inputSliceStride_);
        op.BarrierAll(barrierSeq);
        ++barrierSeq;

        countLeft -= curCount;
        op.input_ += curSize;
        op.output_ += curSize;
        loopTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
}

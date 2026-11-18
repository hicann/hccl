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
// 加 inOutQue 与 3*48640 已超出 256KB 池，inQueueY 仅前 16384B 有效、outQueueZ 完全在 UB 外，
// 超写会报 507035（MTE2 write UB out of bounds）。
constexpr uint64_t AR_SYM_UB_BYTES = 194560ULL;
// 中等数据量时单核 tile 下限（按字节，避免随 dtype 变化）：再小只剩 DMA 固定开销。
constexpr uint64_t AR_SYM_MIN_TILE_BYTES = 8192ULL;
// 单 tile 目标粒度（按字节）：一个 tile 内每核要下发 (1+peers) 笔读 + rankSize 笔写，tile 过大则
// 这组 DMA 在 tile 内排队串行化、跨 tile 的读/写重叠也变少。取值与 RS 保持一致。
constexpr uint64_t AR_SYM_TILE_TARGET_BYTES = 13312ULL;
constexpr uint32_t AR_SYM_READS_PER_EVENT = 3U;
// 读分组事件 ID 只有 0~7，且与 V_MTE3 等固定事件共用编号空间，故分组数收敛在 4 以内：
// peers 超过 AR_SYM_READS_PER_EVENT * AR_SYM_MAX_READ_EVENTS 时按 ceil(peers/4) 扩大每组读的数量。
constexpr uint32_t AR_SYM_MAX_READ_EVENTS = 4U;
// 写段（acc）双缓冲槽数：tile t 的广播写在 tile t+1 的累加/预取期间继续在飞
constexpr uint32_t AR_SYM_WRITE_SLOTS = 2U;
// rankSize >= AR_SYM_NP16_RANK 时，tile 数少于 AR_SYM_NP16_MIN_TILES 回退两 shot（固定开销吃光收益）
constexpr uint32_t AR_SYM_NP16_RANK = 16U;
constexpr uint64_t AR_SYM_NP16_MIN_TILES = 4ULL;
constexpr uint32_t AR_SYM_TWOSHOT_MIN_RATIO = 2U;

template <typename T>
class AivAllReduceMesh1DTwoShot : public AivCommBase {
public:
    __aicore__ inline AivAllReduceMesh1DTwoShot() {}

    __aicore__ inline void Prepare(uint32_t sliceId)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        coreIdx = blockIdx_;
        coreNum = numBlocks_;
        count = len_;

        // 将core分组
        producerNum = rankSize_;
        // 一个producer对应几个consumer，整数倍；核数不足 2*rankSize 时没有 consumer
        groupSize = (coreNum > producerNum) ? ((coreNum - producerNum) / producerNum) : 0;
        consumerNum = groupSize * producerNum;
        // 消费者为 0（核数 < 2*rankSize）时两 shot 切分不可用，对称路径按 tile 均分、不依赖切分结果，
        // 直接返回避免后续 count / sliceNum 除零
        if (consumerNum == 0) {
            return;
        }
        // 返回多余的核
        if (coreIdx >= producerNum + consumerNum) {
            return;
        }
        // 计算核所在组的索引，以及和在组内的索引
        if (coreIdx < producerNum) {
            groupIdx = coreIdx;
            idxInGroup = 0;
        } else {
            groupIdx = (coreIdx - producerNum) / groupSize;
            idxInGroup = (coreIdx - producerNum) % groupSize;
        }

        // 数据切分
        sliceNum = consumerNum;
        smallSliceCount = count / sliceNum;
        smallSliceSize = smallSliceCount * sizeof(T);
        bigSliceNum = count % sliceNum; // 大片数据的数量
        bigSliceCount = smallSliceCount + 1;
        bigSliceSize = bigSliceCount * sizeof(T);
        // HcclBuffer上的数据按照chunkSize对齐存放，便于计算偏移
        if (bigSliceNum == 0) {
            chunkSize = smallSliceSize;
        } else {
            chunkSize = bigSliceSize;
        }

        // 基础数据偏移计算
        storeBuffOffset = 0;                     // HcclBuffer上存储数据的Buffer偏移
        reduceBuffOffset = chunkSize * sliceNum; // HcclBuffer上Reduce计算的Buffer偏移

        // 基础flag偏移计算
        preCopyFlagOffset = 0;
        getRemoteFlagOffset = preCopyFlagOffset + consumerNum;
        localReduceFlagOffset = getRemoteFlagOffset + consumerNum;
    }

    __aicore__ inline void Process()
    {
        // 对称内存：两侧窗口可用且 reduceOp 受支持时走零同步对称路径，否则落回两 shot 基线。
        if (UseSymPath()) {
            SymProcess();
            return;
        }
        ReduceScatter();
        AllGather();
    }

    // rankSize >= AR_SYM_NP16_RANK 时才可能回退；回退目标两 shot 要求核数 >= 2*rankSize，
    // 否则消费者为 0 数据搬不动，这种核数下仍留在对称路径（按 tile 均分，任意核数可跑）。
    __aicore__ inline bool SymSmallFallback() const
    {
        if (rankSize_ < AR_SYM_NP16_RANK) {
            return false;
        }
        if (numBlocks_ < AR_SYM_TWOSHOT_MIN_RATIO * rankSize_) {
            return false;
        }
        // 与 MakeSymPlan 同源估算 tile；只依赖全 rank 一致的量，避免边界档位各卡判定分叉。
        const uint64_t symChunkCount = len_ / rankSize_;
        const uint32_t symPeers = static_cast<uint32_t>(rankSize_ - 1U);
        const uint64_t symMaxChunk = CalcSymTileElems(symChunkCount, symPeers, static_cast<uint32_t>(numBlocks_));
        const uint64_t symTiles = (symChunkCount + symMaxChunk - 1) / symMaxChunk;
        return (symTiles < AR_SYM_NP16_MIN_TILES);
    }

    // 走对称路径的条件：in/out 两侧窗口都就绪 + reduceOp 受 SymReduceLocal 支持 + 未触发小档回退。
    // SymReduceLocal 对 SUM/MAX/MIN 之外的算子不做任何累加（会静默丢对端数据），必须落回基线路径。
    __aicore__ inline bool UseSymPath() const
    {
        if ((symInEnable_ != 1) || (symOutEnable_ != 1)) {
            return false;
        }
        const bool opSupported = (reduceOp_ == HcclReduceOp::HCCL_REDUCE_SUM)
                                 || (reduceOp_ == HcclReduceOp::HCCL_REDUCE_MAX)
                                 || (reduceOp_ == HcclReduceOp::HCCL_REDUCE_MIN);
        return opSupported && !SymSmallFallback();
    }

    // UB 本地累加 acc = acc ⊕ tmp（⊕ 由 reduceOp_ 决定）：用向量单元替代 GM 原子加，按 CHUNK_SIZE 分段下发。
    __aicore__ inline void SymReduceLocal(const LocalTensor<T>& acc, const LocalTensor<T>& tmp, uint32_t cnt)
    {
        uint32_t done = 0;
        while (done < cnt) {
            uint32_t step
                = ((cnt - done) > static_cast<uint32_t>(CHUNK_SIZE)) ? static_cast<uint32_t>(CHUNK_SIZE) : (cnt - done);
            LocalTensor<T> accStep = acc[done];
            LocalTensor<T> tmpStep = tmp[done];
            if (reduceOp_ == HcclReduceOp::HCCL_REDUCE_SUM) {
                Add<T>(accStep, accStep, tmpStep, step);
            } else if (reduceOp_ == HcclReduceOp::HCCL_REDUCE_MAX) {
                Max<T>(accStep, accStep, tmpStep, step);
            } else if (reduceOp_ == HcclReduceOp::HCCL_REDUCE_MIN) {
                Min<T>(accStep, accStep, tmpStep, step);
            }
            done += step;
        }
    }

    // 每核执行计划：一次算好，所有卡数共用同一套流水结构
    struct SymPlan {
        uint64_t chunkStart;   // 本 rank 负责的输出 chunk 在 output 内的元素偏移
        uint64_t chunkCount;   // 本 rank 负责的输出 chunk 元素数
        uint64_t loopOffsetIn; // input_ 相对本 rank 对称窗口基址的偏移（host 分循环后需自己补）
        uint64_t loopOffsetOut;
        uint64_t blkElems; // 单 tile 元素数
        uint64_t tBegin;   // 本核负责的 tile 区间 [tBegin, tBegin+tNum)
        uint64_t tNum;
        uint32_t rot; // 对端轮转起点
        uint32_t peers;
        uint32_t readBlocks;    // 每个对端占一块 UB（own 直接落在 acc 槽里，不占读块）
        uint32_t readsPerEvent; // 每 readsPerEvent 笔读共用一个 MTE2_V 事件（基线 3，大 rank 数时放大）
        uint32_t groups;        // 读分组数，恒 <= AR_SYM_MAX_READ_EVENTS
    };

    // 对称内存「全核零同步」两段式：A 段按环序把 own + 各 peer 的同偏移片读进 UB 并向量累加，
    // B 段把归约结果直写本端与所有对端的 output 对称 VA。全程无 staging/flag/cclBuffer/原子加，
    // 每元素累加顺序固定 → 结果确定；远端流量与两跳算法同量级，跨 rank 完成由入口 BarrierAll 收口。
    __aicore__ inline void SymProcess()
    {
        SymPlan p;
        if (!MakeSymPlan(p)) {
            return; // 入口 AivAllReduceV2Mesh1DTwoShot 在 Process() 之后统一调 BarrierAll
        }
        LocalTensor<T> ubBuf = inOutQue.AllocTensor<T>();
        SymRunTiles(p, ubBuf);
        inOutQue.FreeTensor(ubBuf);
    }

    // 片划分 + UB 预算 + 核/tile 分配；false = 本核无 tile（入口统一调 BarrierAll）
    __aicore__ inline bool MakeSymPlan(SymPlan& p)
    {
        // 片划分以 32B（UB_ALIGN_SIZE）为单位：chunkStart 必须 32B 对齐，否则 SymDataCopy* 会走
        // DataCopyPad 路径，非对齐起点 + 多 tile 会错元素；不足 32B 的零头并给最后一个 rank。
        const uint64_t alignElem = UB_ALIGN_SIZE / sizeof(T);
        const uint64_t units = len_ / alignElem;
        const uint64_t baseUnits = units / rankSize_;
        const uint64_t extraUnits = units % rankSize_;
        p.chunkCount = (baseUnits + ((rank_ < extraUnits) ? 1U : 0U)) * alignElem;
        p.chunkStart = (rank_ * baseUnits + ((rank_ < extraUnits) ? rank_ : extraUnits)) * alignElem;
        if (rank_ + 1U == rankSize_) {
            p.chunkCount += len_ - units * alignElem;
        }
        if (p.chunkCount == 0U) {
            return false; // 本 rank 没有片（数据量小于 rankSize_ 个 32B 单位）；入口统一调 BarrierAll
        }
        // 对端 VA 数组只含「窗口基址 + 算子级 inputOffset」，不含 host 分循环偏移 → 自己补 loopOffset
        const uint64_t mySymIn = reinterpret_cast<uint64_t>(GetGmSymIn(rank_));
        p.loopOffsetIn = (input_ >= mySymIn) ? (input_ - mySymIn) : 0;
        const uint64_t mySymOut = reinterpret_cast<uint64_t>(GetGmSymOut(rank_));
        p.loopOffsetOut = (output_ >= mySymOut) ? (output_ - mySymOut) : 0;
        const uint32_t cores = static_cast<uint32_t>(numBlocks_);
        if (cores == 0) {
            return false;
        }
        p.peers = (rankSize_ > 1) ? static_cast<uint32_t>(rankSize_ - 1) : 0;
        p.readBlocks = p.peers;
        // 事件 ID 有限：每组读数量取 max(3, ceil(peers/4))，保证分组数不超过 AR_SYM_MAX_READ_EVENTS。
        p.readsPerEvent = (p.readBlocks + AR_SYM_MAX_READ_EVENTS - 1U) / AR_SYM_MAX_READ_EVENTS;
        p.readsPerEvent = ((p.readsPerEvent > AR_SYM_READS_PER_EVENT) ? p.readsPerEvent : AR_SYM_READS_PER_EVENT);
        p.groups = (p.readBlocks + p.readsPerEvent - 1U) / p.readsPerEvent;
        p.blkElems = CalcSymTileElems(p.chunkCount, p.readBlocks, cores);
        // 按 tile 切核（小数据量只放少量核参与，保证每核至少一整块）
        const uint64_t tiles = (p.chunkCount + p.blkElems - 1) / p.blkElems;
        const uint32_t usedCores = (tiles < static_cast<uint64_t>(cores)) ? static_cast<uint32_t>(tiles) : cores;
        if (blockIdx_ >= usedCores) {
            return false;
        }
        const uint64_t perCore = tiles / usedCores;
        const uint64_t rem = tiles % usedCores;
        p.tBegin = blockIdx_ * perCore + ((blockIdx_ < rem) ? blockIdx_ : rem);
        p.tNum = perCore + ((blockIdx_ < rem) ? 1 : 0);
        p.rot = static_cast<uint32_t>(blockIdx_ % rankSize_); // 对端轮转，避免所有核同时压同一条链路
        return true;
    }

    // 单 tile 元素数：UB 按「读块(readBlocks，own 直接落 acc 槽) + 写槽(AR_SYM_WRITE_SLOTS)」均分，
    // 再按 tile 目标粒度收敛；中等数据量时按「每核需搬的量」缩小，下限 AR_SYM_MIN_TILE_BYTES。
    static __aicore__ inline uint64_t CalcSymTileElems(uint64_t chunkCount, uint32_t readBlocks, uint32_t cores)
    {
        const uint64_t blocks = static_cast<uint64_t>(readBlocks) + AR_SYM_WRITE_SLOTS; // 对端读块 + acc 双槽
        uint64_t maxChunk = AR_SYM_UB_BYTES / blocks / UB_ALIGN_SIZE * UB_ALIGN_SIZE / sizeof(T);
        const uint64_t target = AR_SYM_TILE_TARGET_BYTES / sizeof(T);
        maxChunk = (target < maxChunk) ? target : maxChunk;
        const uint64_t perCoreNeed = (chunkCount + cores - 1) / cores;
        const uint64_t alignElems = UB_ALIGN_SIZE / sizeof(T);
        const uint64_t minTileElems = AR_SYM_MIN_TILE_BYTES / sizeof(T);
        if (perCoreNeed < maxChunk) {
            uint64_t q = perCoreNeed / alignElems * alignElems;
            q = (q < minTileElems) ? minTileElems : q;
            maxChunk = (q < maxChunk) ? q : maxChunk;
        }
        return maxChunk;
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

    // 全卡数共用的流水：acc 双槽（slot = (t - tBegin) & 1），tile t 的广播写在 tile t+1 的累加/预取期间
    // 继续在飞；预取（tile t+1 的读）在 tile t 的广播写之前下发，使读段与写段重叠。
    // 对端对称 VA 的广播写既不归 MTE3 事件统计，也不被 PIPE_MTE3 屏障排空，因此 acc 槽复用前必须用
    // PipeBarrier<PIPE_ALL> 收口，写段最多压 2 个 tile 在飞。
    __aicore__ inline void SymRunTiles(const SymPlan& p, const LocalTensor<T>& ubBuf)
    {
        if (p.tNum == 0U) {
            return;
        }
        const LocalTensor<T> accSlot0 = ubBuf[p.blkElems * p.readBlocks];
        const LocalTensor<T> accSlot1 = ubBuf[p.blkElems * (p.readBlocks + 1U)];
        const uint64_t tEnd = p.tBegin + p.tNum;
        SymPrefetchTile(p, ubBuf, accSlot0, p.tBegin);
        for (uint64_t t = p.tBegin; t < tEnd; ++t) {
            const bool oddSlot = (((t - p.tBegin) & 1U) != 0U);
            const LocalTensor<T> acc = oddSlot ? accSlot1 : accSlot0;
            const uint64_t tileOff = t * p.blkElems;
            const uint64_t cur = (p.chunkCount - tileOff > p.blkElems) ? p.blkElems : (p.chunkCount - tileOff);
            const uint64_t off = (p.chunkStart + tileOff) * sizeof(T);
            const uint32_t cnt = static_cast<uint32_t>(cur);
            SymReduceTile(p, ubBuf, acc, cnt); // 等本 tile 分组读齐 → 累加对端片
            AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0); // 累加落地才能直写
            if (t + 1U < tEnd) {
                // acc 槽复用前先等该槽上一轮广播写排空（对端对称 VA 的写只能靠全流水屏障收口）
                PipeBarrier<PIPE_ALL>();
                SymPrefetchTile(p, ubBuf, oddSlot ? accSlot0 : accSlot1, t + 1U);
            }
            SymBroadcastTile(p, acc, off, cnt);
        }
        PipeBarrier<PIPE_ALL>(); // 收尾：最后一 tile 的广播写（含对端）排空
    }

    // 下发一个 tile 的读：own 直接进 acc 槽，各对端进 readBlocks 的其余块（环序轮转，避免压同一条链路）
    __aicore__ inline void
    SymPrefetchTile(const SymPlan& p, const LocalTensor<T>& ubBuf, const LocalTensor<T>& acc, uint64_t t)
    {
        const uint64_t tileOff = t * p.blkElems;
        const uint64_t cur = (p.chunkCount - tileOff > p.blkElems) ? p.blkElems : (p.chunkCount - tileOff);
        const uint64_t off = (p.chunkStart + tileOff) * sizeof(T);
        const uint32_t cnt = static_cast<uint32_t>(cur);
        CopySymIn(acc, input_ + off, cnt);
        uint32_t r = 0;
        uint32_t g = 0;
        for (uint32_t k = 0; k < rankSize_ && r < p.peers; ++k) {
            const uint32_t peer = (p.rot + k) % rankSize_;
            if (peer == rank_) {
                continue;
            }
            ++r;
            CopySymPeerIn(ubBuf[p.blkElems * (r - 1U)], peer, p.loopOffsetIn + off, cnt);
            if (((r % p.readsPerEvent) == 0U) || (r == p.peers)) {
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + g);
                ++g;
            }
        }
    }

    // 等分组读齐后逐块累加到 acc（own 已由预取直接落在 acc 里）
    __aicore__ inline void
    SymReduceTile(const SymPlan& p, const LocalTensor<T>& ubBuf, const LocalTensor<T>& acc, uint32_t cnt)
    {
        for (uint32_t g = 0; g < p.groups; ++g) {
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0 + g);
            const uint32_t rBeg = p.readsPerEvent * g;
            uint32_t rEnd = rBeg + p.readsPerEvent;
            rEnd = (rEnd < p.readBlocks) ? rEnd : p.readBlocks;
            for (uint32_t r = rBeg; r < rEnd; ++r) {
                SymReduceLocal(acc, ubBuf[p.blkElems * r], cnt);
            }
        }
    }

    // 归约结果直写所有对端 output（对称 VA）+ 本端 output。写留在 MTE3 管道，由 acc 槽轮转处的
    // PipeBarrier<PIPE_ALL> 收口（对端对称 VA 的写只能靠全流水屏障排空）。
    __aicore__ inline void SymBroadcastTile(const SymPlan& p, const LocalTensor<T>& acc, uint64_t off, uint32_t cnt)
    {
        for (uint32_t k = 0; k < rankSize_; ++k) {
            const uint32_t peer = (p.rot + k) % rankSize_;
            const uint64_t dst = (peer == rank_) ?
                                     (output_ + off) :
                                     (reinterpret_cast<uint64_t>(GetGmSymOut(peer)) + p.loopOffsetOut + off);
            CopySymOut(dst, acc, cnt);
        }
    }

    __aicore__ inline void ReduceScatter()
    {
        if (coreIdx < producerNum) {
            ReduceScatterPreCopy();
        } else if (coreIdx < producerNum + consumerNum) {
            ReduceScatterGetRemote();
            ReduceScatterLocalReduce();
        } else {
            return;
        }
    }

    __aicore__ inline void ReduceScatterPreCopy()
    {
        uint64_t processCount;
        uint64_t sliceIdx;
        uint64_t srcOffset;
        uint64_t dstOffset;
        for (uint64_t idx = 0; idx < groupSize; ++idx) {
            sliceIdx = groupIdx * groupSize + idx;
            if (sliceIdx < bigSliceNum) {
                processCount = bigSliceCount;
                srcOffset = sliceIdx * bigSliceSize;
            } else {
                processCount = smallSliceCount;
                srcOffset = bigSliceNum * bigSliceSize + (sliceIdx - bigSliceNum) * smallSliceSize;
            }
            dstOffset = storeBuffOffset + sliceIdx * chunkSize;

            if (processCount > 0) {
                uint64_t src = input_ + srcOffset;
                uint64_t dst = reinterpret_cast<uint64_t>(myGmIn_) + dstOffset;
                CpGM2GM((__gm__ T*)dst, (__gm__ T*)src, processCount);
                pipe_barrier(PIPE_ALL);
            }

            // 通知数据从input搬运到output完成
            Record(rank_, preCopyFlagOffset + groupIdx * groupSize + idx, curTag_);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ReduceScatterGetRemote()
    {
        // 核按照远端rank分组，每组核负责读取一个远端rank的数据
        uint64_t rmtRank = groupIdx;
        // 数据源的索引和偏移
        uint64_t srcSliceIdx = rank_ * groupSize + idxInGroup;
        uint64_t processCount;
        if (srcSliceIdx < bigSliceNum) {
            processCount = bigSliceCount;
        } else {
            processCount = smallSliceCount;
        }
        uint64_t srcOffset = storeBuffOffset + srcSliceIdx * chunkSize;
        // 数据目的索引和偏移
        uint64_t dstSliceIdx = rmtRank * groupSize + idxInGroup;
        uint64_t dstOffset = reduceBuffOffset + dstSliceIdx * chunkSize;

        WaitFlag(rmtRank, preCopyFlagOffset + srcSliceIdx, curTag_);

        if (processCount > 0) {
            uint64_t src = reinterpret_cast<uint64_t>(GetGmIn(rmtRank)) + srcOffset;
            uint64_t dst = reinterpret_cast<uint64_t>(myGmIn_) + dstOffset;
            CpGM2GM((__gm__ T*)dst, (__gm__ T*)src, processCount);
            pipe_barrier(PIPE_ALL);
        }

        Record(rank_, getRemoteFlagOffset + dstSliceIdx, curTag_);
    }

    __aicore__ inline void ReduceScatterLocalReduce()
    {
        // 确定性计算，只需要ratio个核进行LocalReduce
        // 负责从本端rank搬运至本端rank的一组核，理论效率比读取远端快，用这组核来做Reduce可以减少性能抖动
        if (groupIdx != rank_) {
            return;
        }

        // 计算要处理的数据片的大小
        uint64_t dstSliceIdx = rank_ * groupSize + idxInGroup;
        uint64_t processCount;
        if (dstSliceIdx < bigSliceNum) {
            processCount = bigSliceCount;
        } else {
            processCount = smallSliceCount;
        }

        // 树形折叠（与 Aicpu 保序 RunLocalReduce 同序）：累加器 = rank0 的 slice 槽（Aicpu 基准），slot(dst) +=
        // slot(src) 零存储 flag 等待：源槽仅当 src >= k 时等待（[1,k) 已在作为目的槽被折入时等待过）；
        // 目的槽仅首次被折入时等待（新目的槽只出现在第 0/1 轮）
        uint64_t baseOffset = reduceBuffOffset + idxInGroup * chunkSize;
        uint64_t base = reinterpret_cast<uint64_t>(myGmIn_) + baseOffset;
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
                    WaitFlag(rank_, getRemoteFlagOffset + src * groupSize + idxInGroup, curTag_);
                }
                if (round == 0 || (round == 1 && dst >= firstRoundDstNum)) {
                    WaitFlag(rank_, getRemoteFlagOffset + dst * groupSize + idxInGroup, curTag_);
                }
                if (processCount > 0) {
                    uint64_t srcOffset = reduceBuffOffset + (src * groupSize + idxInGroup) * chunkSize;
                    uint64_t src = reinterpret_cast<uint64_t>(myGmIn_) + srcOffset;
                    uint64_t dstOffset = (dst == 0) ? base :
                                                      (reinterpret_cast<uint64_t>(myGmIn_) + reduceBuffOffset
                                                       + (dst * groupSize + idxInGroup) * chunkSize);
                    CpGM2GM((__gm__ T*)dstOffset, (__gm__ T*)src, processCount, reduceOp_);
                    pipe_barrier(PIPE_ALL);
                }
            }
            curBlocks = M;
            ++round;
        }
        // 结果留在 rank0 槽（累加器），AllGather 直接读 rank0 槽，无需额外拷贝
        // 单核全部Reduce完成再Record
        Record(rank_, localReduceFlagOffset + idxInGroup, curTag_);
    }

    __aicore__ inline void AllGather()
    {
        if (coreIdx >= producerNum && coreIdx < producerNum + consumerNum) {
            AllGatherGetRemote();
        } else {
            return;
        }
    }

    __aicore__ inline void AllGatherGetRemote()
    {
        uint64_t rmtRank = groupIdx;

        // 远端rank是最后一个rank时，本端每组最后一个核搬运尾块
        uint64_t sliceIdx = rmtRank * groupSize + idxInGroup;
        uint64_t processCount;
        uint64_t dstOffset;
        if (sliceIdx < bigSliceNum) {
            processCount = bigSliceCount;
            dstOffset = sliceIdx * bigSliceSize;
        } else {
            processCount = smallSliceCount;
            dstOffset = bigSliceNum * bigSliceSize + (sliceIdx - bigSliceNum) * smallSliceSize;
        }
        // 归约结果落在各卡的 rank0 槽（reduceBuffOffset + idxInGroup*chunkSize），直接从该槽读取，
        // 消除归约段内的结果拷回本端槽
        uint64_t srcOffset = reduceBuffOffset + idxInGroup * chunkSize;

        WaitFlag(rmtRank, localReduceFlagOffset + idxInGroup, curTag_);

        if (processCount > 0) {
            uint64_t src = reinterpret_cast<uint64_t>(GetGmIn(rmtRank)) + srcOffset;
            uint64_t dst = output_ + dstOffset;
            CpGM2GM((__gm__ T*)dst, (__gm__ T*)src, processCount);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline uint64_t RoundUPAiv(uint64_t dividend, uint64_t divisor)
    {
        return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
    }

    __aicore__ inline void SmallCoreReduceScatter(uint32_t sliceId)
    {
        uint64_t dataCount = len_;
        rankChunkStride = RoundUPAiv(dataCount, rankSize_);
        ipc_reduce_flag_offset = rankSize_;
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        // 镜像区/槽区分界。只依赖 len_，各卡一致，对端可直接算出
        slotBase_ = ((len_ * sizeof(T) + UB_ALIGN_SIZE - 1) / UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        // step1: 本卡 input -> 本卡镜像区，然后写对端 flag。索引 rank_ 落在 targetRank 卡上，
        // 对给定 (源rank, 目标片) 只有唯一一个核，握手已精确到片。
        // SK空闲核跳过，避免与忙核重复写数据和flag。
        if (!IsIdleCore()) {
            for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
                targetRank = blockIdx_ + i * numBlocks_;
                rankChunkSize
                    = ((targetRank + 1) * rankChunkStride <= dataCount) ?
                          rankChunkStride :
                          (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));

                uint64_t chunkOffset = (targetRank * rankChunkStride) * sizeof(T);
                if (rankChunkSize > 0) {
                    uint64_t outputOffset = reinterpret_cast<uint64_t>(myGmIn_) + chunkOffset;
                    CpGM2GM((__gm__ T*)outputOffset, (__gm__ T*)(input_ + chunkOffset), rankChunkSize);
                    pipe_barrier(PIPE_ALL);
                }
                // set flag
                Record(targetRank, rank_, curTag_);
            }
            pipe_barrier(PIPE_ALL);
        }
        // step2: 对端镜像区 -> 本卡槽区。核 c 负责 peer r = c, c+numBlocks_, ...，
        // 每核 ceil(rankSize_/numBlocks_) 次跨卡读，
        // 各核写入的槽互不重叠，可完全并行。槽地址与改动前一致，折叠代码原样复用。
        if (!IsIdleCore()) {
            uint64_t myChunkSize
                = ((rank_ + 1) * rankChunkStride <= dataCount) ?
                      rankChunkStride :
                      (dataCount <= rank_ * rankChunkStride ? 0 : (dataCount - rank_ * rankChunkStride));
            uint64_t peerChunkOffset = (rank_ * rankChunkStride) * sizeof(T);
            for (uint32_t r = blockIdx_; r < rankSize_; r += static_cast<uint32_t>(numBlocks_)) {
                WaitFlag(rank_, r, curTag_);
                if (myChunkSize > 0) {
                    uint64_t srcOffset = reinterpret_cast<uint64_t>(GetGmIn(r)) + peerChunkOffset;
                    uint64_t dstOffset
                        = reinterpret_cast<uint64_t>(myGmIn_) + slotBase_ + (r * myChunkSize) * sizeof(T);
                    CpGM2GM((__gm__ T*)dstOffset, (__gm__ T*)srcOffset, myChunkSize);
                    pipe_barrier(PIPE_ALL);
                }
            }
        }
        // 卡内栅栏：槽区由多核分头填充，折叠核必须等全部就绪
        SyncAll<true>();
        pipe_barrier(PIPE_ALL);
        // step3: 树形折叠(与 Aicpu 保序 RunLocalReduce 同序)，仅槽基址平移到 slotBase_
        if (blockIdx_ == numBlocks_ - 1) {
            uint64_t myRankChuckSize
                = ((rank_ + 1) * rankChunkStride <= dataCount) ?
                      rankChunkStride :
                      (dataCount <= rank_ * rankChunkStride ? 0 : (dataCount - rank_ * rankChunkStride));
            ;
            if (myRankChuckSize > 0) {
                // 树形折叠（与 Aicpu 保序 RunLocalReduce 同序）：累加器 = rank0 槽（初值已是 rank0，
                // 由 step2 跨卡读回填充），slot(dst) += slot(src) 零存储 flag 等待：源槽仅当 src >= k 时
                // 等待（[1,k) 已在作为目的槽被折入时等待过）； 目的槽仅首次被折入时等待（新目的槽只在第 0/1 轮）
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
                            WaitFlag(rank_, src, curTag_);
                        }
                        if (round == 0 || (round == 1 && dst >= firstRoundDstNum)) {
                            WaitFlag(rank_, dst, curTag_);
                        }
                        // reduce
                        uint64_t inputOffset
                            = reinterpret_cast<uint64_t>(myGmIn_) + slotBase_ + (src * myRankChuckSize) * sizeof(T);
                        uint64_t outputOffset
                            = reinterpret_cast<uint64_t>(myGmIn_) + slotBase_ + (dst * myRankChuckSize) * sizeof(T);
                        CpGM2GM((__gm__ T*)outputOffset, (__gm__ T*)inputOffset, myRankChuckSize, reduceOp_);
                        pipe_barrier(PIPE_ALL);
                    }
                    curBlocks = M;
                    ++round;
                }
            }
            Record(rank_, ipc_reduce_flag_offset + 1, curTag_);
        }
    }

    __aicore__ inline void SmallCoreAllgather()
    {
        uint64_t dataCount = len_;
        // SK空闲核跳过gather，避免与忙核重复搬运
        if (IsIdleCore()) {
            return;
        }
        for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
            targetRank = blockIdx_ + i * numBlocks_;
            rankChunkSize
                = ((targetRank + 1) * rankChunkStride <= dataCount) ?
                      rankChunkStride :
                      (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));

            if (rankChunkSize <= 0) {
                return;
            }

            // waitflag
            WaitFlag(targetRank, ipc_reduce_flag_offset + 1, curTag_);

            // gather
            uint64_t inputOffset = reinterpret_cast<uint64_t>(GetGmIn(targetRank)) + slotBase_;
            uint64_t outputOffset = output_ + (targetRank * rankChunkStride) * sizeof(T);
            CpGM2GM((__gm__ T*)outputOffset, (__gm__ T*)inputOffset, rankChunkSize);
            pipe_barrier(PIPE_ALL);
        }
    }

    // 基础信息
    uint64_t coreIdx{0};
    uint64_t coreNum{0};
    uint64_t count{0};
    // 分组信息
    uint64_t producerNum{0};
    uint64_t groupSize{0};
    uint64_t consumerNum{0};
    uint64_t groupIdx{0};
    uint64_t idxInGroup{0};
    // 数据切分信息
    uint64_t sliceNum{0};
    uint64_t bigSliceNum{0};
    uint64_t bigSliceCount{0};
    uint64_t bigSliceSize{0};
    uint64_t smallSliceCount{0};
    uint64_t smallSliceSize{0};
    uint64_t chunkSize{0};
    // HcclBuffer偏移信息
    uint64_t storeBuffOffset{0};
    uint64_t reduceBuffOffset{0};
    // flag偏移信息
    uint64_t preCopyFlagOffset{0};
    uint64_t getRemoteFlagOffset{0};
    uint64_t localReduceFlagOffset{0};
    // 少核场景
    uint32_t targetRank;
    uint64_t rankChunkSize;
    uint64_t rankChunkStride;
    uint64_t ipc_reduce_flag_offset{1024};
    uint64_t slotBase_{0}; // 镜像区/槽区分界(= align32(len_*sizeof(T)))，各卡一致
};

template <typename T>
__aicore__ inline void AivAllReduceV2Mesh1DTwoShot(KERNEL_ARGS_DEF)
{
    AivAllReduceMesh1DTwoShot<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    // 对称路径对核数无「>= 2*rankSize」要求，须优先走 Prepare/Process：是否进 SymProcess 由 UseSymPath 决定，
    // 否则 Process() 落回两 shot 基线，此时核数 < 2*rankSize 数据搬不动，必须走 SmallCore 分支。
    const bool symPath = op.UseSymPath();
    // sliceId 是模板对象内每次 launch 自增的序号，每个算子从 1 开始，故 1 即本算子首轮：
    // 首轮已确认全 rank 输入就绪，后续轮次读同一份输入的其他区段，由上一轮出口 BarrierAll 收口即可。
    const bool needEntrySync = symPath && (sliceId == 1);
    if (needEntrySync) {
        op.BarrierAll(0); // 直读对端 input 前确认各 rank 本轮输入已提交
        SyncAll<true>();  // BarrierAll 只确认本核负责的 rank 子集，须补核间同步
    }
    if (symPath || op.numBlocks_ >= rankSize * 2) {
        op.Prepare(sliceId);
        op.Process();
    } else {
        op.SmallCoreReduceScatter(sliceId);
        op.SmallCoreAllgather();
    }
    op.BarrierAll(needEntrySync ? 1U : 0U);
}

template <typename T>
__aicore__ inline void AivAllReduceV2Mesh1DTwoShotSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    // 参数中的input和output是分别加了步长inBuffBaseOff和outBuffBaseOff步长
    AivAllReduceMesh1DTwoShot<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT);

    uint64_t maxCountPerLoop;
    if (op.numBlocks_ >= op.rankSize_ * 2) {
        uint64_t consumerNum = (op.numBlocks_ - op.rankSize_) / op.rankSize_ * op.rankSize_;
        uint64_t baseCount = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE / 2 / sizeof(T);
        maxCountPerLoop = (baseCount > consumerNum) ? (baseCount - consumerNum) : 0;
    } else {
        // 少核路径 scratch = 镜像区 + 槽区，两段各约 curCount*sizeof(T)，需 curCount <= base/2/sizeof(T)。
        // 原式在 rankSize_>=2 时更紧(已覆盖)，取 min 后对 rankSize_>=2 行为不变，只堵住 rankSize_==1。
        uint64_t base = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
        uint64_t bySlotLayout = base / op.rankSize_ / sizeof(T);
        uint64_t byPullRegions = base / DOUBLE / sizeof(T);
        maxCountPerLoop = (bySlotLayout < byPullRegions) ? bySlotLayout : byPullRegions;
    }
    uint64_t countLeft = op.len_;

    int32_t loopTag = op.tag_;
    uint32_t barrierSeq = 0;

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);
        op.len_ = curCount;
        if (op.UseSymPath() || op.numBlocks_ >= op.rankSize_ * 2) {
            op.Prepare(loopTag);
            op.Process();
        } else {
            op.SmallCoreReduceScatter(loopTag);
            op.SmallCoreAllgather();
        }
        op.BarrierAll(barrierSeq);
        ++barrierSeq;

        countLeft -= curCount;
        op.input_ += curSize;
        op.output_ += curSize;
        loopTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
}

__aicore__ inline void sk_ar_mesh_1d_twoshot(SUPERKERNEL_ARGS_DEF)
{
#ifdef HCCL_DTYPE_INT8
    AivAllReduceV2Mesh1DTwoShotSuperKernel<int8_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT16
    AivAllReduceV2Mesh1DTwoShotSuperKernel<int16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT32
    AivAllReduceV2Mesh1DTwoShotSuperKernel<int32_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP16
    AivAllReduceV2Mesh1DTwoShotSuperKernel<half>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP32
    AivAllReduceV2Mesh1DTwoShotSuperKernel<float>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_BFP16
    AivAllReduceV2Mesh1DTwoShotSuperKernel<bfloat16_t>(SUPERKERNEL_ARGS_CALL);
#else
#endif
}

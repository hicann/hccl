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

// 对称内存路径的 UB 缓冲：inOutQue 共 194560B，均分 AG_SYM_BUF_NUM 块、单块 48640B，
// 即 4 笔 DMA 同时在飞的流水深度。
constexpr uint32_t AG_SYM_BUF_NUM = 4;
// 扁平档有效核数上限
constexpr uint32_t AG_SYM_CORE_CAP = 16U;
constexpr uint32_t AG_SYM_TILE_BYTES = 48640U;
// 分层直读的组大小：4 卡一组，组内 UB 直连，组间只有 UB_SW4。
constexpr uint32_t AG_SYM_GROUP_SIZE = 4U;
// 启用分层直读的最小 rankSize，且要求是 4 的倍数（组间只有 UB_SW4 一条瓶颈链路）
constexpr uint32_t AG_SYM_HIER_RANK_SIZE = 16U;

template <typename T>
class AivAllGatherMesh1D : public AivCommBase {
public:
    __aicore__ inline AivAllGatherMesh1D() {}

    // 如果让rankSize个核去写本地，写的时候切成多份去写
    // 然后剩下的核，切成rankSize份，去读
    // 需要知道写本地和读远端的时间比例，才可以调节怎么切分
    __aicore__ inline void InitCoreInfo(uint64_t len) // len is input length
    {
        // 每个核处理 len/curStageCoreNum 个数据
        uint64_t targetRank = blockIdx_;
        uint64_t dataPerRank = len / rankSize_;
        uint64_t remainderPerRank = len % rankSize_;
        // 数据对不齐的情况
        uint64_t innerDisplsPerRank = 0;
        uint64_t sendCurCountPerRank = 0;
        if (targetRank < remainderPerRank) { // 这部分核需要多处理一个数据
            innerDisplsPerRank = targetRank * dataPerRank + targetRank;
            sendCurCountPerRank = dataPerRank + 1;
        } else {
            innerDisplsPerRank = targetRank * dataPerRank + remainderPerRank;
            sendCurCountPerRank = dataPerRank;
        }

        uint64_t dataPerCore = sendCurCountPerRank / cutNum;
        uint64_t remainder = sendCurCountPerRank % cutNum;
        // 这里是要切三分，还是直接一份，然后多个flag？
        for (int64_t coreIndex = 0; coreIndex < cutNum; coreIndex++) {
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
                uint64_t usrInOffset = input_ + (innerDisplsPerRank + innerDispls) * sizeof(T);
                uint64_t cclInOffset
                    = reinterpret_cast<uint64_t>(myGmIn_) + (innerDisplsPerRank + innerDispls) * sizeof(T);
                CpGM2GM((__gm__ T*)cclInOffset, (__gm__ T*)usrInOffset, sendCurCount);
                // 每个核写flag
                Record(rank_, blockIdx_ * cutNum + coreIndex, curTag_);
            }
        }
    }

    __aicore__ inline void Run(uint64_t len, uint64_t stride)
    {
        uint64_t blockInedx = blockIdx_ - coreNumStage1;
        // 然后stage2的核分成rankSize份，每一份有cutNum个核，每个核去读对端的rankSize份数据
        uint32_t coreNumPerRank = coreNumStage2 / rankSize_;
        uint32_t targetRank = blockInedx / coreNumPerRank;
        uint32_t coreIndex = (blockInedx - (targetRank * coreNumPerRank)) % coreNumPerRank;
        for (uint32_t idx = 0; idx < rankSize_; idx++) {
            // 这里根据写数据时候的编排，去计算每次读数据时候的编排
            uint64_t innerIndex
                = (idx + rank_) % rankSize_; // 做一点偏移，不要出现好多张卡读同一块数据的情况，却似有一点点效果
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
                // 先去wait flag，然后把数据拷贝过来
                WaitFlag(targetRank, innerIndex * coreNumPerRank + coreIndex, curTag_);

                uint64_t srcCclInOffset
                    = reinterpret_cast<uint64_t>(GetGmIn(targetRank)) + (innerDisplsPerRank + innerDispls) * sizeof(T);
                uint64_t usrOutOffset = output_ + targetRank * stride + (innerDisplsPerRank + innerDispls) * sizeof(T);
                CpGM2GM((__gm__ T*)usrOutOffset, (__gm__ T*)srcCclInOffset, sendCurCount);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // 握手 flag 索引编码：[0, rankSize) = 「该 rank 本轮 input 已就绪」，
    // [rankSize, 2*rankSize) = 「该 rank 本轮 input 已被读完」，两段互不重叠。
    __aicore__ inline uint32_t FlagIdxInputReady() const { return rank_; }
    __aicore__ inline uint32_t FlagIdxInputConsumed() const { return rank_ + static_cast<uint32_t>(rankSize_); }

    // 分层直读 flag 索引：与握手段错开，编码「(写者 rank, 外组序号)」。
    // idx = 2*rankSize + writerRank*(groupNum-1) + gi
    __aicore__ inline uint32_t HierFlagIdx(uint32_t writerRank, uint32_t gi, uint32_t groupNum) const
    {
        return 2U * static_cast<uint32_t>(rankSize_) + writerRank * (groupNum - 1U) + gi;
    }

    // 对称直读（扁平档）的有效核数：取 rankSize 的 2^k 倍且不超过 AG_SYM_CORE_CAP。
    // 每核负责整数个整片时吞吐最高，一个片被多个核拆开会明显下降；rankSize 大于上限时按 rankSize 取。
    __aicore__ inline uint32_t SymCoreCount() const
    {
        const uint32_t rs = static_cast<uint32_t>(rankSize_);
        uint32_t cap = (rs == 0U) ? 1U : rs;
        while ((cap * 2U) <= AG_SYM_CORE_CAP) {
            cap *= 2U;
        }
        const uint32_t nb = static_cast<uint32_t>(numBlocks_);
        return (nb > cap) ? cap : nb;
    }

    // 对称直读：把各 rank 的 input 搬到本端 output 的对应槽位。
    //   小档（pingpong，无出口 BarrierAll）跨 rank 只靠握手，按 rank 分段、一核可负责多片；
    //   大档无 flag 依赖，由入口/出口 BarrierAll 收口，按 rankSize*count 扁平均分。两者都不限制核数。
    __aicore__ inline void ProcessSymPull(uint64_t count, uint64_t stride)
    {
        if (count == 0) {
            return;
        }
        if (count * sizeof(T) > DATA_LIMIT) {
            PullFlat(count, stride);
        } else {
            PullSlicesByCore(count, stride);
        }
    }

    // 大档直读无跨 rank flag，读对端 input 前须确认对端已进本轮 kernel（小档已有 ready/consumed 握手）。
    __aicore__ inline bool NeedSymEntrySync(uint64_t count) const
    {
        return (symInEnable_ == 1) && (symOutEnable_ == 1) && (count * sizeof(T) > DATA_LIMIT);
    }

    // 小档：按 rank 分片搬，每核负责一段 rank（分段方式与 BarrierAll 的 rank 分段一致）：
    //   核数 >= rankSize_ 时一核一片（多余核空闲）；核数 < rankSize_ 时一核多片。
    // 该档没有出口 BarrierAll，跨 rank 只靠握手，一核多片时必须避免依赖成环：把握手拆成两段，
    // 先把「本 rank 本轮 input 已就绪」广播给负责的所有 rank，再统一等它们的 ready；读完同理。
    __aicore__ inline void PullSlicesByCore(uint64_t count, uint64_t stride)
    {
        const uint32_t nb = (numBlocks_ > 0) ? static_cast<uint32_t>(numBlocks_) : 1U;
        const uint32_t perCoreRankNum = static_cast<uint32_t>(rankSize_) / nb;
        const uint32_t remainRankNum = static_cast<uint32_t>(rankSize_) % nb;
        const uint32_t curCoreRankNum = (blockIdx_ < remainRankNum) ? perCoreRankNum + 1U : perCoreRankNum;
        const uint32_t startRank = (blockIdx_ < remainRankNum) ? (perCoreRankNum + 1U) * blockIdx_ :
                                                                 perCoreRankNum * blockIdx_ + remainRankNum;
        if (curCoreRankNum == 0U) {
            return;
        }
        const uint32_t myRank = static_cast<uint32_t>(rank_);
        // 1) 广播「本 rank 本轮 input 已就绪」
        for (uint32_t r = startRank; r < startRank + curCoreRankNum; ++r) {
            if (r != myRank) {
                Record(r, FlagIdxInputReady(), curTag_);
            }
        }
        // 2) 等这些 rank 的 input 就绪
        for (uint32_t r = startRank; r < startRank + curCoreRankNum; ++r) {
            if (r != myRank) {
                WaitFlag(rank_, r, curTag_);
            }
        }
        // 3) 逐片直读（本端片走本地拷贝）
        for (uint32_t r = startRank; r < startRank + curCoreRankNum; ++r) {
            const uint64_t srcBase
                = (r == myRank) ? input_ : reinterpret_cast<uint64_t>(GetGmSymIn(r)) + symLoopInOffset_;
            PullRange(srcBase, output_ + static_cast<uint64_t>(r) * stride, count);
        }
        // 4) 广播「对端 input 已读完」
        for (uint32_t r = startRank; r < startRank + curCoreRankNum; ++r) {
            if (r != myRank) {
                Record(r, FlagIdxInputConsumed(), curTag_);
            }
        }
        // 5) 等对端读完我的 input（对端才能复用其 user input）
        for (uint32_t r = startRank; r < startRank + curCoreRankNum; ++r) {
            if (r != myRank) {
                WaitFlag(rank_, r + static_cast<uint32_t>(rankSize_), curTag_);
            }
        }
    }

    // 大档：把 rankSize_ 个片看作一条连续空间，有效核分区间、每核负责 [begin, end)（可能跨片），
    // 任意 numBlocks_（1 个核、非 rankSize 整数倍、远大于 rankSize）都能均分。
    // 切分以 32B（UB_ALIGN_SIZE）为单位，使各核起点保持对齐、走 DataCopy 快路径。
    __aicore__ inline void PullFlat(uint64_t count, uint64_t stride)
    {
        const uint32_t nb = SymCoreCount();
        if ((nb == 0U) || (static_cast<uint32_t>(blockIdx_) >= nb)) {
            return;
        }
        const uint64_t total = static_cast<uint64_t>(rankSize_) * count;
        const uint64_t alignElem = UB_ALIGN_SIZE / sizeof(T);
        const uint64_t totalUnits = total / alignElem;
        const uint64_t begin = static_cast<uint64_t>(blockIdx_) * totalUnits / nb * alignElem;
        uint64_t end = (static_cast<uint64_t>(blockIdx_) + 1U) * totalUnits / nb * alignElem;
        if (static_cast<uint32_t>(blockIdx_) + 1U == nb) {
            end = total; // 不足一个 32B 的零头并给最后一个核
        }
        for (uint32_t r = static_cast<uint32_t>(begin / count); (r < rankSize_) && (r * count < end); ++r) {
            const uint64_t sliceStart = static_cast<uint64_t>(r) * count;
            const uint64_t sliceEnd = sliceStart + count;
            const uint64_t from = ((begin > sliceStart) ? begin : sliceStart) - sliceStart; // 片内偏移
            const uint64_t to = ((end < sliceEnd) ? end : sliceEnd) - sliceStart;
            if (to <= from) {
                continue;
            }
            const uint64_t srcBase
                = (r == rank_) ? input_ : reinterpret_cast<uint64_t>(GetGmSymIn(r)) + symLoopInOffset_;
            PullRange(
                srcBase + from * sizeof(T), output_ + static_cast<uint64_t>(r) * stride + from * sizeof(T), to - from);
        }
    }

    __aicore__ inline uint32_t SymTileElems() const { return static_cast<uint32_t>(AG_SYM_TILE_BYTES / sizeof(T)); }

    // 发起一笔读：从 srcAddr 起第 off 个元素开始、最多 tileElem 个，落 ubBuf[slot]，完成后置 MTE2->MTE3 事件；
    // 返回本笔实际搬运的元素数。
    __aicore__ inline uint64_t
    IssueReadTile(const LocalTensor<T>& ubBuf, uint64_t srcAddr, uint64_t off, uint64_t left, uint32_t slot)
    {
        const uint32_t tileElem = SymTileElems();
        const uint64_t n = (left > tileElem) ? tileElem : left;
        const uint64_t addr = srcAddr + off * sizeof(T);
        GlobalTensor<T> gSrc;
        gSrc.SetGlobalBuffer((__gm__ T*)addr, static_cast<uint32_t>(n));
        DataCopyGM2UBByAddr(ubBuf[tileElem * slot], gSrc, static_cast<uint32_t>(n), addr);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + slot);
        return n;
    }

    // 发起一笔写：把 ubBuf[slot] 写到 dstAddr 起第 off 个元素，并置 MTE3->MTE2 事件供该缓冲复用。
    // 返回本笔实际搬运的元素数。
    __aicore__ inline uint64_t
    IssueWriteTile(const LocalTensor<T>& ubBuf, uint64_t dstAddr, uint64_t off, uint64_t left, uint32_t slot)
    {
        const uint32_t tileElem = SymTileElems();
        const uint64_t n = (left > tileElem) ? tileElem : left;
        const uint64_t addr = dstAddr + off * sizeof(T);
        GlobalTensor<T> gDst;
        gDst.SetGlobalBuffer((__gm__ T*)addr, static_cast<uint32_t>(n));
        DataCopyUB2GMByAddr(gDst, ubBuf[tileElem * slot], static_cast<uint32_t>(n), addr);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4 + slot);
        return n;
    }

    // 单核把 [srcAddr, srcAddr + count*sizeof(T)) 拉到 dstAddr，AG_SYM_BUF_NUM 块深流水
    __aicore__ inline void PullRange(uint64_t srcAddr, uint64_t dstAddr, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        LocalTensor<T> ubBuf = inOutQue.AllocTensor<T>();
        uint64_t pDone = 0;
        for (uint32_t issued = 0; (issued < AG_SYM_BUF_NUM) && (pDone < count); ++issued) {
            pDone += IssueReadTile(ubBuf, srcAddr, pDone, count - pDone, issued);
        }
        uint64_t cDone = 0;
        uint32_t usedMask = 0;
        for (uint32_t idx = 0; cDone < count; ++idx) {
            const uint32_t slot = idx % AG_SYM_BUF_NUM;
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0 + slot);
            cDone += IssueWriteTile(ubBuf, dstAddr, cDone, count - cDone, slot);
            usedMask |= (1U << slot);
            if (pDone < count) { // 回填：该缓冲的本地写完成后才能复用
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4 + slot);
                usedMask &= ~(1U << slot);
                pDone += IssueReadTile(ubBuf, srcAddr, pDone, count - pDone, slot);
            }
        }
        for (uint32_t slot = 0; slot < AG_SYM_BUF_NUM; ++slot) {
            if ((usedMask & (1U << slot)) != 0U) {
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4 + slot);
            }
        }
        inOutQue.FreeTensor(ubBuf);
    }

    // 分层直读（rankSize = 4G，G >= 4）：rank 分成 G 个 4 卡组（组内 UB 直连，组间只有 UB_SW4）。
    // 扁平读时外组流量全部压在 UB_SW4 上、组内直连链路基本闲置；这里外组数据「组内一卡一份」拿回来，
    // 再用组内直连链路互相补齐：
    //   核 [0,4)              : 组内 4 个 rank 的整片（含自己，走本地拷贝）
    //   核 [4, 4+G-1)         : (G-1) 个外组里「与本 rank 组内同序号」的那片（switch），落盘后给组内 3 卡打 flag
    //   核 [4+G-1, 4G)        : 3(G-1) 个（组内对端 x 外组）补齐片（直连，读对端 output 对称 VA）
    // 组内序号为 j 的卡负责外组第 j 片，补齐时 slot = fg*4 + pj，正好等于对端 rank 号。
    __aicore__ inline void ProcessSymPullHier(uint64_t count, uint64_t stride)
    {
        if (count == 0) {
            return;
        }
        const uint32_t groupNum = static_cast<uint32_t>(rankSize_) / AG_SYM_GROUP_SIZE;
        if (groupNum < 2U) {
            ProcessSymPull(count, stride); // 不足两组时分层无收益，退回扁平直读
            return;
        }
        // 一槽一核：槽位总数 = 4G = rankSize，核数多于 rankSize 的核空闲（由 BarrierAll 收口）
        const uint32_t core = static_cast<uint32_t>(blockIdx_);
        const uint32_t directEnd = AG_SYM_GROUP_SIZE + (groupNum - 1U); // 组内整片 + 外组各一片
        const uint32_t relayEnd = directEnd + (groupNum - 1U) * (AG_SYM_GROUP_SIZE - 1U); // 再加组内补齐片
        if (core < AG_SYM_GROUP_SIZE) {
            PullHierLocalGroup(core, count, stride);
        } else if (core < directEnd) {
            PullHierOuterGroup(core - AG_SYM_GROUP_SIZE, groupNum, count, stride);
        } else if (core < relayEnd) {
            PullHierRelay(core - directEnd, groupNum, count, stride);
        }
    }

    // 组内整片：槽位 0..3 分别搬组内 4 个 rank 的 input（含自己，走本地拷贝）
    __aicore__ inline void PullHierLocalGroup(uint32_t slot, uint64_t count, uint64_t stride)
    {
        const uint32_t myGroup = static_cast<uint32_t>(rank_ / AG_SYM_GROUP_SIZE);
        const uint32_t myJ = static_cast<uint32_t>(rank_ % AG_SYM_GROUP_SIZE);
        const uint32_t peer = myGroup * AG_SYM_GROUP_SIZE + (myJ + slot) % AG_SYM_GROUP_SIZE;
        const uint64_t src = (peer == static_cast<uint32_t>(rank_)) ?
                                 input_ :
                                 reinterpret_cast<uint64_t>(GetGmSymIn(peer)) + symLoopInOffset_;
        PullRange(src, output_ + static_cast<uint64_t>(peer) * stride, count);
    }

    // 外组一片：每个外组只由「组内同序号」的卡拉回来（switch 流量 1/4），落盘后给组内另外 3 卡打 flag
    __aicore__ inline void PullHierOuterGroup(uint32_t gi, uint32_t groupNum, uint64_t count, uint64_t stride)
    {
        const uint32_t myGroup = static_cast<uint32_t>(rank_ / AG_SYM_GROUP_SIZE);
        const uint32_t myJ = static_cast<uint32_t>(rank_ % AG_SYM_GROUP_SIZE);
        const uint32_t fg = (myGroup + 1U + gi) % groupNum; // 轮转外组，避免所有组同时压同一个外组
        const uint32_t peer = fg * AG_SYM_GROUP_SIZE + myJ;
        PullRange(
            reinterpret_cast<uint64_t>(GetGmSymIn(peer)) + symLoopInOffset_,
            output_ + static_cast<uint64_t>(peer) * stride, count);
        for (uint32_t pj = 1U; pj < AG_SYM_GROUP_SIZE; ++pj) {
            const uint32_t dst = myGroup * AG_SYM_GROUP_SIZE + (myJ + pj) % AG_SYM_GROUP_SIZE;
            Record(dst, HierFlagIdx(rank_, gi, groupNum), curTag_);
        }
    }

    // 组内补齐：从组内对端 peer 的 output 对称 VA 取它已落盘的外组片
    __aicore__ inline void PullHierRelay(uint32_t t, uint32_t groupNum, uint64_t count, uint64_t stride)
    {
        const uint32_t myGroup = static_cast<uint32_t>(rank_ / AG_SYM_GROUP_SIZE);
        const uint32_t myJ = static_cast<uint32_t>(rank_ % AG_SYM_GROUP_SIZE);
        const uint32_t pjOff = t / (groupNum - 1U); // 组内第几个对端
        const uint32_t gi = t % (groupNum - 1U);    // 哪个外组
        const uint32_t pj = (myJ + 1U + pjOff) % AG_SYM_GROUP_SIZE;
        const uint32_t peer = myGroup * AG_SYM_GROUP_SIZE + pj;
        const uint32_t fg = (myGroup + 1U + gi) % groupNum;
        const uint32_t slot = fg * AG_SYM_GROUP_SIZE + pj;
        WaitFlag(rank_, HierFlagIdx(peer, gi, groupNum), curTag_); // 等对端 switch 核把该外组片写入其 output
        PullRange(
            reinterpret_cast<uint64_t>(GetGmSymOut(peer)) + symLoopOutOffset_ + static_cast<uint64_t>(slot) * stride,
            output_ + static_cast<uint64_t>(slot) * stride, count);
    }

    __aicore__ inline void Process(uint64_t count, uint64_t sliceId, uint64_t stride)
    {
        uint32_t rankMultiple = 2;
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        // 两侧对称窗口都可用时直接读对端 user input（免 staging、免 flag 中转），大档/小档均不限制核数。
        const bool smallData = (count * sizeof(T) <= DATA_LIMIT);
        if (symInEnable_ == 1 && symOutEnable_ == 1) {
            // rankSize >= 16 且为组大小整数倍、一槽一核（核数 >= rankSize）才分层，否则退扁平直读
            const uint32_t groupNum = static_cast<uint32_t>(rankSize_) / AG_SYM_GROUP_SIZE;
            const bool hier = (rankSize_ >= AG_SYM_HIER_RANK_SIZE) && !smallData && (groupNum >= 2U)
                              && ((static_cast<uint32_t>(rankSize_) % AG_SYM_GROUP_SIZE) == 0U)
                              && (numBlocks_ >= rankSize_);
            const uint64_t mySymIn = reinterpret_cast<uint64_t>(GetGmSymIn(rank_));
            symLoopInOffset_ = (input_ >= mySymIn) ? (input_ - mySymIn) : 0;
            const uint64_t mySymOut = reinterpret_cast<uint64_t>(GetGmSymOut(rank_));
            symLoopOutOffset_ = (output_ >= mySymOut) ? (output_ - mySymOut) : 0;
            if (hier) {
                ProcessSymPullHier(count, stride);
            } else {
                ProcessSymPull(count, stride);
            }
            return;
        }
        if (count * sizeof(T) >= DATA_LIMIT && numBlocks_ >= rankMultiple * rankSize_) {
            // 核数大于等于2倍ranksize
            curStageCoreNum = numBlocks_ / rankSize_ * rankSize_; // 总的核数
            coreNumStage1 = rankSize_;
            coreNumStage2 = curStageCoreNum - coreNumStage1;
            cutNum = coreNumStage2 / rankSize_;

            if (blockIdx_ < coreNumStage1) {
                InitCoreInfo(count);
            } else if (blockIdx_ < curStageCoreNum) {
                Run(count, stride);
            }
        } else {
            // 核数小于ranksize
            RunCtrlCore(count, stride);
        }
    }

    __aicore__ inline void RunCtrlCore(uint64_t count, uint64_t stride)
    {
        int32_t curNumBlocks = numBlocks_;
        // 核数小于ranksize
        if (curNumBlocks > rankSize_) {
            curNumBlocks = rankSize_;
        }
        // SK满核启动时，超出curNumBlocks的核不搬数据，只执行相同次数的SyncAll
        bool coreIdle = blockIdx_ >= static_cast<uint32_t>(curNumBlocks);
        // 分核把数据从input搬到gm
        if (!coreIdle) {
            auto input = reinterpret_cast<__gm__ T*>(input_);
            uint64_t dataTypeSize = sizeof(T);
            uint64_t countPerCore = count / curNumBlocks;
            uint64_t curCountCore
                = blockIdx_ == curNumBlocks - 1 ? count - countPerCore * (curNumBlocks - 1) : countPerCore;
            auto gmIn = reinterpret_cast<__gm__ T*>(
                reinterpret_cast<uint64_t>(myGmIn_) + blockIdx_ * countPerCore * dataTypeSize);
            CpGM2GM(gmIn, input + blockIdx_ * countPerCore, curCountCore);
        }
        SyncAll<true>();

        // 每个核分配多个rank搬运数据从gm到对端output
        if (!coreIdle) {
            uint32_t perCoreRankNum = rankSize_ / curNumBlocks;
            uint32_t remainRankNum = rankSize_ % curNumBlocks;
            uint32_t curCoreRankNum = blockIdx_ < remainRankNum ? perCoreRankNum + 1 : perCoreRankNum;
            uint32_t startRank = blockIdx_ < remainRankNum ? (perCoreRankNum + 1) * blockIdx_ :
                                                             perCoreRankNum * blockIdx_ + remainRankNum;
            for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
                Record(rank, rank_, curTag_);
            }
            for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
                auto gmOthers = reinterpret_cast<__gm__ T*>(reinterpret_cast<uint64_t>(GetGmIn(rank)));
                auto output = reinterpret_cast<__gm__ T*>(output_ + rank * stride);
                WaitFlag(rank_, rank, curTag_);
                CpGM2GM(output, gmOthers, count);
                PipeBarrier<PIPE_ALL>();
                Record(rank, rank_ + rankSize_, curTag_);
            }
            for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
                WaitFlag(rank_, rank + rankSize_, curTag_);
            }
        }
    }

    uint64_t coreOffset;
    uint64_t curCount;
    uint64_t coreNumStage1;
    uint64_t coreNumStage2;
    uint64_t cutNum;
    uint64_t curStageCoreNum;
    // 对称路径：input_/output_ 相对本 rank 对称窗口基址的偏移（host 分循环时需自己补）
    uint64_t symLoopInOffset_ = 0;
    uint64_t symLoopOutOffset_ = 0;
};

template <typename T>
__aicore__ inline void AivAllGatherV2Mesh1D(KERNEL_ARGS_DEF)
{
    AivAllGatherMesh1D<T> op;
    bool pingpong = false;
    if (len * sizeof(T) <= DATA_LIMIT && rankSize * len * sizeof(T) <= PINGPONG_TOTAL_DATA_LIMIT) {
        pingpong = true;
    }
    op.Init(KERNEL_CLASS_INIT, true, pingpong);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    const bool symEntrySync = op.NeedSymEntrySync(len);
    // sliceId 是本算子内的轮次序号，1 即首轮：首轮已确认全 rank 输入就绪，
    // 后续轮次读同一份输入的其他区段，由上一轮出口 BarrierAll 收口即可。
    const bool needEntrySync = symEntrySync && (sliceId == 1);
    if (needEntrySync) {
        op.BarrierAll(0); // 直读对端 input 前确认各 rank 本轮输入已提交
        SyncAll<true>();  // BarrierAll 只确认本核负责的 rank 子集，须补核间同步
    }
    op.Process(len, sliceId, outputSliceStride);
    if (!pingpong) {
        op.BarrierAll(needEntrySync ? 1U : 0U);
    }
}

template <typename T>
__aicore__ inline void AivAllGatherV2Mesh1DSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    AivAllGatherMesh1D<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT, false);
    uint64_t maxCountPerLoop = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE / op.rankSize_ / sizeof(T);
    uint64_t countLeft = op.len_;

    int32_t loopTag = op.tag_;
    uint32_t barrierSeq = 0;

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        op.Process(curCount, loopTag, op.outputSliceStride_);
        op.BarrierAll(barrierSeq);
        ++barrierSeq;
        countLeft -= curCount;
        op.input_ += curSize;
        op.output_ += curSize;
        loopTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
}

__aicore__ inline void sk_ag_mesh_1d(SUPERKERNEL_ARGS_DEF)
{
#ifdef HCCL_DTYPE_INT8
    AivAllGatherV2Mesh1DSuperKernel<int8_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_UINT8
    AivAllGatherV2Mesh1DSuperKernel<uint8_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT16
    AivAllGatherV2Mesh1DSuperKernel<int16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_UINT16
    AivAllGatherV2Mesh1DSuperKernel<uint16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT32
    AivAllGatherV2Mesh1DSuperKernel<int32_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_UINT32
    AivAllGatherV2Mesh1DSuperKernel<uint32_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP16
    AivAllGatherV2Mesh1DSuperKernel<half>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP32
    AivAllGatherV2Mesh1DSuperKernel<float>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_BFP16
    AivAllGatherV2Mesh1DSuperKernel<bfloat16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT64
    AivAllGatherV2Mesh1DSuperKernel<int64_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_UINT64
    AivAllGatherV2Mesh1DSuperKernel<uint64_t>(SUPERKERNEL_ARGS_CALL);
#else
#endif
}

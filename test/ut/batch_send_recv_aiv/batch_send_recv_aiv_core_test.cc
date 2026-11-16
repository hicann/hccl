/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "aiv_batch_send_recv_mesh_1D.h"

// AIV BatchSendRecv kernel 的 CPU 桩级覆盖：单线程执行生产 kernel（aiv_batch_send_recv_mesh_1D.h），
// GM 用按页稀疏的伪内存建模，因此既能断言"flag 落在哪个地址、值是多少"，也能断言"数据真的搬到了哪"。
//   1) tag 只由 host 下发的轮次值决定：既不读也不写 65MB 块里那个全通信域共享的计数器（旧方案 GetTag +
//      RestoreTagCounter 的分界线），且值域与其它 AIV 算子的 [1,1000] 不相交；
//   2) flag 半区由 tag 奇偶决定，收发两侧的 (rank, slot, tag) 必须完全对上，否则在没有超时的 AIV 上会永久空转；
//   3) 数据路径：发送方写进自己的 CCL bucket，接收方从对端的同一个偏移取，最终落进用户 buffer；
//   4) Producer/Consumer 按 workIdx 分核：每条任务恰好被一个核处理一次；itemNum 超限被钳制在 512；
//   5) 自收发走用户地址、屏障只覆盖本轮活跃对端（去重、跳过本 rank、无对端不屏障）；
//   6) 消费侧读完立即把同一格清 0，避免下一轮同槽位假到达。
// liveness 不建模：跨 rank 收发互相等待，单线程按 rank 串行执行无法满足，故 WaitFlag 只读一次并记录命中与否。
// 因此跨 rank 的断言只针对"数据 flag"（sender 先跑完、receiver 再跑，命中可验证），屏障只做结构断言。

using AivBatchSendRecv::AIV_BSR_RECV;
using AivBatchSendRecv::AIV_BSR_SELF_RECV;
using AivBatchSendRecv::AIV_BSR_SELF_SEND;
using AivBatchSendRecv::AIV_BSR_SEND;
using AivBatchSendRecv::AivBsrItem;
using AivBatchSendRecv::BATCH_SEND_RECV_ITEM_MAX;
using AivBatchSendRecv::ExtraArgsBsr;
using bsr_aiv_test::Event;
using bsr_aiv_test::EventKind;

namespace {

// 起点：host 侧取 AIV_BSR_TAG_BASE(1024) + 1，即第一轮的 tag（见 aiv_temp_batch_send_recv_mesh_1D.h 的
// AIV_BSR_TAG_BASE / AIV_BSR_TAG_MAX / AIV_BSR_TAG_SPAN）。这两个常量是从 host 侧抄来的：那边改值时这里要跟，
// 否则本文件的"值域隔离"断言会失去意义。
constexpr uint32_t kFirstTag = 1025;
constexpr uint32_t kTagMax = 0x7FFF;
// 数据 flag 的槽位号上界：rankSize<=256 时最大 slot = rank*flagBlock+sliceIdx < 256（见 host 侧 flagBlock 计算），
// 屏障区槽位号 >= 32768，两者用 512 区分不会误判。
constexpr uint64_t kDataSlotLimit = 512;

uint64_t TagAddr(uint32_t rank) { return bsr_aiv_test::BlockBase(rank) + AIV_FLAG_CLEAR_OFFSET; }

// 数据 flag 的绝对地址：块内半区（由 tag 奇偶选）+ slot*FLAG_SIZE。
uint64_t DataFlagAddr(uint32_t ownerRank, uint32_t roundTag, uint64_t slot)
{
    const uint64_t half = (roundTag % 2 == 0) ? FLAG1_OFFSET : FLAG2_OFFSET;
    return bsr_aiv_test::BlockBase(ownerRank) + half + slot * FLAG_SIZE;
}

// 屏障到达槽：不管落在哪个半区，绝对地址恒在块内 BASE_FLAG_OFFSET 处（BASE - 半区 再乘回去正好抵消）。
uint64_t BarrierFlagAddr(uint32_t ownerRank, uint32_t writerRank)
{
    return bsr_aiv_test::BlockBase(ownerRank) + BASE_FLAG_OFFSET + static_cast<uint64_t>(writerRank) * FLAG_SIZE;
}

// 判断某个绝对地址是否落在任何 rank 的屏障区（块内偏移 >= 9M）。
bool IsBarrierAddr(uint64_t addr)
{
    const uint64_t offsetInBlock = (addr - bsr_aiv_test::BLOCK_BASE) % bsr_aiv_test::BLOCK_STRIDE;
    return offsetInBlock >= BASE_FLAG_OFFSET;
}

bool IsBarrierEvent(const Event& event) { return IsBarrierAddr(event.dst); }

// 数据 flag 都在半区内（1M/5M，slot <= 256），屏障到达槽在 9M：用块内偏移区分，与槽位号无关
// （槽位号本身还随半区变化，用它做判据会误分类）。
bool IsDataEvent(const Event& event) { return !IsBarrierAddr(event.dst); }

// 抢答/发布事件（tag != 0）与"消费即清"事件（tag == 0）要分开数。
bool IsPublish(const Event& event) { return IsDataEvent(event) && event.tag != 0; }

template <typename Pred>
std::vector<Event> SelectIf(EventKind kind, Pred pred)
{
    std::vector<Event> out;
    for (const Event& event : bsr_aiv_test::Events()) {
        if (event.kind == kind && pred(event)) {
            out.push_back(event);
        }
    }
    return out;
}

std::vector<Event> Select(EventKind kind)
{
    return SelectIf(kind, [](const Event&) {
        return true;
    });
}

// 铺两张表（buffersIn 在块首、buffersOut 在 +16K），表项按"对端 rank"索引，故每个 rank 的块里都要写全
// 所有 rank 的块地址 —— 这正是 kernel 里 GetGmIn/GetGmOut 查表的方式。
// 另外给共享 tag 计数器塞一个"别的算子的计数"，且 rank 间故意错开（seed + rank）：旧方案在这种通信域里
// 会 tag 错位，新方案应完全不看它。
void SetupWorld(uint32_t rankSize, uint32_t counterSeed = 0)
{
    bsr_aiv_test::ResetTrace();
    for (uint32_t owner = 0; owner < rankSize; owner++) {
        for (uint32_t rank = 0; rank < rankSize; rank++) {
            const uint64_t block = bsr_aiv_test::BlockBase(owner);
            bsr_aiv_test::GmWrite64(block + rank * sizeof(uint64_t), bsr_aiv_test::BlockBase(rank));
            bsr_aiv_test::GmWrite64(
                block + BUFFER_OUT_ADDR_OFFSET + rank * sizeof(uint64_t), bsr_aiv_test::BlockBase(rank));
        }
        const uint32_t counter = counterSeed + owner;
        std::vector<uint8_t> bytes(
            reinterpret_cast<const uint8_t*>(&counter), reinterpret_cast<const uint8_t*>(&counter) + sizeof(counter));
        bsr_aiv_test::SeatBytes(bsr_aiv_test::BlockBase(owner) + AIV_FLAG_CLEAR_OFFSET, bytes);
    }
    // 铺表本身不算被测代码的访问，清掉计数以便断言"kernel 没读写计数器"。
    bsr_aiv_test::store().writes.clear();
    bsr_aiv_test::store().reads.clear();
}

struct KernelArgs {
    uint32_t rank = 0;
    uint32_t rankSize = 2;
    uint32_t numBlocks = 2;
    uint32_t core = 0;
    uint32_t roundTag = kFirstTag;
};

void RunKernel(const KernelArgs& args, ExtraArgsBsr& extra)
{
    bsr_aiv_test::store().currentBlock = args.core;
    bsr_aiv_test::store().currentRank = args.rank;
    AivBatchSendRecvV2Mesh1D(
        reinterpret_cast<GM_ADDR>(static_cast<uintptr_t>(bsr_aiv_test::BlockBase(args.rank))),
        /*input*/ 0, /*output*/ 0, args.rank, /*sendRecvRemoteRank*/ 0, args.rankSize, /*len*/ 0, /*dataType*/ 0,
        /*reduceOp*/ 0, /*root*/ 0, /*sliceId = roundTag*/ args.roundTag, /*inputSliceStride*/ 0,
        /*outputSliceStride*/ 0, /*repeatNum*/ 0, /*inputRepeatStride*/ 0, /*outputRepeatStride*/ 0, args.numBlocks,
        /*isOpBase*/ false, /*headCountMem*/ nullptr, /*tailCountMem*/ nullptr, /*addOneMem*/ nullptr,
        /*counterMemSize*/ 0, /*isEnableCounter*/ false, /*isWrite*/ false, extra);
}

// 一个 kernel 的所有核依次跑一遍（核号在构造期取，故每次调用都要重建对象）。
void RunAllCores(const KernelArgs& args, ExtraArgsBsr& extra)
{
    for (uint32_t core = 0; core < args.numBlocks; core++) {
        KernelArgs perCore = args;
        perCore.core = core;
        RunKernel(perCore, extra);
    }
}

AivBsrItem& NextItem(ExtraArgsBsr& args)
{
    AivBsrItem& item = args.items[args.itemNum];
    args.itemNum++;
    item = AivBsrItem{};
    return item;
}

void AddSend(
    ExtraArgsBsr& args, uint32_t remoteRank, uint64_t bufAddr, uint32_t bytes, uint16_t sliceIdx, uint32_t cclOffset)
{
    AivBsrItem& item = NextItem(args);
    item.bufAddr = bufAddr;
    item.byteCount = bytes;
    item.sliceIdx = sliceIdx;
    item.sendRecvType = static_cast<uint8_t>(AIV_BSR_SEND);
    item.peer.remoteRank = static_cast<uint16_t>(remoteRank);
    item.peer.cclOffset = cclOffset;
}

void AddRecv(
    ExtraArgsBsr& args, uint32_t remoteRank, uint64_t bufAddr, uint32_t bytes, uint16_t sliceIdx, uint32_t cclOffset)
{
    AivBsrItem& item = NextItem(args);
    item.bufAddr = bufAddr;
    item.byteCount = bytes;
    item.sliceIdx = sliceIdx;
    item.sendRecvType = static_cast<uint8_t>(AIV_BSR_RECV);
    item.peer.remoteRank = static_cast<uint16_t>(remoteRank);
    item.peer.cclOffset = cclOffset;
}

// 自收发：dst 直接是用户目的地址，不经 cclBuffer。
void AddSelfSend(ExtraArgsBsr& args, uint64_t dstAddr, uint64_t srcAddr, uint32_t bytes, uint16_t sliceIdx)
{
    AivBsrItem& item = NextItem(args);
    item.bufAddr = srcAddr;
    item.byteCount = bytes;
    item.sliceIdx = sliceIdx;
    item.sendRecvType = static_cast<uint8_t>(AIV_BSR_SELF_SEND);
    item.dstAddr = dstAddr;
}

void AddSelfRecv(ExtraArgsBsr& args, uint64_t dstAddr, uint32_t bytes, uint16_t sliceIdx)
{
    AivBsrItem& item = NextItem(args);
    item.bufAddr = dstAddr;
    item.byteCount = bytes;
    item.sliceIdx = sliceIdx;
    item.sendRecvType = static_cast<uint8_t>(AIV_BSR_SELF_RECV);
}

std::vector<uint8_t> Pattern(uint64_t bytes, uint8_t seed = 1)
{
    std::vector<uint8_t> data(static_cast<size_t>(bytes));
    for (uint64_t i = 0; i < bytes; i++) {
        data[static_cast<size_t>(i)] = static_cast<uint8_t>(seed + i * 7);
    }
    return data;
}

} // namespace

// tag 直接等于 host 下发的轮次值左移 16 位，与通信域里那个共享计数器无关：计数器既不读也不写。
// 落哪个半区也只看 tag 奇偶（偶 → FLAG1/1M，奇 → FLAG2/5M），相邻轮次必然换区。
TEST(BsrAivKernel, TagComesFromRoundTagAndPicksHalfRegionByParity)
{
    const std::vector<uint32_t> roundTags = {kFirstTag, 1026, 1027, 1028, 32766, 32767};
    const std::vector<uint32_t> seeds = {0, 1, 500, 1000, 12345};
    for (uint32_t roundTag : roundTags) {
        for (uint32_t seed : seeds) {
            SCOPED_TRACE(::testing::Message() << "roundTag=" << roundTag << " seed=" << seed);
            SetupWorld(2, seed);
            ExtraArgsBsr extra{};
            extra.flagBlock = 4;
            AddSend(extra, 1, bsr_aiv_test::UserBuf(0, 0x100), 32, 3, 64);

            KernelArgs args;
            args.rank = 0;
            args.roundTag = roundTag;
            RunAllCores(args, extra);

            const std::vector<Event> records = SelectIf(EventKind::RECORD, IsPublish);
            ASSERT_EQ(records.size(), 1u);
            // flag 值 = (tag<<16) | sliceIdx，tag 完全由 roundTag 决定
            EXPECT_EQ(records[0].tag, static_cast<int32_t>((roundTag << 16) | 3u));
            // 槽位 = remoteRank*flagBlock + sliceIdx = 1*4+3，半区按奇偶选
            EXPECT_EQ(records[0].dst, DataFlagAddr(0, roundTag, 1 * 4 + 3));
            const uint64_t offset = records[0].dst - bsr_aiv_test::BlockBase(0);
            EXPECT_EQ(offset, ((roundTag % 2 == 0) ? FLAG1_OFFSET : FLAG2_OFFSET) + (1 * 4 + 3) * FLAG_SIZE);
            // 数据 flag 整体落在半区内，不会溢出到 9M 的屏障区
            EXPECT_TRUE(offset + kDataSlotLimit * FLAG_SIZE < BASE_FLAG_OFFSET);
            // 65MB 块里 512K 处那个计数器：一次都没读、一次都没写
            EXPECT_EQ(bsr_aiv_test::ReadCountAt(TagAddr(0)), 0u);
            EXPECT_EQ(bsr_aiv_test::WriteCountAt(TagAddr(0)), 0u);
            EXPECT_EQ(bsr_aiv_test::ReadCountAt(TagAddr(1)), 0u);
            EXPECT_EQ(bsr_aiv_test::WriteCountAt(TagAddr(1)), 0u);
            // 不走 GetTag 也就不会去申请 inOutQue
            EXPECT_EQ(bsr_aiv_test::store().ubAllocs, 0u);
        }
    }
}

// 值域隔离：其它 AIV 算子的 tag 恒在 [1,1000]（GetTag 到 1000 回绕），本算子必须整段落在它上面，
// 且 tag<<16 拼上 sliceIdx 后仍是正数（int32 的 flag 值不能进符号位）。
TEST(BsrAivKernel, TagDomainStaysAboveOtherAivOps)
{
    const uint32_t otherMax = static_cast<uint32_t>(TAG_RESET_COUNT);
    const uint32_t bsrMin = kFirstTag;
    const uint32_t bsrMax = kTagMax;
    EXPECT_LT(otherMax, bsrMin);
    EXPECT_LT(bsrMin, bsrMax);

    // 任何其它算子的 flag 值都不会等于任何 BSR 的 flag 值（低 16 位的 sliceIdx 取满也一样）
    for (uint32_t other = 1; other <= otherMax; other++) {
        const int64_t otherValue = static_cast<int64_t>(other) << 16;
        for (uint32_t tag : {bsrMin, 1026u, 32766u, bsrMax}) {
            const int64_t bsrValue = static_cast<int64_t>(tag) << 16;
            EXPECT_NE(otherValue, bsrValue);
            EXPECT_TRUE(bsrValue > 0);
        }
    }

    // sliceIdx 取满 0xFFFF 时，与上界 tag 拼出来的 flag 值 = 0x7FFFFFFF，仍是正数（不左移进符号位）。
    // 这是纯算术断言：kernel 里就是 baseTag | (sliceIdx & LOW_16_BITS)，与这里的表达式逐字等价。
    EXPECT_EQ(static_cast<int32_t>((bsrMax << 16) | 0xFFFFu), static_cast<int32_t>(0x7FFFFFFF));
    EXPECT_TRUE(static_cast<int32_t>((bsrMax << 16) | 0xFFFFu) > 0);

    // 上界一侧的实测：tag=32767 时 flag 仍落在正区间，且半区按奇数取 FLAG2
    SetupWorld(2);
    ExtraArgsBsr extra{};
    extra.flagBlock = 4;
    AddSend(extra, 1, bsr_aiv_test::UserBuf(0, 0x100), 32, 0, 64);
    KernelArgs args;
    args.rank = 0;
    args.roundTag = bsrMax;
    RunAllCores(args, extra);
    const std::vector<Event> records = SelectIf(EventKind::RECORD, IsPublish);
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].tag, static_cast<int32_t>(bsrMax << 16));
    EXPECT_TRUE(records[0].tag > 0);
    EXPECT_EQ(records[0].dst, DataFlagAddr(0, bsrMax, 1 * 4));
}

// 收发两侧：即使两个 rank 的共享计数器故意不同（0 与 1），同一轮算出的 flag 地址与期望值也必须完全一致。
TEST(BsrAivKernel, SendAndRecvSidesAgreeOnFlagAddressAndTag)
{
    SetupWorld(2, /*counterSeed*/ 0); // rank0 计数器=0，rank1 计数器=1
    ExtraArgsBsr sender{};
    sender.flagBlock = 4;
    AddSend(sender, 1, bsr_aiv_test::UserBuf(0, 0x100), 32, 0, 0x80);

    ExtraArgsBsr receiver{};
    receiver.flagBlock = 4;
    AddRecv(receiver, 0, bsr_aiv_test::UserBuf(1, 0x100), 32, 0, 0x80);

    KernelArgs args;
    args.roundTag = 1026;
    args.rank = 0;
    RunAllCores(args, sender);
    args.rank = 1;
    RunAllCores(args, receiver);

    const std::vector<Event> records = SelectIf(EventKind::RECORD, IsPublish);
    const std::vector<Event> waits = SelectIf(EventKind::WAIT, IsDataEvent);
    ASSERT_EQ(records.size(), 1u);
    ASSERT_EQ(waits.size(), 1u);
    EXPECT_EQ(records[0].dst, waits[0].dst);
    EXPECT_EQ(records[0].tag, waits[0].tag);
    EXPECT_EQ(records[0].slot, waits[0].slot);
    EXPECT_TRUE(waits[0].matched);
    EXPECT_EQ(waits[0].target, 0u);
    // 取完立刻清同一格：消费侧会再写一次 0
    const std::vector<Event> clears = SelectIf(EventKind::RECORD, [](const Event& event) {
        return IsDataEvent(event) && event.tag == 0;
    });
    ASSERT_EQ(clears.size(), 1u);
    EXPECT_EQ(clears[0].dst, records[0].dst);
    // 两侧落在同一个半区（1026 为偶数 → FLAG1），槽位 = remoteRank*flagBlock + sliceIdx = 1*4+0
    EXPECT_EQ(records[0].dst, DataFlagAddr(0, 1026, 1 * 4));
}

// 数据路径 + 消费即清：发送方写进自己的 CCL bucket，接收方按同一偏移从对端 bucket 取，最终内容与源一致；
// 每条被等过的 flag 读完后必须回到 0，否则下一轮同槽位会（在 tag 相同的情况下）假到达。
TEST(BsrAivKernel, DataFlowsThroughSenderBucketAndClearsEveryConsumedFlag)
{
    constexpr uint32_t kItems = 4;
    std::vector<std::vector<uint8_t>> payloads;
    for (uint32_t i = 0; i < kItems; i++) {
        payloads.push_back(Pattern(256, static_cast<uint8_t>(3 + i)));
        bsr_aiv_test::SeatBytes(bsr_aiv_test::UserBuf(0, 0x2000 + i * 0x400), payloads[i]);
    }

    SetupWorld(2, 7);
    ExtraArgsBsr sender{};
    sender.flagBlock = 8;
    ExtraArgsBsr receiver{};
    receiver.flagBlock = 8;
    for (uint16_t i = 0; i < kItems; i++) {
        AddSend(sender, 1, bsr_aiv_test::UserBuf(0, 0x2000 + i * 0x400), 256, i, 0x400 + i * 0x400);
        AddRecv(receiver, 0, bsr_aiv_test::UserBuf(1, 0x8000 + i * 0x400), 256, i, 0x400 + i * 0x400);
    }

    KernelArgs args;
    args.roundTag = kFirstTag;
    args.rank = 0;
    RunAllCores(args, sender);
    args.rank = 1;
    RunAllCores(args, receiver);

    // 发方向 kItems 次、收方向 kItems 次，源/目的逐条对齐
    const std::vector<Event> copies = Select(EventKind::COPY);
    ASSERT_EQ(copies.size(), 2u * kItems);
    for (uint32_t i = 0; i < kItems; i++) {
        EXPECT_EQ(copies[i].src, bsr_aiv_test::UserBuf(0, 0x2000 + i * 0x400));
        EXPECT_EQ(copies[i].dst, bsr_aiv_test::BlockBase(0) + 0x400 + i * 0x400);
        EXPECT_EQ(copies[i].bytes, 256u);
        EXPECT_EQ(copies[kItems + i].src, bsr_aiv_test::BlockBase(0) + 0x400 + i * 0x400);
        EXPECT_EQ(copies[kItems + i].dst, bsr_aiv_test::UserBuf(1, 0x8000 + i * 0x400));
        EXPECT_EQ(bsr_aiv_test::ReadBytes(bsr_aiv_test::UserBuf(1, 0x8000 + i * 0x400), 256), payloads[i]);
    }

    // 4 条 flag 都被等到，且逐条清回 0（只清第一条的实现会在这里露出来）
    const std::vector<Event> waits = SelectIf(EventKind::WAIT, IsDataEvent);
    ASSERT_EQ(waits.size(), kItems);
    for (const Event& wait : waits) {
        EXPECT_TRUE(wait.matched);
        EXPECT_EQ(bsr_aiv_test::GmRead32(wait.dst), 0);
    }
}

// 分核：按 workIdx 轮转，每条任务恰好被一个核处理一次（不能漏、不能重）；核数为奇数时多出来的核空转。
TEST(BsrAivKernel, ProducerAndConsumerOwnEveryItemExactlyOnce)
{
    constexpr uint32_t kItems = 6;
    SetupWorld(2);

    ExtraArgsBsr sender{};
    sender.flagBlock = 8;
    std::vector<uint64_t> expectedSources;
    for (uint16_t i = 0; i < kItems; i++) {
        AddSend(sender, 1, bsr_aiv_test::UserBuf(0, 0x1000 + i * 64), 64, i, 0x100 + i * 64);
        expectedSources.push_back(bsr_aiv_test::UserBuf(0, 0x1000 + i * 64));
    }
    KernelArgs args;
    args.numBlocks = 4; // 2 个生产核
    args.rank = 0;
    RunAllCores(args, sender);

    std::vector<Event> copies = Select(EventKind::COPY);
    ASSERT_EQ(copies.size(), kItems);
    std::vector<uint64_t> sources;
    for (const Event& copy : copies) {
        EXPECT_LT(copy.core, 2u); // 只有生产核在搬
        sources.push_back(copy.src);
    }
    std::sort(sources.begin(), sources.end());
    std::sort(expectedSources.begin(), expectedSources.end());
    EXPECT_EQ(sources, expectedSources); // 每条任务恰好搬一次，无重复无遗漏

    std::vector<Event> records = SelectIf(EventKind::RECORD, IsPublish);
    ASSERT_EQ(records.size(), kItems);
    std::vector<uint64_t> slots;
    for (const Event& record : records) {
        slots.push_back(record.slot);
    }
    std::sort(slots.begin(), slots.end());
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    EXPECT_EQ(slots.size(), kItems);

    // 收方向：同样 6 条，只有消费核在搬，源地址都是对端 bucket 里的对应偏移
    SetupWorld(2);
    ExtraArgsBsr receiver{};
    receiver.flagBlock = 8;
    std::vector<uint64_t> recvSources;
    for (uint16_t i = 0; i < kItems; i++) {
        AddRecv(receiver, 0, bsr_aiv_test::UserBuf(1, 0x4000 + i * 64), 64, i, 0x100 + i * 64);
        recvSources.push_back(bsr_aiv_test::BlockBase(0) + 0x100 + i * 64);
    }
    args.rank = 1;
    RunAllCores(args, receiver);
    copies = Select(EventKind::COPY);
    ASSERT_EQ(copies.size(), kItems);
    std::vector<uint64_t> actualSources;
    for (const Event& copy : copies) {
        EXPECT_GE(copy.core, 2u); // 只有消费核在搬
        actualSources.push_back(copy.src);
    }
    std::sort(actualSources.begin(), actualSources.end());
    std::sort(recvSources.begin(), recvSources.end());
    EXPECT_EQ(actualSources, recvSources);

    // 奇数核：halfCoreNum = 1，core0 生产、core1 消费、core2 只在收尾 Sync 里出现
    SetupWorld(2);
    ExtraArgsBsr odd{};
    odd.flagBlock = 4;
    AddSend(odd, 1, bsr_aiv_test::UserBuf(0, 0x100), 32, 0, 0x100);
    AddSend(odd, 1, bsr_aiv_test::UserBuf(0, 0x200), 32, 1, 0x200);
    args.numBlocks = 3;
    args.rank = 0;
    RunAllCores(args, odd);

    uint32_t idleSync = 0;
    for (const Event& event : bsr_aiv_test::Events()) {
        if (event.core != 2) {
            continue;
        }
        EXPECT_EQ(event.kind, EventKind::SYNC);
        idleSync++;
    }
    EXPECT_EQ(idleSync, 1u);
    EXPECT_EQ(Select(EventKind::COPY).size(), 2u);
}

// 任务表内容侧的两条防线：itemNum 越界声明钳到表容量；零字节与非法 sendRecvType 不参与任何方向。
TEST(BsrAivKernel, ItemNumIsClampedAndInvalidItemsAreIgnored)
{
    // 1) 多声明的部分不会被当成 item 去搬 GM
    SetupWorld(2);
    ExtraArgsBsr extra{};
    extra.flagBlock = BATCH_SEND_RECV_ITEM_MAX;
    for (uint16_t i = 0; i < BATCH_SEND_RECV_ITEM_MAX; i++) {
        AddSend(extra, 0, bsr_aiv_test::UserBuf(1, 0x1000 + i * 8), 8, i, 0x100 + i * 8);
    }
    extra.itemNum = 1000; // 故意越界：只有前 512 条合法

    KernelArgs args;
    args.rank = 1;
    args.roundTag = kFirstTag;
    RunAllCores(args, extra); // numBlocks=2 → 单个生产核

    EXPECT_EQ(Select(EventKind::COPY).size(), static_cast<size_t>(BATCH_SEND_RECV_ITEM_MAX));
    EXPECT_EQ(SelectIf(EventKind::RECORD, IsPublish).size(), static_cast<size_t>(BATCH_SEND_RECV_ITEM_MAX));

    // 2) 零字节表项与非法类型既不搬也不发 flag，也不进屏障集合
    SetupWorld(2);
    ExtraArgsBsr mixed{};
    mixed.flagBlock = 4;
    AddSend(mixed, 1, bsr_aiv_test::UserBuf(0, 0x100), 0, 0, 0x100); // 零字节

    AivBsrItem& bad = NextItem(mixed);
    bad.bufAddr = bsr_aiv_test::UserBuf(0, 0x200);
    bad.byteCount = 64;
    bad.sliceIdx = 1;
    bad.sendRecvType = 7; // 非法类型

    args.rank = 0;
    RunAllCores(args, mixed);

    EXPECT_EQ(Select(EventKind::COPY).size(), 0u);
    EXPECT_EQ(SelectIf(EventKind::RECORD, IsPublish).size(), 0u);
    EXPECT_EQ(SelectIf(EventKind::WAIT, IsDataEvent).size(), 0u);
    // 无跨 rank 活跃对端 ⇒ 不进出屏障区
    EXPECT_EQ(SelectIf(EventKind::RECORD, IsBarrierEvent).size(), 0u);
    EXPECT_EQ(SelectIf(EventKind::WAIT, IsBarrierEvent).size(), 0u);
}

// 自收发：数据直接写用户目的地址（不经 bucket），flag 用本 rank 区段；读方等同一格并清 0。
TEST(BsrAivKernel, SelfSendRecvUsesUserBufferAndOwnFlagSlot)
{
    const std::vector<uint8_t> payload = Pattern(64, 11);
    bsr_aiv_test::SeatBytes(bsr_aiv_test::UserBuf(0, 0x100), payload);

    SetupWorld(2);
    ExtraArgsBsr extra{};
    extra.flagBlock = 4;
    AddSelfSend(extra, bsr_aiv_test::UserBuf(0, 0x4000), bsr_aiv_test::UserBuf(0, 0x100), 64, 2);
    AddSelfRecv(extra, bsr_aiv_test::UserBuf(0, 0x4000), 64, 2);

    KernelArgs args;
    args.rank = 0;
    args.roundTag = kFirstTag;
    RunAllCores(args, extra);

    const std::vector<Event> copies = Select(EventKind::COPY);
    ASSERT_EQ(copies.size(), 1u);
    EXPECT_EQ(copies[0].src, bsr_aiv_test::UserBuf(0, 0x100));
    EXPECT_EQ(copies[0].dst, bsr_aiv_test::UserBuf(0, 0x4000));
    EXPECT_EQ(bsr_aiv_test::ReadBytes(bsr_aiv_test::UserBuf(0, 0x4000), 64), payload);

    const std::vector<Event> records = SelectIf(EventKind::RECORD, IsPublish);
    const std::vector<Event> waits = SelectIf(EventKind::WAIT, IsDataEvent);
    ASSERT_EQ(records.size(), 1u);
    ASSERT_EQ(waits.size(), 1u);
    // 槽位 = rank_*flagBlock + sliceIdx，只有自收发会写这一格
    EXPECT_EQ(records[0].slot, 0u * 4u + 2u);
    EXPECT_EQ(records[0].dst, DataFlagAddr(0, kFirstTag, 2));
    EXPECT_EQ(records[0].dst, waits[0].dst);
    EXPECT_TRUE(waits[0].matched);
    // 读方消费后清 0
    EXPECT_EQ(bsr_aiv_test::GmRead32(records[0].dst), 0);
}

// 屏障只覆盖本轮真正交互的对端：去重、跳过本 rank、排除零字节；每个到达槽只写一次（写两次会让对端下一轮
// 假到达），等到后要清 0。本轮只有自发自收时 peerNum==0，直接跳过屏障——各 rank 发射次数不一致也不会互相卡住。
TEST(BsrAivKernel, BarrierCoversOnlyActivePeers)
{
    SetupWorld(4);
    ExtraArgsBsr extra{};
    extra.flagBlock = 8;
    AddSend(extra, 1, bsr_aiv_test::UserBuf(0, 0x100), 32, 0, 0x100);
    AddSend(extra, 2, bsr_aiv_test::UserBuf(0, 0x200), 32, 1, 0x200);
    AddSend(extra, 1, bsr_aiv_test::UserBuf(0, 0x300), 32, 2, 0x300); // 与第 1 条同 peer，必须去重
    AddSend(extra, 0, bsr_aiv_test::UserBuf(0, 0x400), 32, 3, 0x400); // 防御：本 rank 不进屏障
    AddRecv(extra, 3, bsr_aiv_test::UserBuf(0, 0x500), 0, 4, 0x500);  // 零字节，不进屏障

    KernelArgs args;
    args.rank = 0;
    args.rankSize = 4;
    RunAllCores(args, extra);

    // 到达槽的发布事件（tag=1）与随后清 0 的收尾事件（tag=0）分开数
    const std::vector<Event> records = SelectIf(EventKind::RECORD, [](const Event& event) {
        return IsBarrierEvent(event) && event.tag == 1;
    });
    ASSERT_EQ(records.size(), 2u);
    std::vector<uint64_t> destinations;
    for (const Event& record : records) {
        destinations.push_back(record.dst);
    }
    std::sort(destinations.begin(), destinations.end());
    const std::vector<uint64_t> expected = {BarrierFlagAddr(1, 0), BarrierFlagAddr(2, 0)};
    EXPECT_EQ(destinations, expected);

    const std::vector<Event> clears = SelectIf(EventKind::RECORD, [](const Event& event) {
        return IsBarrierEvent(event) && event.tag == 0;
    });
    ASSERT_EQ(clears.size(), 2u); // 每个等到过的到达槽都清了 0，下一轮不会假到达

    const std::vector<Event> waits = SelectIf(EventKind::WAIT, IsBarrierEvent);
    ASSERT_EQ(waits.size(), 2u);
    std::vector<uint64_t> expectedWaits = {BarrierFlagAddr(0, 1), BarrierFlagAddr(0, 2)};
    std::sort(expectedWaits.begin(), expectedWaits.end());
    std::vector<uint64_t> actualWaits;
    for (const Event& wait : waits) {
        actualWaits.push_back(wait.dst);
    }
    std::sort(actualWaits.begin(), actualWaits.end());
    EXPECT_EQ(actualWaits, expectedWaits);

    // 本轮只有自发自收 ⇒ peerNum==0，不进出屏障区，各核仍各自 Sync 一次
    SetupWorld(2);
    ExtraArgsBsr selfOnly{};
    selfOnly.flagBlock = 4;
    AddSelfSend(selfOnly, bsr_aiv_test::UserBuf(0, 0x4000), bsr_aiv_test::UserBuf(0, 0x100), 32, 0);
    AddSelfRecv(selfOnly, bsr_aiv_test::UserBuf(0, 0x4000), 32, 0);

    KernelArgs noPeerArgs;
    noPeerArgs.rank = 0;
    RunAllCores(noPeerArgs, selfOnly);

    EXPECT_EQ(SelectIf(EventKind::RECORD, IsBarrierEvent).size(), 0u);
    EXPECT_EQ(SelectIf(EventKind::WAIT, IsBarrierEvent).size(), 0u);
    EXPECT_EQ(Select(EventKind::SYNC).size(), 2u);
}

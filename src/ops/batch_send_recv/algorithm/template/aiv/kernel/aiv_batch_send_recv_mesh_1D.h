/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_BATCH_SEND_RECV_MESH_1D_H
#define AIV_BATCH_SEND_RECV_MESH_1D_H

#include <cstddef>

#include "aiv_communication_base_v2.h"
#include "aiv_defines.h"

using namespace AscendC;

namespace AivBatchSendRecv {

// 单次 kernel 下发的最大任务条数，必须与 host 侧 ops_hccl::AIV_BSR_ITEM_MAX 一致。
// 2*rankSize*itemCapPerPair 在 rankSize<=256 时恰好取到上界 512，没有余量，扩充 item 类型须重算。
constexpr uint32_t BATCH_SEND_RECV_ITEM_MAX = 512;
constexpr uint32_t AIV_BSR_SEND = 0;
constexpr uint32_t AIV_BSR_RECV = 1;
// 自发自收拆成发/收两条 item，走与跨 rank 完全相同的流水（切片、分核、flag 握手），
// 只有数据路径不同：SELF_SEND 直接 GM->GM 写到目的地址，SELF_RECV 不搬数据、只等握手完成。
constexpr uint32_t AIV_BSR_SELF_SEND = 2;
constexpr uint32_t AIV_BSR_SELF_RECV = 3;

// 跨 rank 任务的私有视图：SEND/RECV 用它描述"数据落在哪个桶的哪个位置"。具名定义以便断言 sizeof；
// 两个 u16 让本视图恰好 8B，与 dstAddr 共用 union 时尺寸对齐。
struct AivBsrPeerView {
    uint32_t cclOffset;  // @0 相对 cclBuffer 起始的偏移
    uint16_t remoteRank; // @4 对端 rank
    uint16_t peerRsv;    // @6 显式占位
};

// 布局必须与 host 侧 ops_hccl::AivBsrItemArgs 完全一致（24B）：byteCount/cclOffset 用 u32 的前提是
// host 把中转容量 clamp 到 UB_MAX_DATA_SIZE(256MB) 再按 rankSize 分桶，放开该上限须同步改回 u64。
struct AivBsrItem {
    uint64_t bufAddr;     // @0  本地数据地址（send/self_send: 源地址；recv/self_recv: 目的地址）
    uint32_t byteCount;   // @8  本片需要搬运的字节数
    uint16_t sliceIdx;    // @12 本轮内该 pair 的片序号，收发两侧一致
    uint8_t sendRecvType; // @14 AIV_BSR_SEND / RECV / SELF_SEND / SELF_RECV
    uint8_t rsv;          // @15
    // 两种互斥视图，由 sendRecvType 选择：SELF_SEND 取 dstAddr，SEND/RECV 取 peer，SELF_RECV 都不读。
    // 必须先判类型：把 SELF_SEND 的 remoteRank 位当 rank 用，会去等一个不存在的 rank 而挂死。
    union {
        uint64_t dstAddr;    // @16 仅 SELF_SEND：本卡目的地址
        AivBsrPeerView peer; // @16 仅 SEND/RECV：桶位 + 对端 rank
    };
};

struct ExtraArgsBsr {
    uint64_t itemNum;
    // 每对端在 flag 信箱里预留的格数（= itemCapPerPair），与 host 侧 ops_hccl::ExtraArgsBsr::flagBlock 对齐。
    uint64_t flagBlock;
    AivBsrItem items[BATCH_SEND_RECV_ITEM_MAX];
};

static_assert(sizeof(AivBsrPeerView) == 8, "AivBsrPeerView must be 8B to union with dstAddr");
static_assert(sizeof(AivBsrItem) == 24, "AivBsrItem layout must match host side AivBsrItemArgs");
// 偏移钉死：只比 sizeof 挡不住"两字段互换、总大小不变"。union 内成员属 conditionally-supported，
// 不对其断言，起点与宽度由下面两条（sendRecvType 在 14、总长 24）间接锁死。
static_assert(
    offsetof(AivBsrItem, bufAddr) == 0 && offsetof(AivBsrItem, byteCount) == 8,
    "AivBsrItem bufAddr/byteCount offsets must match host side AivBsrItemArgs");
static_assert(
    offsetof(AivBsrItem, sliceIdx) == 12 && offsetof(AivBsrItem, sendRecvType) == 14,
    "AivBsrItem sliceIdx/sendRecvType offsets must match host side AivBsrItemArgs");
static_assert(
    sizeof(ExtraArgsBsr) == 2 * sizeof(uint64_t) + sizeof(AivBsrItem) * BATCH_SEND_RECV_ITEM_MAX,
    "ExtraArgsBsr layout must match host side ops_hccl::ExtraArgsBsr");

// 16(itemNum+flagBlock) + 24*512 = 12304。
static_assert(sizeof(ExtraArgsBsr) == 12304, "ExtraArgsBsr ABI size must match host side ops_hccl::ExtraArgsBsr");

} // namespace AivBatchSendRecv

// BatchSendRecv 的 AIV 实现：按字节搬运，不区分数据类型。一半核发送（搬进本 rank 的 CCL bucket），
// 一半核接收（从 sender 的 bucket 搬回本地）；自收发共用同一套分核与 flag 握手，只有数据路径不同。
class AivBatchSendRecvMesh1D : public AivCommBase {
public:
    __aicore__ inline AivBatchSendRecvMesh1D() {}

    // 与 AivCommBase::Init 逐行同构（含 buffer 注册顺序），唯一差别在末段：不调 GetTag，直接用 host 下发的
    // tag 作 tag_。即既不读也不写 65M 块里那个全通信域共享的 tag 计数器（GetTag 内部还有一次 SyncAll）。
    // 收益：tag 只由"本轮是本 rank 的第几轮"决定（host 侧换算成 1024+轮次），与通信域里其它 rank / 其它
    // 算子下发过多少次完全无关，因此各 rank 天然一致（旧方案要先把计数器倒推回起点才能保证两侧奇偶相同），
    // 也就不需要任何 tag 回退与失败后还原。代价：不同调用的同号轮次 tag 相同，flag 不复用只靠"消费即清 +
    // 轮末屏障"；一旦某轮失败留下未清的 flag，下一次调用可能被误判为已到达（失败语义见 executor 里
    // Orchestrate 的说明）。
    // 维护约束：本函数与基类 Init 是两份代码，基类改动不会带来编译错误、单测也覆盖不到，漂移的表现是 flag
    // 编址错位或 UB 越界。基类 Init 若有以下三类改动，必须同步本函数：
    //   ① 成员赋值块（aiv_communication_base_v2.h 中 Init 起首的 rank_ / rankSize_ / input_ 等 16 项）；
    //   ② 5 段 flag 编址公式 localOffset / multiOffset / pingpongOffset / countOffset / separateOffset；
    //   ③ UB 注册顺序：InitBuffer(localFlagBuf) + 5 个 GetWithOffset + InitBuffer(inOutQue) +
    //      InitBuffer(inQueueX) / InitBuffer(inQueueY) / InitBuffer(outQueueZ)。
    // 注：同样的公式在基类内部本就有两份（普通 Init 与 SuperKernel 的 Init），本函数是第三份；三处合并属
    // 公共层重构，不在本算子范围内，此处只保证与普通 Init 同步。
    __aicore__ inline void InitBsr(
        GM_ADDR buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank, uint32_t rankSize,
        uint64_t len, uint32_t dataType, uint32_t reduceOp, uint32_t root, uint64_t inputSliceStride,
        uint64_t outputSliceStride, uint64_t repeatNum, uint64_t inputRepeatStride, uint64_t outputRepeatStride,
        GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize, bool isEnableCounter,
        uint32_t numBlocks, bool useDoubleBuffer, bool pingpong, uint32_t roundTag)
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
        pipe.InitBuffer(inOutQue, 1, UB_MAX_DATA_SIZE);

        // BSR 无 reduce 路径，inQueueX/Y/Z 只为与基类保持同一套 UB 注册顺序而保留（CpGM2GM 只用 inOutQue）。
        uint64_t chunkSize = UB_MAX_DATA_SIZE / TILING_NUM / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
        pipe.InitBuffer(inQueueX, 1, chunkSize);
        pipe.InitBuffer(inQueueY, 1, chunkSize);
        pipe.InitBuffer(outQueueZ, 1, chunkSize);

        tag_ = roundTag;
        InitBuffArray(buffIn, pingpong);
    }

    // 发送完成标记按 (对端rank, sliceIdx) 唯一编址，每个 item 独占一格：按 coreIndex 编址时同一核的
    // 后续 item 会覆盖前者，消费侧永远等不到。rankSize*flagBlock <= 256 格，不与屏障区（>=32768）重叠。
    __aicore__ inline uint64_t WriteDoneFlagOffset(uint32_t remoteRank, uint32_t sliceIdx, uint64_t flagBlock) const
    {
        return static_cast<uint64_t>(remoteRank) * flagBlock + sliceIdx;
    }

    __aicore__ inline void
    Producer(AivBatchSendRecv::ExtraArgsBsr& extraArgs, uint64_t itemNum, uint32_t halfCoreNum, int32_t baseTag)
    {
        uint32_t coreIndex = blockIdx_;
        uint64_t flagBlock = extraArgs.flagBlock;
        // 分核按"本方向第几个待办 item"（workIdx）轮转，而非 sliceIdx：后者取值域只有 [0, itemCapPerPair)，
        // rankSize 大时大部分核空转；workIdx 只依赖 item 自身，各核序列一致，归属完备且互斥。
        uint32_t workIdx = 0;
        for (uint64_t i = 0; i < itemNum; i++) {
            AivBatchSendRecv::AivBsrItem& item = extraArgs.items[i];
            bool isSend = (item.sendRecvType == AivBatchSendRecv::AIV_BSR_SEND);
            bool isSelf = (item.sendRecvType == AivBatchSendRecv::AIV_BSR_SELF_SEND);
            // 穷举：SEND / SELF_SEND 在本函数处理，RECV / SELF_RECV 交给 Consumer，其余取值视为非法表项跳过。
            if (item.byteCount == 0 || (!isSend && !isSelf)) {
                continue;
            }
            uint32_t owner = workIdx % halfCoreNum;
            workIdx++;
            if (owner != coreIndex) {
                continue;
            }
            int32_t curTag = baseTag | static_cast<int32_t>(item.sliceIdx & LOW_16_BITS);
            if (isSelf) {
                // 自收发：数据直接落到用户目的地址，不经 cclBuffer；flag 用本 rank 区段（只有自收发会写这一格）。
                __gm__ uint8_t* selfSrc = reinterpret_cast<__gm__ uint8_t*>(item.bufAddr);
                __gm__ uint8_t* selfDst = reinterpret_cast<__gm__ uint8_t*>(item.dstAddr);
                CpGM2GM(selfDst, selfSrc, item.byteCount);
                PipeBarrier<PIPE_ALL>();
                Record(rank_, WriteDoneFlagOffset(rank_, item.sliceIdx, flagBlock), curTag);
                continue;
            }
            __gm__ uint8_t* src = reinterpret_cast<__gm__ uint8_t*>(item.bufAddr);
            __gm__ uint8_t* dst
                = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uint64_t>(myGmIn_) + item.peer.cclOffset);
            CpGM2GM(dst, src, item.byteCount);
            PipeBarrier<PIPE_ALL>();
            // 通知对端：本片数据已写入本 rank 的 CCL bucket，可以来取
            Record(rank_, WriteDoneFlagOffset(item.peer.remoteRank, item.sliceIdx, flagBlock), curTag);
            // 同一 round 内每片占独立桶位和独立 flag 单元，桶位跨 round 复用由每轮 kernel 末尾的
            // BatchSendRecvBarrierAll 保序，这里不再等待对端"已读"，避免多核多对端互相等待形成环路
        }
    }

    __aicore__ inline void
    Consumer(AivBatchSendRecv::ExtraArgsBsr& extraArgs, uint64_t itemNum, uint32_t halfCoreNum, int32_t baseTag)
    {
        uint32_t coreIndex = blockIdx_ - halfCoreNum;
        uint64_t flagBlock = extraArgs.flagBlock;
        // 与 Producer 同一套 workIdx 分核；收方向 items[] 顺序虽不同，判据只依赖 item 自身，覆盖完备。
        uint32_t workIdx = 0;
        for (uint64_t i = 0; i < itemNum; i++) {
            AivBatchSendRecv::AivBsrItem& item = extraArgs.items[i];
            bool isRecv = (item.sendRecvType == AivBatchSendRecv::AIV_BSR_RECV);
            bool isSelf = (item.sendRecvType == AivBatchSendRecv::AIV_BSR_SELF_RECV);
            // 穷举：RECV / SELF_RECV 在本函数处理，SEND / SELF_SEND 交给 Producer，其余取值视为非法表项跳过。
            if (item.byteCount == 0 || (!isRecv && !isSelf)) {
                continue;
            }
            uint32_t owner = workIdx % halfCoreNum;
            workIdx++;
            if (owner != coreIndex) {
                continue;
            }
            int32_t curTag = baseTag | static_cast<int32_t>(item.sliceIdx & LOW_16_BITS);
            if (isSelf) {
                // 自收发：数据已由发方向核直接写进目的地址，这里只需等到握手 flag 置位，
                // 保证"本片自收发已完成"在 device 内有一个与跨 rank 对称的完成点，不再搬数据。
                WaitFlag(rank_, WriteDoneFlagOffset(rank_, item.sliceIdx, flagBlock), curTag);
                // 消费即清：由读方在本核的顺序指令里把刚读过的那一格置 0，不依赖任何跨 rank 时序。
                Record(rank_, WriteDoneFlagOffset(rank_, item.sliceIdx, flagBlock), 0);
                continue;
            }
            // 等发送方把本片数据写入它自己的 CCL bucket（flag 单元按 sliceIdx 唯一，不会被后续 item 覆盖）
            WaitFlag(item.peer.remoteRank, WriteDoneFlagOffset(rank_, item.sliceIdx, flagBlock), curTag);
            __gm__ uint8_t* src = reinterpret_cast<__gm__ uint8_t*>(
                reinterpret_cast<uint64_t>(GetGmIn(item.peer.remoteRank)) + item.peer.cclOffset);
            __gm__ uint8_t* dst = reinterpret_cast<__gm__ uint8_t*>(item.bufAddr);
            CpGM2GM(dst, src, item.byteCount);
            PipeBarrier<PIPE_ALL>();
            // 取完立刻清同一格（实参与 WaitFlag 一致）：读 → 取 → 清同核顺序执行，下一发要过本轮屏障，必在清之后。
            Record(item.peer.remoteRank, WriteDoneFlagOffset(rank_, item.sliceIdx, flagBlock), 0);
        }
    }

    __aicore__ inline void Process(AivBatchSendRecv::ExtraArgsBsr& extraArgs, uint64_t itemNum)
    {
        if (numBlocks_ < 2) {
            return;
        }
        uint32_t halfCoreNum = numBlocks_ / 2;
        if (blockIdx_ >= halfCoreNum * 2) {
            return;
        }

        // tag_ 由 host 按"本轮是第几轮"换算后下发（1024+轮次，值域与其它 AIV 算子的 [1,1000] 隔离，
        // 保证它们的遗留 flag 骗不到我们、我们的也骗不到它们），同一轮在各 rank 上恒等；同一轮内用 sliceIdx
        // 区分同一 pair 的不同片，收发两侧的 sliceIdx 由 host 侧切片顺序决定且严格一致
        int32_t baseTag = static_cast<int32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS;
        if (blockIdx_ < halfCoreNum) {
            Producer(extraArgs, itemNum, halfCoreNum, baseTag);
        } else {
            Consumer(extraArgs, itemNum, halfCoreNum, baseTag);
        }
    }

    // 只对本轮实际交互的对端集合做屏障（复用 BASE_FLAG_OFFSET 区的 per-rank 到达槽，与 BarrierAll 同构）：
    // 零任务 rank / 不对称流量（各 rank 发射次数不一致）不再被全局屏障卡死。
    __aicore__ inline void BatchSendRecvBarrierAll(AivBatchSendRecv::ExtraArgsBsr& extraArgs, uint64_t itemNum)
    {
        SyncAll<true>();
        if (IsIdleCore()) {
            return;
        }

        // 判据必须与 Producer/Consumer 一致（单方向 + byteCount != 0），否则会去等一个不进屏障的对端；
        // 且必须先判类型再读 remoteRank —— union 下该位是 dstAddr 的低 16 位。
        uint32_t peers[AivBatchSendRecv::BATCH_SEND_RECV_ITEM_MAX];
        uint32_t peerNum = 0;
        for (uint64_t i = 0; i < itemNum && peerNum < AivBatchSendRecv::BATCH_SEND_RECV_ITEM_MAX; i++) {
            uint8_t type = extraArgs.items[i].sendRecvType;
            if (type != AivBatchSendRecv::AIV_BSR_SEND && type != AivBatchSendRecv::AIV_BSR_RECV) {
                continue;
            }
            if (extraArgs.items[i].byteCount == 0) {
                continue;
            }
            uint32_t p = extraArgs.items[i].peer.remoteRank;
            // 跨 rank 项的 remoteRank 恒不等于本 rank（host 侧按 remoteRank 是否等于 myRank 分流），
            // 这里保留一条防御：万一 host 打错标签，不至于让本 rank 去等自己。
            if (p == rank_) {
                continue;
            }
            bool dup = false;
            for (uint32_t j = 0; j < peerNum; j++) {
                if (peers[j] == p) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                peers[peerNum++] = p;
            }
        }
        if (peerNum == 0) {
            return; // 本轮无跨 rank 交互，无需屏障
        }

        // 按 peerNum 分核（startRank 是 peers 的下标），保证同一到达槽只有一个核 Record：不分核时
        // 晚到的核会再写一次 1，对端读走后残留的 1 会让下一轮 WaitFlag 假到达，跨轮保序失效。
        uint32_t perCoreRankNum = peerNum / numBlocks_;
        uint32_t remainRankNum = peerNum % numBlocks_;
        uint32_t curCoreRankNum = blockIdx_ < remainRankNum ? perCoreRankNum + 1 : perCoreRankNum;
        uint32_t startRank
            = blockIdx_ < remainRankNum ? (perCoreRankNum + 1) * blockIdx_ : perCoreRankNum * blockIdx_ + remainRankNum;

        // 槽位编址与 BarrierAll 逐字一致：BASE_FLAG_OFFSET 是块内绝对偏移（9M），第几格 = 偏移 / FLAG_SIZE。
        // 减 gmOutOffset_ 是因为 GetGmOut() 会把它加回来，一加一减抵消后恒落在块内 9M 处。
        uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset_ + rank_ * FLAG_SIZE;
        for (uint32_t i = startRank; i < startRank + curCoreRankNum; i++) {
            Record(peers[i], flag_offset / FLAG_SIZE, 1);
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = startRank; i < startRank + curCoreRankNum; i++) {
            uint64_t flag_offset = BASE_FLAG_OFFSET - gmOutOffset_ + peers[i] * FLAG_SIZE;
            WaitFlag(rank_, flag_offset / FLAG_SIZE, 1);
            Record(rank_, flag_offset / FLAG_SIZE, 0);
        }
    }
};

__aicore__ inline void AivBatchSendRecvV2Mesh1D(KERNEL_ARGS_DEF, AivBatchSendRecv::ExtraArgsBsr& extraArgs)
{
    AivBatchSendRecvMesh1D op;
    // 数据区直接用通信域 cclBuffer（pingpong=false：gmInOffset_ 恒为 0），flag/tag/屏障仍在 65M 块内。
    // sliceId 在本算子承载 tag 值（host 侧 = 1024 + 轮次，越界回绕）：信箱窗口由 tag_ 奇偶决定，而 tag_ 完全
    // 由它决定、不再从 65M 块的共享计数器里取，故两侧奇偶天然一致，不依赖通信域里谁先跑过什么。
    op.InitBsr(KERNEL_CLASS_INIT, true, false, sliceId);
    // itemNum 上界钳制：三处循环都以它索引 12KB 参数块尾部的 items[]，越界会被当成 byteCount/dstAddr
    // 去搬 GM。host 侧两道校验使正常路径到不了，这里只做最后防御，超限截断并显式传参（不写回入参）。
    const uint64_t itemNum = extraArgs.itemNum < AivBatchSendRecv::BATCH_SEND_RECV_ITEM_MAX ?
                                 extraArgs.itemNum :
                                 static_cast<uint64_t>(AivBatchSendRecv::BATCH_SEND_RECV_ITEM_MAX);
    op.Process(extraArgs, itemNum);
    // 只对本轮活跃对端做屏障（内部按 peerNum 分核），而非全局 BarrierAll：避免各 rank 发射次数不一致时互等。
    op.BatchSendRecvBarrierAll(extraArgs, itemNum);
}

#endif // AIV_BATCH_SEND_RECV_MESH_1D_H

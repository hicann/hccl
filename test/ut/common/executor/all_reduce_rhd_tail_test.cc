/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <numeric>
#include <vector>
#include "ins_v2_all_reduce_sequence_rhd_tail.h"

using namespace ops_hccl;

extern "C" void DlogRecord(int32_t, int32_t, const char*, ...) {}

bool HcclCheckLogLevel(int, int) { return false; }
bool IsErrorToWarn() { return false; }

namespace ops_hccl {
// Execute local copies on CPU; CheckerL2 separately verifies cross-rank instruction ordering.
HcclResult LocalCopy(const ThreadHandle&, const DataSlice& src, const DataSlice& dst)
{
    if (src.addr_ == nullptr || dst.addr_ == nullptr || src.size_ != dst.size_ || src.count_ != dst.count_) {
        return HCCL_E_PARA;
    }
    std::copy_n(static_cast<const u8*>(src.addr_) + src.offset_, src.size_, static_cast<u8*>(dst.addr_) + dst.offset_);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

TEST(AllReduceRhdTail, AlignedInputKeepsEntireWorkspace)
{
    std::vector<u8> buffer(4096);
    HcclMem workspace{};
    workspace.addr = buffer.data();
    workspace.size = buffer.size();
    ArSeqRhdTail tail;
    ASSERT_EQ(ArSeqRhdPrepareTailWorkspace(144, 72, 4, workspace, tail), HCCL_SUCCESS);
    EXPECT_EQ(workspace.size, buffer.size());
    EXPECT_EQ(tail.storage.addr, nullptr);
}

TEST(AllReduceRhdTail, TilesAreAlignedAndOutsideAlgorithmWorkspace)
{
    std::vector<u8> buffer(4096);
    HcclMem workspace{};
    workspace.addr = buffer.data();
    workspace.size = buffer.size();
    ArSeqRhdTail tail;
    ASSERT_EQ(ArSeqRhdPrepareTailWorkspace(73, 72, 4, workspace, tail), HCCL_SUCCESS);
    EXPECT_EQ(tail.inputOffset, workspace.size);
    EXPECT_EQ(tail.inputOffset % HCCL_MIN_SLICE_ALIGN, 0);
    EXPECT_EQ(tail.outputOffset % HCCL_MIN_SLICE_ALIGN, 0);
    EXPECT_LE(tail.inputOffset + 72 * 4, tail.outputOffset);
    EXPECT_LE(tail.outputOffset + 72 * 4, buffer.size());
}

TEST(AllReduceRhdTail, InvalidOrInsufficientWorkspaceIsRejectedWithoutModification)
{
    std::vector<u8> buffer(512);
    HcclMem workspace{};
    workspace.addr = buffer.data();
    workspace.size = buffer.size();
    ArSeqRhdTail tail;
    EXPECT_EQ(ArSeqRhdPrepareTailWorkspace(73, 0, 4, workspace, tail), HCCL_E_PARA);
    EXPECT_EQ(ArSeqRhdPrepareTailWorkspace(73, 72, 0, workspace, tail), HCCL_E_PARA);
    EXPECT_EQ(ArSeqRhdPrepareTailWorkspace(73, 72, 4, workspace, tail), HCCL_E_INTERNAL);
    EXPECT_EQ(workspace.size, buffer.size());
}

TEST(AllReduceRhdTail, EveryInputElementIsProcessedExactlyOnce)
{
    for (u64 ranks : {2, 3, 6, 12, 16, 24, 72}) {
        const u64 capacity = 3 * ranks;
        for (u64 total = 0; total <= 1000; total++) {
            u64 processed = 0;
            while (processed < total) {
                u64 count = ArSeqRhdSelectLoopCount(total - processed, capacity, ranks);
                ASSERT_GT(count, 0);
                ASSERT_LE(count, capacity);
                if (count < ranks) {
                    EXPECT_EQ(count, total - processed);
                } else {
                    EXPECT_EQ(count % ranks, 0);
                }
                processed += count;
            }
            EXPECT_EQ(processed, total);
        }
    }
}

TEST(AllReduceRhdTail, OneMiBOn72RanksPreservesExactCount)
{
    const u64 totalCount = 1024 * 1024 / sizeof(int32_t);
    u64 processed = 0;
    u64 tailCount = 0;
    while (processed < totalCount) {
        const u64 count = ArSeqRhdSelectLoopCount(totalCount - processed, 4032, 72);
        if (count < 72) {
            tailCount = count;
        }
        processed += count;
    }
    EXPECT_EQ(processed, totalCount);
    EXPECT_EQ(tailCount, 64);
}

TEST(AllReduceRhdTail, StageInitializesEveryLaneAndOutputCopyPreservesCanaries)
{
    for (u64 validCount : {1, 5, 15}) {
        constexpr u64 RANKS = 16;
        constexpr u64 PROCESSED = 32;
        constexpr int32_t CANARY = -9999;
        std::vector<int32_t> input(PROCESSED + validCount);
        std::iota(input.begin(), input.end(), -9);
        std::vector<int32_t> output(input.size() + 3, CANARY);
        std::vector<int32_t> buffer(1024, CANARY);
        HcclMem workspace{};
        workspace.addr = buffer.data();
        workspace.size = buffer.size() * sizeof(int32_t);
        ArSeqRhdTail tail;
        ASSERT_EQ(ArSeqRhdPrepareTailWorkspace(input.size(), RANKS, sizeof(int32_t), workspace, tail), HCCL_SUCCESS);
        OpParam param{};
        param.inputPtr = input.data();
        param.outputPtr = output.data();
        param.DataDes.count = input.size();
        ASSERT_EQ(ArSeqRhdStageTail(param, tail, PROCESSED, validCount, 0), HCCL_SUCCESS);
        auto stagedInput = buffer.data() + tail.inputOffset / sizeof(int32_t);
        auto stagedOutput = buffer.data() + tail.outputOffset / sizeof(int32_t);
        for (u64 i = 0; i < RANKS; i++) {
            EXPECT_EQ(stagedInput[i], input[PROCESSED + i % validCount]);
            stagedOutput[i] = 2 * stagedInput[i];
        }
        ASSERT_EQ(ArSeqRhdCopyTailOutput(param, tail, PROCESSED, validCount, 0), HCCL_SUCCESS);
        for (u64 i = 0; i < output.size(); i++) {
            EXPECT_EQ(output[i], i >= PROCESSED && i < input.size() ? 2 * input[i] : CANARY);
        }
    }
}

// AICPU 二级：RS 输出网格占收缩后 workspace 前半区，scratch 占后半区，尾块 tile 在其后。
TEST(AllReduceRhdTail, AicpuTwoLevelTailTilesStayOutsideGridAndScratch)
{
    for (u64 count : {100ULL, 101ULL}) { // 100/16 余 4，101/16 余 5
        constexpr u64 D0 = 8;
        constexpr u64 D1 = 2;
        constexpr u64 RANK_ALIGN = D0 * D1;
        std::vector<u8> buffer(4096);
        HcclMem workspace{};
        workspace.addr = buffer.data();
        workspace.size = buffer.size();
        ArSeqRhdTail tail;
        ASSERT_EQ(ArSeqRhdPrepareTailWorkspace(count, RANK_ALIGN, sizeof(int32_t), workspace, tail), HCCL_SUCCESS);
        // 二级分区：前半区 RS 网格 [0, size/2)，后半区 scratch [size/2, size)。
        const u64 half = workspace.size / 2;
        EXPECT_LE(half, tail.inputOffset); // scratch 不侵入 input tile
        EXPECT_LE(tail.inputOffset, tail.outputOffset);
        EXPECT_LE(tail.outputOffset + RANK_ALIGN * sizeof(int32_t), buffer.size());
        EXPECT_EQ(tail.inputOffset % HCCL_MIN_SLICE_ALIGN, 0);
        EXPECT_EQ(tail.outputOffset % HCCL_MIN_SLICE_ALIGN, 0);
        // 输出 tile 位于收缩后 workspace 之外，不会与网格/scratch 重叠。
        EXPECT_GE(tail.outputOffset, workspace.size);
    }
}

// AICPU 三级：RSL0 结果区 + mesh scratch 必须整体落在收缩后 workspace 内，tile 在其后。
TEST(AllReduceRhdTail, AicpuThreeLevelScratchPartitionStaysInsideShrunkenWorkspace)
{
    constexpr u64 D0 = 8;
    constexpr u64 D1 = 2;
    constexpr u64 D2 = 2;
    constexpr u64 RANK_ALIGN = D0 * D1 * D2;
    constexpr u64 TYPE_SIZE = sizeof(int32_t);
    std::vector<u8> buffer(8192);
    HcclMem workspace{};
    workspace.addr = buffer.data();
    workspace.size = buffer.size();
    ArSeqRhdTail tail;
    ASSERT_EQ(ArSeqRhdPrepareTailWorkspace(100, RANK_ALIGN, TYPE_SIZE, workspace, tail), HCCL_SUCCESS);
    // 与 executor 相同的三级分区：scratchMultiplier = RSL0 CalcScratchMultiple = D0。
    const u64 scratchMultiplier = D0;
    const u64 sliceNum = scratchMultiplier + 1;
    const u64 slice = workspace.size / sliceNum / TYPE_SIZE * TYPE_SIZE;
    const u64 rsResultSize = slice;
    const u64 meshCommSize = scratchMultiplier * slice;
    EXPECT_LE(rsResultSize + meshCommSize, workspace.size);
    EXPECT_LE(workspace.size, tail.inputOffset); // scratch 不侵入 input tile
    EXPECT_LE(tail.inputOffset, tail.outputOffset);
    EXPECT_LE(tail.outputOffset + RANK_ALIGN * TYPE_SIZE, buffer.size());
}

// 二/三级共用的 loop plan：整除前缀 + 单轮余量，且余量轮 curr 补齐到 rankAlign。
TEST(AllReduceRhdTail, AicpuLoopPlanUsesAlignedPrefixAndSinglePaddedTail)
{
    for (u64 rankAlign : {16ULL, 32ULL}) {
        const u64 maxCountPerLoop = rankAlign * 6;
        for (u64 total : {1ULL, 7ULL, rankAlign - 1, rankAlign, rankAlign + 1, rankAlign * 7 + 5}) {
            u64 processed = 0;
            u64 tailRounds = 0;
            u64 tailValid = 0;
            while (processed < total) {
                const ArSeqRhdLoopPlan plan = ArSeqRhdPlanLoop(total - processed, maxCountPerLoop, rankAlign);
                ASSERT_GT(plan.validCount, 0);
                ASSERT_LE(plan.validCount, maxCountPerLoop);
                if (plan.paddedTail) {
                    EXPECT_LT(plan.validCount, rankAlign);
                    EXPECT_EQ(plan.currDataCount, rankAlign);
                    tailRounds++;
                    tailValid = plan.validCount;
                } else {
                    EXPECT_EQ(plan.validCount % rankAlign, 0);
                    EXPECT_EQ(plan.validCount, plan.currDataCount);
                }
                processed += plan.validCount;
            }
            EXPECT_EQ(processed, total);
            EXPECT_LE(tailRounds, 1ULL);
            EXPECT_EQ(tailValid, total % rankAlign);
        }
    }
}

// AICPU 二/三级：1MiB 量级非整除 count 的 staged 尾块有效字节回拷闭环。
TEST(AllReduceRhdTail, AicpuTailStagingCopiesOnlyValidBytesForBothLevels)
{
    // 262144 元素（int32，1MiB）对 16/32 rank 均整除；附加 8 元素制造非整除余量。
    constexpr u64 BASE_COUNT = 262144;
    constexpr u64 EXTRA = 8;
    for (u64 rankAlign : {16ULL, 32ULL}) {
        const u64 total = BASE_COUNT + EXTRA;
        const u64 maxCountPerLoop = rankAlign * 1024;
        constexpr int32_t CANARY = -7777;
        std::vector<int32_t> input(total);
        std::iota(input.begin(), input.end(), 1);
        std::vector<int32_t> output(total + rankAlign, CANARY);
        // 收缩后 workspace 需容纳一轮对齐前缀，尾部再留两块 rank-align tile。
        std::vector<int32_t> buffer(2 * maxCountPerLoop + 2 * 256, CANARY);
        HcclMem workspace{};
        workspace.addr = buffer.data();
        workspace.size = buffer.size() * sizeof(int32_t);
        ArSeqRhdTail tail;
        ASSERT_EQ(ArSeqRhdPrepareTailWorkspace(total, rankAlign, sizeof(int32_t), workspace, tail), HCCL_SUCCESS);
        OpParam param{};
        param.inputPtr = input.data();
        param.outputPtr = output.data();
        param.DataDes.count = total;
        u64 processed = 0;
        u64 paddedRounds = 0;
        while (processed < total) {
            const ArSeqRhdLoopPlan plan = ArSeqRhdPlanLoop(total - processed, maxCountPerLoop, rankAlign);
            if (plan.paddedTail) {
                paddedRounds++;
                ASSERT_EQ(ArSeqRhdStageTail(param, tail, processed, plan.validCount, 0), HCCL_SUCCESS);
                // 模拟归约：尾块 tile 内 rankAlign 个槽全部有值，有效前缀为 2*input。
                auto staged = buffer.data() + tail.inputOffset / sizeof(int32_t);
                auto stagedOut = buffer.data() + tail.outputOffset / sizeof(int32_t);
                for (u64 i = 0; i < rankAlign; i++) {
                    EXPECT_EQ(staged[i], input[processed + i % plan.validCount]);
                    stagedOut[i] = 2 * staged[i];
                }
                ASSERT_EQ(ArSeqRhdCopyTailOutput(param, tail, processed, plan.validCount, 0), HCCL_SUCCESS);
            } else {
                // 整除前缀轮由链直接回写用户输出（executor 不追加回拷）。
                for (u64 i = 0; i < plan.validCount; i++) {
                    output[processed + i] = 2 * input[processed + i];
                }
            }
            processed += plan.validCount;
        }
        EXPECT_EQ(processed, total);
        EXPECT_EQ(paddedRounds, 1ULL);
        for (u64 i = 0; i < total; i++) {
            EXPECT_EQ(output[i], 2 * input[i]);
        }
        for (u64 i = total; i < output.size(); i++) {
            EXPECT_EQ(output[i], CANARY); // 回拷不越过有效字节
        }
    }
}

// 整除 count：新 plan 的每轮 curr 必须与原 for(loop) 序列逐轮相等，且从不触发 staged 尾块。
TEST(AllReduceRhdTail, AicpuDivisibleLoopPlanMatchesLegacyLoop)
{
    for (u64 rankAlign : {16ULL, 32ULL}) {
        const u64 maxCountPerLoop = rankAlign * 5;
        for (u64 total = 0; total <= maxCountPerLoop * 5; total += rankAlign) {
            const u64 legacyLoopTimes = total / maxCountPerLoop + static_cast<u64>(total % maxCountPerLoop != 0);
            u64 legacyProcessed = 0;
            u64 processed = 0;
            u64 rounds = 0;
            while (processed < total) {
                const ArSeqRhdLoopPlan plan = ArSeqRhdPlanLoop(total - processed, maxCountPerLoop, rankAlign);
                const u64 legacyCurr = (rounds == legacyLoopTimes - 1) ? total - legacyProcessed : maxCountPerLoop;
                EXPECT_FALSE(plan.paddedTail);
                EXPECT_EQ(plan.validCount, legacyCurr);
                EXPECT_EQ(plan.currDataCount, legacyCurr);
                processed += plan.validCount;
                legacyProcessed += legacyCurr;
                rounds++;
            }
            EXPECT_EQ(rounds, legacyLoopTimes);
            EXPECT_EQ(processed, total);
        }
    }
}

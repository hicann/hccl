/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// RS NHR-step 几何：黄金值、非 2 的幂取整、方向契约、ΣnSlices 不变量、非法输入。

#include "nhr_test_access.h"

namespace nhr_ut {
namespace {

    struct ExpectedStep {
        u32 toRank;
        u32 fromRank;
        u32 nSlices;
        std::vector<u32> txSliceIdxs;
        std::vector<u32> rxSliceIdxs;
    };

    AicpuNHRStepInfo StepInfoOf(InsTempReduceScatterNhrDpu& tpl, u32 step)
    {
        AicpuNHRStepInfo info;
        EXPECT_EQ(tpl.GetStepInfo(step, GetNHRStepNum(tpl.templateRankSize_), info), HCCL_SUCCESS);
        return info;
    }

    void ExpectSteps(u32 rankSize, u32 myRank, const std::vector<ExpectedStep>& expected)
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, myRank, MakeRankIds(rankSize));
        ASSERT_EQ(GetNHRStepNum(rankSize), expected.size());

        for (u32 step = 0; step < expected.size(); ++step) {
            SCOPED_TRACE(testing::Message() << "rankSize=" << rankSize << " myRank=" << myRank << " step=" << step);
            AicpuNHRStepInfo info = StepInfoOf(tpl, step);
            EXPECT_EQ(info.step, step);
            EXPECT_EQ(info.toRank, expected[step].toRank);
            EXPECT_EQ(info.fromRank, expected[step].fromRank);
            EXPECT_EQ(info.nSlices, expected[step].nSlices);
            EXPECT_EQ(info.txSliceIdxs, expected[step].txSliceIdxs);
            EXPECT_EQ(info.rxSliceIdxs, expected[step].rxSliceIdxs);
        }
    }

    TEST(NhrGeometryTest, GoldenValuesRankSize2)
    {
        ExpectSteps(2, 0, {{1, 1, 1, {1}, {0}}});
        ExpectSteps(2, 1, {{0, 0, 1, {0}, {1}}});
    }

    TEST(NhrGeometryTest, GoldenValuesRankSize3)
    {
        ExpectSteps(3, 0, {{2, 1, 1, {2}, {0}}, {1, 2, 1, {1}, {0}}});
        ExpectSteps(3, 1, {{0, 2, 1, {0}, {1}}, {2, 0, 1, {2}, {1}}});
    }

    TEST(NhrGeometryTest, GoldenValuesRankSize4)
    {
        ExpectSteps(4, 0, {{3, 1, 2, {3, 1}, {0, 2}}, {2, 2, 1, {2}, {0}}});
        ExpectSteps(4, 3, {{2, 0, 2, {2, 0}, {3, 1}}, {1, 1, 1, {1}, {3}}});
    }

    TEST(NhrGeometryTest, GoldenValuesRankSize5)
    {
        ExpectSteps(5, 1, {{0, 2, 2, {0, 3}, {1, 4}}, {4, 3, 1, {4}, {1}}, {2, 0, 1, {2}, {1}}});
        ExpectSteps(5, 0, {{4, 1, 2, {4, 2}, {0, 3}}, {3, 2, 1, {3}, {0}}, {1, 4, 1, {1}, {0}}});
    }

    TEST(NhrGeometryTest, GoldenValuesRankSize6)
    {
        ExpectSteps(6, 0, {{5, 1, 3, {5, 3, 1}, {0, 4, 2}}, {4, 2, 1, {4}, {0}}, {2, 4, 1, {2}, {0}}});
        ExpectSteps(6, 5, {{4, 0, 3, {4, 2, 0}, {5, 3, 1}}, {3, 1, 1, {3}, {5}}, {1, 3, 1, {1}, {5}}});
    }

    TEST(NhrGeometryTest, GoldenValuesRankSize8)
    {
        ExpectSteps(8, 0, {{7, 1, 4, {7, 5, 3, 1}, {0, 6, 4, 2}}, {6, 2, 2, {6, 2}, {0, 4}}, {4, 4, 1, {4}, {0}}});
    }

    TEST(NhrGeometryTest, SumOfSlicesEqualsRankSizeMinusOne)
    {
        for (u32 rankSize = 2; rankSize <= 9; ++rankSize) {
            for (u32 myRank = 0; myRank < rankSize; ++myRank) {
                InsTempReduceScatterNhrDpu tpl;
                SetupTemplate(tpl, myRank, MakeRankIds(rankSize));
                u32 total = 0;
                for (u32 step = 0; step < GetNHRStepNum(rankSize); ++step) {
                    AicpuNHRStepInfo info = StepInfoOf(tpl, step);
                    EXPECT_EQ(info.txSliceIdxs.size(), info.nSlices);
                    EXPECT_EQ(info.rxSliceIdxs.size(), info.nSlices);
                    total += info.nSlices;
                }
                EXPECT_EQ(total, rankSize - 1u);
            }
        }
    }

    TEST(NhrGeometryTest, SliceCountMatchesFloorFormula)
    {
        // nSlices = floor((N - 1 + 2^s) / 2^(s+1))，非 2 的幂时取整行为必须稳定
        for (u32 rankSize = 2; rankSize <= 9; ++rankSize) {
            InsTempReduceScatterNhrDpu tpl;
            SetupTemplate(tpl, 0, MakeRankIds(rankSize));
            for (u32 step = 0; step < GetNHRStepNum(rankSize); ++step) {
                AicpuNHRStepInfo info = StepInfoOf(tpl, step);
                const u32 expected = (rankSize - 1 + (1u << step)) / (1u << (step + 1));
                EXPECT_EQ(info.nSlices, expected) << "rankSize=" << rankSize << " step=" << step;
            }
        }
    }

    TEST(NhrGeometryTest, DeltaRankIncreasesWithStep)
    {
        // 方向契约：RS 版 deltaRank = 1 << step 递增，与 AG 版 NHR 的递减方向相反
        for (u32 rankSize = 2; rankSize <= 9; ++rankSize) {
            for (u32 myRank = 0; myRank < rankSize; ++myRank) {
                InsTempReduceScatterNhrDpu tpl;
                SetupTemplate(tpl, myRank, MakeRankIds(rankSize));
                for (u32 step = 0; step < GetNHRStepNum(rankSize); ++step) {
                    AicpuNHRStepInfo info = StepInfoOf(tpl, step);
                    const u32 delta = 1u << step;
                    EXPECT_EQ((myRank + rankSize - info.toRank) % rankSize, delta)
                        << "rankSize=" << rankSize << " myRank=" << myRank << " step=" << step;
                    EXPECT_EQ((info.fromRank + rankSize - myRank) % rankSize, delta)
                        << "rankSize=" << rankSize << " myRank=" << myRank << " step=" << step;
                }
            }
        }
    }

    TEST(NhrGeometryTest, SendRecvAreReciprocal)
    {
        // idx 的收端必须是收端的发端，否则 write-reduce 会写到没有对端等待的槽位
        for (u32 rankSize = 2; rankSize <= 9; ++rankSize) {
            std::vector<std::vector<AicpuNHRStepInfo>> allSteps(rankSize);
            for (u32 myRank = 0; myRank < rankSize; ++myRank) {
                InsTempReduceScatterNhrDpu tpl;
                SetupTemplate(tpl, myRank, MakeRankIds(rankSize));
                for (u32 step = 0; step < GetNHRStepNum(rankSize); ++step) {
                    allSteps[myRank].push_back(StepInfoOf(tpl, step));
                }
            }
            for (u32 myRank = 0; myRank < rankSize; ++myRank) {
                for (u32 step = 0; step < allSteps[myRank].size(); ++step) {
                    const AicpuNHRStepInfo& info = allSteps[myRank][step];
                    EXPECT_EQ(allSteps[info.fromRank][step].toRank, myRank);
                    EXPECT_EQ(allSteps[info.toRank][step].fromRank, myRank);
                }
            }
        }
    }

    TEST(NhrGeometryTest, TxAndRxSlicesAreDisjointWhenMultipleSlices)
    {
        for (u32 rankSize = 2; rankSize <= 9; ++rankSize) {
            for (u32 myRank = 0; myRank < rankSize; ++myRank) {
                InsTempReduceScatterNhrDpu tpl;
                SetupTemplate(tpl, myRank, MakeRankIds(rankSize));
                for (u32 step = 0; step < GetNHRStepNum(rankSize); ++step) {
                    AicpuNHRStepInfo info = StepInfoOf(tpl, step);
                    if (info.nSlices <= 1) {
                        continue;
                    }
                    for (u32 txIdx : info.txSliceIdxs) {
                        EXPECT_EQ(
                            std::find(info.rxSliceIdxs.begin(), info.rxSliceIdxs.end(), txIdx), info.rxSliceIdxs.end())
                            << "rankSize=" << rankSize << " myRank=" << myRank << " step=" << step << " tx=" << txIdx;
                    }
                }
            }
        }
    }

    TEST(NhrGeometryTest, NonContiguousRankGroupUsesLogicalIndex)
    {
        // 组表 {0,2,4,6} 中物理 rank 6 的逻辑下标为 3
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 6, {0, 2, 4, 6});
        AicpuNHRStepInfo info = StepInfoOf(tpl, 0);
        EXPECT_EQ(info.myRank, 3u);
        EXPECT_EQ(info.toRank, 2u);
        EXPECT_EQ(info.fromRank, 0u);
        EXPECT_EQ(tpl.GetRankFromMap(info.toRank), 4u);
        EXPECT_EQ(tpl.GetRankFromMap(info.fromRank), 0u);

        info = StepInfoOf(tpl, 1);
        EXPECT_EQ(info.toRank, 1u);
        EXPECT_EQ(info.fromRank, 1u);
        EXPECT_EQ(tpl.GetRankFromMap(info.toRank), 2u);
    }

    TEST(NhrGeometryTest, RankNotInGroupReturnsParaError)
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 7, {0, 2, 4, 6});
        AicpuNHRStepInfo info;
        EXPECT_EQ(tpl.GetStepInfo(0, 2, info), HCCL_E_PARA);
    }

    TEST(NhrGeometryTest, ZeroRankSizeReturnsParaErrorInsteadOfDivideByZero)
    {
        InsTempReduceScatterNhrDpu tpl;
        tpl.myRank_ = 0;
        AicpuNHRStepInfo info;
        EXPECT_EQ(tpl.GetStepInfo(0, 1, info), HCCL_E_PARA);
    }

    TEST(NhrGeometryTest, ReusedStepInfoDoesNotAccumulateSlices)
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(4));
        AicpuNHRStepInfo info;
        ASSERT_EQ(tpl.GetStepInfo(0, 2, info), HCCL_SUCCESS);
        ASSERT_EQ(info.txSliceIdxs.size(), 2u);
        ASSERT_EQ(tpl.GetStepInfo(1, 2, info), HCCL_SUCCESS);
        EXPECT_EQ(info.txSliceIdxs.size(), 1u);
        EXPECT_EQ(info.rxSliceIdxs.size(), 1u);
    }

    TEST(NhrGeometryTest, StepNumGuardMotivation)
    {
        // GetNHRStepNum(0) 因无符号下溢返回 32，这是 GetStepInfo 必须显式做除 0 保护的原因
        EXPECT_EQ(GetNHRStepNum(0), 32u);
        EXPECT_EQ(GetNHRStepNum(1), 0u);
        EXPECT_EQ(GetNHRStepNum(2), 1u);
        EXPECT_EQ(GetNHRStepNum(3), 2u);
        EXPECT_EQ(GetNHRStepNum(4), 2u);
        EXPECT_EQ(GetNHRStepNum(5), 3u);
        EXPECT_EQ(GetNHRStepNum(8), 3u);
    }

} // namespace
} // namespace nhr_ut

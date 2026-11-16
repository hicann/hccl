/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// DPUKernelRun host 侧防御校验全路径断言覆盖：subCommRanks 空/首层空、rankSize<2、步骤游标
// （rhdStepCnt==0、rhdStepIdx>=rhdStepCnt）、allRank 三向量等长契约、dataType 越界与字节数为 0、
// 本地/远端步骤表长度 mismatch。DPUKernelRun 的传输部分在 AICPU_COMPILE 下编译掉，防御校验
// 已置于守卫之外（见 ins_temp_reduce_scatter_rhd_dpu_inter.cc 的 DPUKernelRun），host 单测可直接调用。

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <vector>

#include "ins_temp_reduce_scatter_rhd_dpu_inter.h"

namespace rhd_dpu_inter_ut {
namespace {

    using namespace ops_hccl;

    // 构造能通过 DPUKernelRun 前段校验（subCommRanks/rankSize/GetAlgRank/allRank*/dataType）的最小入参
    TemplateDataParams MakeParams(u32 rankSize)
    {
        TemplateDataParams params;
        params.dataType = HCCL_DATA_TYPE_FP16; // DATATYPE_SIZE_TABLE[FP16]=2，非 0
        params.allRankSliceSize.assign(rankSize, 16);
        params.allRankDispls.resize(rankSize);
        for (u32 i = 0; i < rankSize; ++i) {
            params.allRankDispls[i] = static_cast<u64>(i) * 32;
        }
        params.allRankProcessedDataCount.assign(rankSize, 4);
        params.rhdStepCnt = 0;
        params.rhdStepIdx = 0;
        return params;
    }

    std::vector<std::vector<u32>> MakeSubCommRanks(u32 rankSize)
    {
        std::vector<u32> rankIds(rankSize);
        for (u32 i = 0; i < rankSize; ++i) {
            rankIds[i] = i;
        }
        return {rankIds};
    }

    // rankSize=2（2 次幂）本地步骤表长度 = 1（仅 1 个 MAIN 步，无 PRE/POST）
    TEST(RhdDpuInterGuardTest, StepCntZeroIsRejected)
    {
        InsTempReduceScatterRhdDpuInter tpl;
        TemplateDataParams params = MakeParams(2);
        params.rhdStepCnt = 0;
        params.rhdStepIdx = 0;
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, MakeSubCommRanks(2)), HCCL_E_PARA);
    }

    // subCommRanks 外层为空 / 首层为空：入口两条独立防御
    TEST(RhdDpuInterGuardTest, EmptySubCommIsRejected)
    {
        InsTempReduceScatterRhdDpuInter tpl;
        TemplateDataParams params = MakeParams(2);
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, {}), HCCL_E_PARA);
        std::vector<std::vector<u32>> emptyInner = {{}};
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, emptyInner), HCCL_E_PARA);
    }

    // rankSize<2：单 rank 子通信域（N==1 由 AICPU 侧拦截，DPU 侧防御报 E_PARA）
    TEST(RhdDpuInterGuardTest, SingleRankSubCommIsRejected)
    {
        InsTempReduceScatterRhdDpuInter tpl;
        TemplateDataParams params = MakeParams(1);
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, MakeSubCommRanks(1)), HCCL_E_PARA);
    }

    // allRank 三向量等长 = rankSize 契约（需求 5.2 变长分片唯一强制点）：任一向量短缺即拒
    TEST(RhdDpuInterGuardTest, AllRankVectorLengthMismatchIsRejected)
    {
        InsTempReduceScatterRhdDpuInter tpl;
        TemplateDataParams params = MakeParams(2);
        params.rhdStepCnt = 1;
        params.rhdStepIdx = 0;
        params.allRankSliceSize.resize(1); // rankSize-1
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, MakeSubCommRanks(2)), HCCL_E_PARA);

        TemplateDataParams params2 = MakeParams(2);
        params2.rhdStepCnt = 1;
        params2.rhdStepIdx = 0;
        params2.allRankDispls.resize(1);
        EXPECT_EQ(tpl.DPUKernelRun(params2, {}, 0, MakeSubCommRanks(2)), HCCL_E_PARA);

        TemplateDataParams params3 = MakeParams(2);
        params3.rhdStepCnt = 1;
        params3.rhdStepIdx = 0;
        params3.allRankProcessedDataCount.resize(1);
        EXPECT_EQ(tpl.DPUKernelRun(params3, {}, 0, MakeSubCommRanks(2)), HCCL_E_PARA);
    }

    // dataType 越界（RESERVED 及以上）：查表前的显式拦截，防御分支确定可达
    TEST(RhdDpuInterGuardTest, InvalidDataTypeIsRejected)
    {
        InsTempReduceScatterRhdDpuInter tpl;
        TemplateDataParams params = MakeParams(2);
        params.rhdStepCnt = 1;
        params.rhdStepIdx = 0;
        params.dataType = HCCL_DATA_TYPE_RESERVED;
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, MakeSubCommRanks(2)), HCCL_E_PARA);
    }

    TEST(RhdDpuInterGuardTest, StepIdxOutOfRangeIsRejected)
    {
        InsTempReduceScatterRhdDpuInter tpl;
        TemplateDataParams params = MakeParams(2);
        params.rhdStepCnt = 2;
        params.rhdStepIdx = 2; // rhdStepIdx >= rhdStepCnt
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, MakeSubCommRanks(2)), HCCL_E_PARA);
    }

    TEST(RhdDpuInterGuardTest, StepTableLengthMismatchIsRejected)
    {
        InsTempReduceScatterRhdDpuInter tpl;
        TemplateDataParams params = MakeParams(2);
        // rankSize=2 本地步骤表长度 = 1；rhdStepCnt 填 3 与本地重算表长不符 → HCCL_E_INTERNAL
        params.rhdStepCnt = 3;
        params.rhdStepIdx = 0;
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, MakeSubCommRanks(2)), HCCL_E_INTERNAL);
    }

    // 正路：游标与表长均一致时防御校验全过（AICPU_COMPILE 下传输部分编译掉 → 直接返回 SUCCESS）
    TEST(RhdDpuInterGuardTest, ValidCursorAndTablePasses)
    {
        InsTempReduceScatterRhdDpuInter tpl;
        TemplateDataParams params = MakeParams(2);
        params.rhdStepCnt = 1; // 与本地步骤表长度一致
        params.rhdStepIdx = 0;
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, MakeSubCommRanks(2)), HCCL_SUCCESS);
    }

    // 步骤表构造层（本地步骤表长度，自文档化 mismatch 断言的口径）：
    // rankSize=2 → 1（仅 MAIN）；rankSize=3 → PRE + MAIN + POST = 3；skipPost=true（rankSize=3 代表）→ 去 POST = 2
    TEST(RhdDpuInterGuardTest, FlatStepTableLength)
    {
        std::vector<RhdStepInfo> steps;
        ASSERT_EQ(InsTempReduceScatterRhdDpuInter::BuildRhdFlatSteps(2, 0, false, steps), HCCL_SUCCESS);
        EXPECT_EQ(steps.size(), 1u);

        steps.clear();
        ASSERT_EQ(InsTempReduceScatterRhdDpuInter::BuildRhdFlatSteps(3, 0, false, steps), HCCL_SUCCESS);
        EXPECT_EQ(steps.size(), 3u);

        steps.clear();
        ASSERT_EQ(InsTempReduceScatterRhdDpuInter::BuildRhdFlatSteps(3, 0, true, steps), HCCL_SUCCESS);
        EXPECT_EQ(steps.size(), 2u);
    }

    // 被合并方分支（rankSize=3、myAlgRank=1：inPart1 且非代表）：跳过 MAIN 步，表 = PRE + POST = 2，
    // 首尾相位固定；skipPost=true 时再去 POST → 仅 PRE = 1。isMergedParty 判反/漏判时两侧表长
    // 不一致将由 DPUKernelRun 以 E_INTERNAL 暴露，此处直接钉死被合并方表形
    TEST(RhdDpuInterGuardTest, FlatStepTableMergedParty)
    {
        std::vector<RhdStepInfo> steps;
        ASSERT_EQ(InsTempReduceScatterRhdDpuInter::BuildRhdFlatSteps(3, 1, false, steps), HCCL_SUCCESS);
        EXPECT_EQ(steps.size(), 2u);
        EXPECT_EQ(steps.front().phase, RhdPhase::PRE);
        EXPECT_EQ(steps.back().phase, RhdPhase::POST);

        steps.clear();
        ASSERT_EQ(InsTempReduceScatterRhdDpuInter::BuildRhdFlatSteps(3, 1, true, steps), HCCL_SUCCESS);
        EXPECT_EQ(steps.size(), 1u);
        EXPECT_EQ(steps.front().phase, RhdPhase::PRE);
    }

} // namespace
} // namespace rhd_dpu_inter_ut

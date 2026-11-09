/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 小函数守卫：DPUKernelRun 空入参、CalcScratchMultiple、GetRankFromMap、Describe，
// 以及 KernelRun 的 batch 窗口与错误路径优先级。

#include "nhr_test_access.h"

#include <cstdio>

namespace nhr_ut {
namespace {

    constexpr u32 kRankSize = 2;

    InsTempReduceScatterNhrDpu MakeTemplate()
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(kRankSize));
        return tpl;
    }

    TemplateResource MakeResource()
    {
        TemplateResource resource;
        resource.threads = {1};
        resource.npu2DpuShmemPtr = reinterpret_cast<void*>(0x4000);
        resource.dpu2NpuShmemPtr = reinterpret_cast<void*>(0x5000);
        return resource;
    }

    OpParam MakeOpParam()
    {
        OpParam param;
        std::snprintf(param.algTag, sizeof(param.algTag), "%s", "nhr_ut_tag");
        param.reduceType = HCCL_REDUCE_SUM;
        param.DataDes.dataType = HCCL_DATA_TYPE_FP16;
        param.opConfig.execTimeout = 7;
        return param;
    }

    TEST(NhrGuardTest, DpuKernelRunRejectsEmptyOuterRanks)
    {
        InsTempReduceScatterNhrDpu tpl;
        TemplateDataParams params = MakeStrideParams();
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, {}), HCCL_E_PARA);
        EXPECT_EQ(tpl.templateRankSize_, 0u);
    }

    TEST(NhrGuardTest, DpuKernelRunRejectsEmptyInnerRanks)
    {
        InsTempReduceScatterNhrDpu tpl;
        TemplateDataParams params = MakeStrideParams();
        std::vector<std::vector<u32>> subCommRanks(1);
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 0, subCommRanks), HCCL_E_PARA);
        EXPECT_EQ(tpl.templateRankSize_, 0u);
    }

    TEST(NhrGuardTest, DpuKernelRunRefreshesMembersForValidRanks)
    {
        InsTempReduceScatterNhrDpu tpl;
        TemplateDataParams params = MakeStrideParams();
        const std::vector<std::vector<u32>> subCommRanks = {{0, 2, 4}};
        EXPECT_EQ(tpl.DPUKernelRun(params, {}, 4, subCommRanks), HCCL_SUCCESS);
        EXPECT_EQ(tpl.myRank_, 4u);
        EXPECT_EQ(tpl.templateRankSize_, 3u);
        EXPECT_EQ(tpl.subCommRanks_, subCommRanks);
    }

    TEST(NhrGuardTest, CalcScratchMultipleEqualsRankSize)
    {
        for (u32 rankSize : {1u, 2u, 4u, 8u}) {
            InsTempReduceScatterNhrDpu tpl;
            SetupTemplate(tpl, 0, MakeRankIds(rankSize));
            EXPECT_EQ(tpl.CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT), rankSize);
            EXPECT_EQ(tpl.CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::DEFAULT), rankSize);
        }
    }

    TEST(NhrGuardTest, GetRankFromMapThrowsOnOutOfRange)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        EXPECT_EQ(tpl.GetRankFromMap(0), 0u);
        EXPECT_EQ(tpl.GetRankFromMap(1), 1u);
        EXPECT_THROW(tpl.GetRankFromMap(2), std::out_of_range);
    }

    TEST(NhrGuardTest, DescribeContainsRankSize)
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(3));
        const std::string info = tpl.Describe();
        EXPECT_NE(info.find("tempRankSize 3"), std::string::npos) << info;
    }

    TEST(NhrGuardTest, KernelRunRejectsEmptyThreads)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource;
        ResetStubState();

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_E_INTERNAL);
        EXPECT_TRUE(CopyCalls().empty());
        EXPECT_EQ(BatchModeCounts().startCount, 0u);
    }

    TEST(NhrGuardTest, KernelRunFailsWhenBatchStartFails)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetBatchModeStartResult(HCCL_E_INTERNAL);

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_E_INTERNAL);
        EXPECT_TRUE(CopyCalls().empty());
        EXPECT_EQ(BatchModeCounts().startCount, 1u);
        EXPECT_EQ(BatchModeCounts().endCount, 0u);
    }

    TEST(NhrGuardTest, KernelRunFailsWhenBatchEndFails)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetBatchModeEndResult(HCCL_E_INTERNAL);

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_E_INTERNAL);
        // 预拷贝已完成，但 DPU 未下发
        EXPECT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * params.repeatNum));
        EXPECT_EQ(BatchModeCounts().startCount, 1u);
        EXPECT_EQ(BatchModeCounts().endCount, 1u);
    }

    TEST(NhrGuardTest, KernelRunPropagatesThreadSyncFailure)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetThreadSyncResult(HCCL_E_INTERNAL);

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_E_INTERNAL);
        EXPECT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * params.repeatNum));
        EXPECT_EQ(BatchModeCounts().startCount, 2u);
        EXPECT_EQ(BatchModeCounts().endCount, 1u);
    }

    TEST(NhrGuardTest, KernelRunPropagatesSendRequestFailure)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetSendRequestResult(HCCL_E_INTERNAL);

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_E_INTERNAL);
        EXPECT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * params.repeatNum));
    }

    TEST(NhrGuardTest, KernelRunPropagatesWaitResponseFailure)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetWaitResponseResult(HCCL_E_INTERNAL);

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_E_INTERNAL);
        // 收尾回写未执行
        EXPECT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * params.repeatNum));
    }

    TEST(NhrGuardTest, KernelRunRejectsMismatchedMsgId)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeResource();
        ResetStubState();
        SetSendMsgId(1);
        SetWaitResponseMsgId(2);

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_E_INTERNAL);
        EXPECT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * params.repeatNum));
    }

    TEST(NhrGuardTest, KernelRunHappyPathWritesBackWithBatchWindow)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeResource();
        ResetStubState();

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_SUCCESS);
        // 预拷贝 N*repeatNum + 收尾回写 repeatNum
        EXPECT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * params.repeatNum + params.repeatNum));
        // End 后立即背靠背 Start，收窄 eager 窗口
        EXPECT_EQ(BatchModeCounts().startCount, 2u);
        EXPECT_EQ(BatchModeCounts().endCount, 1u);
    }

} // namespace
} // namespace nhr_ut

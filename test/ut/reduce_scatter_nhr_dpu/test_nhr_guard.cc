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
// 以及 KernelRun 的 batch 窗口、DPU 线程握手时序与错误路径优先级。

#include "nhr_test_access.h"
#include <cstdio>
#include "exec_timeout_manager.h"

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

    TemplateResource MakeDpuThreadResource()
    {
        TemplateResource resource = MakeResource();
        resource.isDpuThreadEnabled = true;
        resource.dpuThread = 42;
        resource.dpuThreadNotifyIdxOnMain = 5;
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

    TEST(NhrGuardTest, KernelRunDpuThreadEnabledHandshakeOrder)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeDpuThreadResource();
        ResetStubState();
        // 取一个与默认值不同的超时，验证 wait 按 ExecTimeoutManager 现取值下发
        constexpr u32 kNotifyTimeout = 4321;
        ExecTimeoutManager::Instance().SetExecTimeout(kNotifyTimeout);

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_SUCCESS);
        // 新路径走线程通知，不再打开共享内存 batch 窗口
        EXPECT_EQ(BatchModeCounts().startCount, 0u);
        EXPECT_EQ(BatchModeCounts().endCount, 0u);
        // 预拷贝 N*repeatNum + 收尾回写 repeatNum
        EXPECT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * params.repeatNum + params.repeatNum));

        const std::vector<NotifyRecordCall>& records = NotifyRecordCalls();
        ASSERT_EQ(records.size(), 2u);
        // 开工通知：主流 → DPU 线程，通知位固定
        EXPECT_EQ(records[0].thread, resource.threads[0]);
        EXPECT_EQ(records[0].dstThread, resource.dpuThread);
        EXPECT_EQ(records[0].notifyIdx, HOST_DPU_THREAD_NOTIFY_IDX);
        // 完工通知：DPU 线程 → 主流，通知位取配置值
        EXPECT_EQ(records[1].thread, resource.dpuThread);
        EXPECT_EQ(records[1].dstThread, resource.threads[0]);
        EXPECT_EQ(records[1].notifyIdx, resource.dpuThreadNotifyIdxOnMain);

        const std::vector<NotifyWaitCall>& waits = NotifyWaitCalls();
        ASSERT_EQ(waits.size(), 2u);
        EXPECT_EQ(waits[0].thread, resource.dpuThread);
        EXPECT_EQ(waits[0].notifyIdx, HOST_DPU_THREAD_NOTIFY_IDX);
        EXPECT_EQ(waits[0].timeOut, kNotifyTimeout);
        EXPECT_EQ(waits[1].thread, resource.threads[0]);
        EXPECT_EQ(waits[1].notifyIdx, resource.dpuThreadNotifyIdxOnMain);
        EXPECT_EQ(waits[1].timeOut, kNotifyTimeout);

        // 恢复默认超时，避免单例状态泄漏到后续用例
        ExecTimeoutManager::Instance().SetExecTimeout(CUSTOM_TIMEOUT);

        // write-reduce 任务必须下发到 DPU 线程
        const std::vector<DpuTransferCall>& transfers = DpuTransferCalls();
        ASSERT_EQ(transfers.size(), static_cast<size_t>(GetNHRStepNum(kRankSize) * params.repeatNum));
        for (const DpuTransferCall& call : transfers) {
            EXPECT_EQ(call.thread, resource.dpuThread);
        }

        // 完整时序：预拷贝 → record → wait → RunNHR → record → wait → 收尾回写
        std::vector<HandshakeEvent> expected;
        expected.insert(expected.end(), kRankSize * params.repeatNum, HandshakeEvent::kLocalCopy);
        expected.push_back(HandshakeEvent::kNotifyRecord);
        expected.push_back(HandshakeEvent::kNotifyWait);
        expected.insert(expected.end(), GetNHRStepNum(kRankSize) * params.repeatNum, HandshakeEvent::kDpuTransfer);
        expected.push_back(HandshakeEvent::kNotifyRecord);
        expected.push_back(HandshakeEvent::kNotifyWait);
        expected.insert(expected.end(), params.repeatNum, HandshakeEvent::kLocalCopy);
        EXPECT_EQ(HandshakeEvents(), expected);
    }

    TEST(NhrGuardTest, RunNhrRepeatIndexFollowsOuterLoop)
    {
        InsTempReduceScatterNhrDpu tpl;
        // 取 4 rank（2 个 NHR step），使「repeat 在外层、step 在内层」的嵌套可从下标序列区分
        SetupTemplate(tpl, 0, MakeRankIds(4));
        TemplateDataParams params = MakeStrideParams();
        ResetStubState();

        EXPECT_EQ(tpl.RunNHR(params, {}, 42, true), HCCL_SUCCESS);
        const u32 nSteps = GetNHRStepNum(4);
        ASSERT_EQ(nSteps, 2u);
        const std::vector<DpuTransferCall>& transfers = DpuTransferCalls();
        ASSERT_EQ(transfers.size(), static_cast<size_t>(nSteps * params.repeatNum));
        for (size_t i = 0; i < transfers.size(); ++i) {
            EXPECT_EQ(transfers[i].repeat, static_cast<u32>(i / nSteps)) << "index " << i;
            EXPECT_EQ(transfers[i].thread, 42u) << "index " << i;
        }
    }

    TEST(NhrGuardTest, KernelRunDpuThreadEnabledSkipsCompletionNotifyOnRunNhrFailure)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeDpuThreadResource();
        ResetStubState();
        SetDpuTransferResult(static_cast<s32>(HCCL_E_INTERNAL));

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_E_INTERNAL);
        // 预拷贝已完成，但收尾回写不执行
        EXPECT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * params.repeatNum));
        // RunNHR 失败后不得补完工通知：只保留开工的 record/wait
        ASSERT_EQ(NotifyRecordCalls().size(), 1u);
        ASSERT_EQ(NotifyWaitCalls().size(), 1u);
        EXPECT_EQ(NotifyRecordCalls()[0].dstThread, resource.dpuThread);
        EXPECT_EQ(NotifyWaitCalls()[0].thread, resource.dpuThread);
    }

    TEST(NhrGuardTest, KernelRunDpuThreadEnabledPropagatesNotifyRecordFailure)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeDpuThreadResource();
        ResetStubState();
        SetNotifyRecordResult(static_cast<s32>(HCCL_E_INTERNAL));

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_E_INTERNAL);
        // 预拷贝先于开工通知完成，此后不得再有 Wait/RunNHR/收尾回写
        EXPECT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * params.repeatNum));
        ASSERT_EQ(NotifyRecordCalls().size(), 1u);
        EXPECT_EQ(NotifyRecordCalls()[0].dstThread, resource.dpuThread);
        EXPECT_TRUE(NotifyWaitCalls().empty());
        EXPECT_TRUE(DpuTransferCalls().empty());
    }

    TEST(NhrGuardTest, KernelRunDpuThreadEnabledPropagatesNotifyWaitFailure)
    {
        InsTempReduceScatterNhrDpu tpl = MakeTemplate();
        const OpParam param = MakeOpParam();
        TemplateDataParams params = MakeStrideParams();
        TemplateResource resource = MakeDpuThreadResource();
        ResetStubState();
        SetNotifyWaitResult(static_cast<s32>(HCCL_E_INTERNAL));

        EXPECT_EQ(tpl.KernelRun(param, params, resource), HCCL_E_INTERNAL);
        // 开工通知已发出，但等不到 DPU 开工，不得下发 write-reduce 或补完工通知
        EXPECT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * params.repeatNum));
        ASSERT_EQ(NotifyRecordCalls().size(), 1u);
        ASSERT_EQ(NotifyWaitCalls().size(), 1u);
        EXPECT_EQ(NotifyWaitCalls()[0].thread, resource.dpuThread);
        EXPECT_TRUE(DpuTransferCalls().empty());
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

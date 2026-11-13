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
#include "barrier_rhd_sync_stub.h"
#include "ins_temp_barrier_rhd_dpu.h"

using namespace ops_hccl;

namespace {
TemplateResource MakeHostResource(bool useDpuThread)
{
    TemplateResource resource;
    resource.threads = {10};
    resource.isDpuThreadEnabled = useDpuThread;
    resource.npu2DpuShmemPtr = useDpuThread ? nullptr : reinterpret_cast<void*>(0x1000);
    resource.dpu2NpuShmemPtr = useDpuThread ? nullptr : reinterpret_cast<void*>(0x2000);
    for (u32 peer : {1U, 2U}) {
        ChannelInfo channel;
        channel.remoteRank = peer;
        channel.locationType = ENDPOINT_LOC_TYPE_HOST;
        resource.channels[peer] = {channel};
    }
    return resource;
}

HcclResult RunFourRankHostBarrier(bool useDpuThread)
{
    OpParam param{};
    InsTempBarrierRhdDpu barrier(param, 0, {{0, 1, 2, 3}});
    TemplateResource resource = MakeHostResource(useDpuThread);
    return barrier.KernelRun(param, TemplateDataParams{}, resource);
}
} // namespace

TEST(BarrierRhdDpu, RejectsDpuThreadResource)
{
    barrier_rhd_sync_ut::Reset();
    EXPECT_EQ(RunFourRankHostBarrier(true), HCCL_E_NOT_SUPPORT);
    EXPECT_TRUE(barrier_rhd_sync_ut::calls.empty());
}

TEST(BarrierRhdDpu, LegacyFlowSendsOnePlanAndRestoresBatchMode)
{
    barrier_rhd_sync_ut::Reset();
    EXPECT_EQ(RunFourRankHostBarrier(false), HCCL_SUCCESS);
    EXPECT_EQ(barrier_rhd_sync_ut::calls, (std::vector<std::string>{"end", "sync", "request", "response", "start"}));
}

TEST(BarrierRhdDpu, ExecutesDpuPlanInStepOrder)
{
    barrier_rhd_sync_ut::Reset();
    OpParam param{};
    InsTempBarrierRhdDpu barrier(param, 0, {{0, 1, 2, 3}});
    TemplateResource resource = MakeHostResource(false);
    EXPECT_EQ(barrier.DPUKernelRun(TemplateDataParams{}, resource.channels, 0, {{0, 1, 2, 3}}), HCCL_SUCCESS);
    EXPECT_EQ(barrier_rhd_sync_ut::calls, (std::vector<std::string>{"dpu:1:0:0", "dpu:2:0:0"}));
}

TEST(BarrierRhdDpu, DpuPlanStopsAfterFailedStep)
{
    barrier_rhd_sync_ut::Reset();
    barrier_rhd_sync_ut::failDpuSendCall = 1;
    OpParam param{};
    InsTempBarrierRhdDpu barrier(param, 0, {{0, 1, 2, 3}});
    TemplateResource resource = MakeHostResource(false);
    EXPECT_EQ(barrier.DPUKernelRun(TemplateDataParams{}, resource.channels, 0, {{0, 1, 2, 3}}), HCCL_E_INTERNAL);
    EXPECT_EQ(barrier_rhd_sync_ut::calls, (std::vector<std::string>{"dpu:1:0:0"}));
}

TEST(BarrierRhdDpu, LegacyFlowRejectsMismatchedResponseAndRestoresBatchMode)
{
    barrier_rhd_sync_ut::Reset();
    barrier_rhd_sync_ut::responseMsgId = barrier_rhd_sync_ut::REQUEST_MSG_ID + 1;
    EXPECT_EQ(RunFourRankHostBarrier(false), HCCL_E_INTERNAL);
    EXPECT_EQ(barrier_rhd_sync_ut::calls, (std::vector<std::string>{"end", "sync", "request", "response", "start"}));
}

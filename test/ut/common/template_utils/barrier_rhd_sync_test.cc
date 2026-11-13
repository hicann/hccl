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
#include "ins_temp_barrier_rhd_aicpu.h"

using namespace ops_hccl;

namespace {
TemplateResource MakeResource()
{
    TemplateResource resource;
    resource.threads.push_back(0);
    for (u32 peer : {1U, 2U}) {
        ChannelInfo channel;
        channel.remoteRank = peer;
        channel.locationType = ENDPOINT_LOC_TYPE_DEVICE;
        resource.channels[peer] = {channel};
    }
    return resource;
}

HcclResult RunFourRankBarrier()
{
    OpParam param{};
    InsTempBarrierRhdAicpu barrier(param, 0, {{0, 1, 2, 3}});
    TemplateResource resource = MakeResource();
    return barrier.KernelRun(param, TemplateDataParams{}, resource);
}
} // namespace

TEST(BarrierRhdSync, FlushesPriorWorkAndSynchronizesEachStep)
{
    barrier_rhd_sync_ut::Reset();
    EXPECT_EQ(RunFourRankBarrier(), HCCL_SUCCESS);
    EXPECT_EQ(
        barrier_rhd_sync_ut::calls,
        (std::vector<std::string>{"end", "sync", "send:1", "sync", "send:2", "sync", "start"}));
}

TEST(BarrierRhdSync, RestoresBatchModeAfterTransferFailure)
{
    barrier_rhd_sync_ut::Reset();
    barrier_rhd_sync_ut::failSendCall = 1;
    EXPECT_EQ(RunFourRankBarrier(), HCCL_E_INTERNAL);
    EXPECT_EQ(barrier_rhd_sync_ut::calls, (std::vector<std::string>{"end", "sync", "send:1", "start"}));
}

TEST(BarrierRhdSync, StopsBeforeNextStepAfterSyncFailure)
{
    barrier_rhd_sync_ut::Reset();
    barrier_rhd_sync_ut::failSyncCall = 2;
    EXPECT_EQ(RunFourRankBarrier(), HCCL_E_INTERNAL);
    EXPECT_EQ(barrier_rhd_sync_ut::calls, (std::vector<std::string>{"end", "sync", "send:1", "sync", "start"}));
}

TEST(BarrierRhdSync, DoesNotRestartBatchModeWhenFlushFails)
{
    barrier_rhd_sync_ut::Reset();
    barrier_rhd_sync_ut::failBatchEnd = true;
    EXPECT_EQ(RunFourRankBarrier(), HCCL_E_INTERNAL);
    EXPECT_EQ(barrier_rhd_sync_ut::calls, (std::vector<std::string>{"end"}));
}

TEST(BarrierRhdSync, ReportsBatchModeRestoreFailure)
{
    barrier_rhd_sync_ut::Reset();
    barrier_rhd_sync_ut::failBatchStart = true;
    EXPECT_EQ(RunFourRankBarrier(), HCCL_E_INTERNAL);
    EXPECT_EQ(
        barrier_rhd_sync_ut::calls,
        (std::vector<std::string>{"end", "sync", "send:1", "sync", "send:2", "sync", "start"}));
}

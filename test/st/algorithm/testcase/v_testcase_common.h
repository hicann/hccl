/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "sim_world.h"
#include "hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include "gtest/gtest.h"
#include <thread>
#include <functional>
#include "alg_env_config.h"
using namespace HcclSim;
using namespace ops_hccl;

static inline u32 AnalyseRankSize(const TopoMeta &topoInfo)
{
    u32 rankSize = 0;
    for (const auto &superPod : topoInfo) {
        for (const auto &podIdx : superPod) {
            rankSize += podIdx.size();
        }
    }
    return rankSize;
}

template<typename DispatchFn, typename VerifyFn>
void RunVMultilevelTest(const TopoMeta &topoInfo, VDataDesTag vDataDes,
    std::function<void()> extraEnvSetup, DispatchFn dispatchFn, VerifyFn verifyFn)
{
    SimWorld::Global()->Init(topoInfo, DevType::DEV_TYPE_950);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    if (extraEnvSetup) { extraEnvSetup(); }

    auto rankSize = AnalyseRankSize(topoInfo);

    u64 totalCount = 0;
    for (u32 rankId = 0; rankId < rankSize; ++rankId) {
        totalCount += vDataDes.counts[rankId];
    }

    std::vector<std::thread> threads;
    for (u32 rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);
            HcclComm comm = nullptr;
            HcclResult ret = HcclCommInitClusterInfo("./ranktable.json", rankId, &comm);
            if (ret != HCCL_SUCCESS) { return ret; }

            ret = dispatchFn(rankId, totalCount, vDataDes, comm, stream);
            if (ret != HCCL_SUCCESS) { return ret; }

            ret = HcclCommDestroy(comm);
            if (ret != HCCL_SUCCESS) { return ret; }
            return HCCL_SUCCESS;
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = verifyFn(taskQueues, rankSize, vDataDes);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    SimWorld::Global()->Deinit();
}
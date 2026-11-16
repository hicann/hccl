/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_TESTCASE_COMMOM_H
#define SCATTER_TESTCASE_COMMOM_H

#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include <thread>
#include "alg_env_config.h"

using namespace HcclSim;
using namespace ops_hccl;

constexpr uint32_t DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED]
    = {sizeof(int8_t),
       sizeof(int16_t),
       sizeof(int32_t),
       2,
       sizeof(float),
       sizeof(int64_t),
       sizeof(uint64_t),
       sizeof(uint8_t),
       sizeof(uint16_t),
       sizeof(uint32_t),
       8,
       2,
       16,
       2,
       1,
       1,
       1,
       1};

// 公共执行 core：Init → 逐 rank 线程（SetDevice/Stream/Comm/Malloc/Scatter/Destroy）→ join →
// CheckScatter → 追加校验（可选）→ Deinit。三个变体仅差 sendbuf 分配策略与追加校验。
// allocInputForAllRanks=false 时非 root 的 sendBuf 为 nullptr（Scatter API：sendbuf 仅 root 有效）。
static void RunScatterTestCore(
    int root, TopoMeta& topoMeta, const u64 dataCount, HcclDataType dataType, bool allocInputForAllRanks,
    const std::function<void(const std::decay_t<decltype(SimTaskQueue::Global()->GetAllRankTaskQueues())>&)>& postCheck
    = nullptr)
{
    SimWorld::Global()->Init(topoMeta, HcclDevType::DEV_TYPE_950);

    auto rankSize = CalRankSize(topoMeta);

    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));
            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
            u64 sendDataSize = dataCount * DATATYPE_SIZE_TABLE[dataType] * rankSize;
            u64 recvDataSize = dataCount * DATATYPE_SIZE_TABLE[dataType];
            if (allocInputForAllRanks || rankId == root) {
                aclrtMalloc(&sendBuf, sendDataSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            }
            aclrtMalloc(&recvBuf, recvDataSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            CHK_RET(HcclScatter(sendBuf, recvBuf, dataCount, dataType, root, comm, stream));
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    // 标准 Scatter 语义校验（收向正确性）
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckScatter(taskQueues, rankSize, dataType, dataCount, root);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    if (postCheck) {
        postCheck(taskQueues);
    }

    SimWorld::Global()->Deinit();
}

void RunScatterTest(int root, TopoMeta& topoMeta, const u64 dataCount, HcclDataType dataType)
{
    RunScatterTestCore(root, topoMeta, dataCount, dataType, true);
}

// Scatter 语义不承诺修改 sendbuf：在标准校验之上追加 input 不可变检查——扫描任务图，
// 任何本地写目标（LOCAL_COPY/LOCAL_REDUCE 的 dst、READ/READ_REDUCE 的 localSlice）落在
// INPUT 池即违规。用于验证 RHD 非 2^N 的 σ 置换暂存不借用户输入。
void RunScatterTestInputImmutable(int root, TopoMeta& topoMeta, const u64 dataCount, HcclDataType dataType)
{
    RunScatterTestCore(root, topoMeta, dataCount, dataType, true, [](const auto& taskQueues) {
        u32 inputWriteCount = 0;
        std::string violations;
        for (const auto& podIt : taskQueues) {
            for (const auto& serIt : podIt.second) {
                for (const auto& phyIt : serIt.second) {
                    for (const auto& queue : phyIt.second) {
                        for (const auto& task : queue) {
                            // 按任务类型取本地写目标 slice（各子类 getter 名不同，无法统一 switch）
                            const DataSlice* dstSlice = nullptr;
                            switch (task->GetType()) {
                                case TaskTypeStub::LOCAL_COPY:
                                    dstSlice = &static_cast<const TaskStubLocalCopy*>(task.get())->GetDstSlice();
                                    break;
                                case TaskTypeStub::LOCAL_REDUCE:
                                    dstSlice = &static_cast<const TaskStubLocalReduce*>(task.get())->GetDstSlice();
                                    break;
                                case TaskTypeStub::READ:
                                    dstSlice = &static_cast<const TaskStubRead*>(task.get())->GetLocalSlice();
                                    break;
                                case TaskTypeStub::READ_REDUCE:
                                    dstSlice = &static_cast<const TaskStubReadReduce*>(task.get())->GetLocalSlice();
                                    break;
                                default:
                                    break;
                            }
                            if (dstSlice != nullptr && dstSlice->GetType() == BufferType::INPUT) {
                                inputWriteCount++;
                                if (violations.size() < 512) {
                                    violations += std::to_string(static_cast<int>(task->GetType()))
                                                  + "->INPUT: " + task->Describe() + "; ";
                                }
                            }
                        }
                    }
                }
            }
        }
        EXPECT_EQ(inputWriteCount, 0u) << "tasks writing into INPUT pool: " << violations;
    });
}

// Scatter API：sendbuf 仅在 root 有效——非 root 传 nullptr，验证算子全链路不依赖非 root
// 的输入指针。
void RunScatterTestNullInputNonRoot(int root, TopoMeta& topoMeta, const u64 dataCount, HcclDataType dataType)
{
    RunScatterTestCore(root, topoMeta, dataCount, dataType, false);
}

#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOSTDPU_INLINE_REDUCE_ST_COMMON_H
#define HOSTDPU_INLINE_REDUCE_ST_COMMON_H

#include <array>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <thread>
#include <vector>

#include "acl/acl_rt.h"
#include "alg_env_config.h"
#include "check_utils.h"
#include "gtest/gtest.h"
#include "hccl.h"
#include "hccl_verifier.h"
#include "hcomm_primitives_dl.h"
#include "op_common/algorithm/template/template_utils.h"
#include "op_common/inc/alg_param.h"
#include "sim_communicator.h"
#include "sim_task.h"
#include "sim_world.h"

// Model CPU fallback reductions are recorded by LocalReduce in hccl_proxy/hccl_stub.cc.

namespace hostdpu_st {
using namespace HcclSim;
using namespace ops_hccl;

struct FlowCase {
    TopoMeta topology{{{0, 1, 2}, {0, 1, 2}}};
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
    u32 root = 4;
    bool legacy = false;
    bool deterministic = false;
    bool capability = true;
    bool expectInline = true;
    u64 cclBufferBytes = 200 * 1024 * 1024;
};

class ScopedEnvironment {
public:
    explicit ScopedEnvironment(bool deterministic)
    {
        for (const char* name :
             {"HCCL_OP_EXPANSION_MODE", "ENABLE_HOSTDPU_FOR_LLT", "HCCL_INDEPENDENT_OP", "HCCL_ALGO",
              "HCCL_DETERMINISTIC", "HCCL_ENABLE_OPEN_AICPU"}) {
            const char* value = std::getenv(name);
            saved_.push_back({name, value == nullptr ? "" : value, value != nullptr});
        }
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        setenv("HCCL_DETERMINISTIC", deterministic ? "true" : "false", 1);
        unsetenv("HCCL_ALGO");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        ResetAlgEnvConfigInitState();
    }
    ~ScopedEnvironment()
    {
        for (const auto& saved : saved_) {
            if (saved.present) {
                setenv(saved.name, saved.value.c_str(), 1);
            } else {
                unsetenv(saved.name);
            }
        }
        ResetAlgEnvConfigInitState();
    }

private:
    struct SavedValue {
        const char* name;
        std::string value;
        bool present;
    };
    std::vector<SavedValue> saved_;
};

class ScopedWriteReduceSupport {
public:
    ScopedWriteReduceSupport()
    {
        original_ = dlopen("libhcomm.so", RTLD_NOW | RTLD_NOLOAD);
        model_ = dlopen("libhccl-model-proxy.so", RTLD_NOW | RTLD_NOLOAD);
        if (original_ != nullptr && model_ != nullptr) {
            HcommPrimitivesDlInit(model_);
        }
    }
    ~ScopedWriteReduceSupport()
    {
        if (original_ != nullptr) {
            HcommPrimitivesDlInit(original_);
            EXPECT_EQ(dlclose(original_), 0);
        }
        if (model_ != nullptr) {
            EXPECT_EQ(dlclose(model_), 0);
        }
    }
    bool Ready() const
    {
        return original_ != nullptr && model_ != nullptr && HcommIsSupportHcommWriteReduceWithNotifyOnThread();
    }

private:
    void* original_ = nullptr;
    void* model_ = nullptr;
};

// Supply the already-aggregated capability. No endpoint query or real capability negotiation belongs to these STs.
class ScopedDomainCapability {
public:
    explicit ScopedDomainCapability(HcclComm comm) : comm_(comm) {}
    ~ScopedDomainCapability()
    {
        if (context_.addr != nullptr) {
            // The shared proxy's HcclEngineCtxDestroy is a no-op; destroy the context through the model.
            auto* model = static_cast<SimCommunicator*>(comm_);
            EXPECT_EQ(model->contextManager_->DestroyCommEngineCtx(&context_), HCCL_SUCCESS);
        }
    }
    HcclResult Set(bool supported)
    {
        void* address = nullptr;
        CHK_RET(HcclEngineCtxCreate(
            comm_, "hostdpu_inline_reduce_cap", CommEngine::COMM_ENGINE_CPU_TS, sizeof(int8_t), &address));
        CHK_PTR_NULL(address);
        context_ = {HCCL_MEM_TYPE_HOST, address, sizeof(int8_t)};
        // InlineReduceCap（与 hostdpu_inline_reduce.cc 枚举逐值锁定）：
        // NEGOTIATION_DEGRADED = 0，SOME_RANK_UNSUPPORTED = 1，ALL_RANKS_SUPPORTED = 2
        *static_cast<int8_t*>(address) = supported ? 2 : 1;
        return HCCL_SUCCESS;
    }

private:
    HcclComm comm_;
    HcclMem context_{};
};

struct RankResources {
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    void* send = nullptr;
    void* recv = nullptr;
    ~RankResources()
    {
        if (comm != nullptr) {
            EXPECT_EQ(HcclCommDestroy(comm), HCCL_SUCCESS);
        }
        if (send != nullptr) {
            EXPECT_EQ(aclrtFree(send), ACL_SUCCESS);
        }
        if (recv != nullptr) {
            EXPECT_EQ(aclrtFree(recv), ACL_SUCCESS);
        }
        if (stream != nullptr) {
            EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
        }
    }
};

const char* TopologyPrefix(HcclCMDType op)
{
    switch (op) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            return "AllReduce_";
        case HcclCMDType::HCCL_CMD_REDUCE:
            return "Reduce_";
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            return "ReduceScatter_";
        default:
            return nullptr;
    }
}

HcclResult TopologyContext(HcclComm comm, HcclCMDType op, const FlowCase& test, std::vector<char>& bytes, bool capture)
{
    const char* prefix = TopologyPrefix(op);
    CHK_PRT_RET(
        prefix == nullptr, HCCL_ERROR("[TopologyContext] unsupported opType[%d].", static_cast<int>(op)), HCCL_E_PARA);
    char name[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, name));
    const std::string tag = std::string(prefix) + name;
    void* context = nullptr;
    if (capture) {
        uint64_t size = 0;
        CHK_RET(HcclEngineCtxGet(comm, tag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &context, &size));
        CHK_PTR_NULL(context);
        bytes.assign(static_cast<char*>(context), static_cast<char*>(context) + size);
        TopoInfoWithNetLayerDetails topology;
        topology.DeSerialize(bytes);
        // Supply the aggregated capability through the topology consumed by both selectors.
        // Endpoint queries and real capability negotiation are outside these model STs.
        topology.allDpuSupportInlineReduce = test.capability;
        if (test.legacy) {
            // Keep real model ranks/channels/slices while reaching the legacy execution entry.
            topology.level0Topo = Level0Shape::CLOS;
        }
        bytes = topology.Serialize();
    } else {
        CHK_RET(HcclEngineCtxCreate(comm, tag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, bytes.size(), &context));
        CHK_PTR_NULL(context);
        std::memcpy(context, bytes.data(), bytes.size());
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteOperator(HcclCMDType op, const FlowCase& test, RankResources& resources)
{
    switch (op) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            return HcclAllReduce(
                resources.send, resources.recv, test.count, test.dataType, test.reduceOp, resources.comm,
                resources.stream);
        case HcclCMDType::HCCL_CMD_REDUCE:
            return HcclReduce(
                resources.send, resources.recv, test.count, test.dataType, test.reduceOp, test.root, resources.comm,
                resources.stream);
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            return HcclReduceScatter(
                resources.send, resources.recv, test.count, test.dataType, test.reduceOp, resources.comm,
                resources.stream);
        default:
            return HCCL_E_PARA;
    }
}

HcclResult
RunRank(HcclCMDType op, const FlowCase& test, u32 rank, u32 rankSize, bool prepareTopology, std::vector<char>& topology)
{
    RankResources resources;
    if (aclrtSetDevice(rank) != ACL_SUCCESS || aclrtCreateStream(&resources.stream) != ACL_SUCCESS) {
        return HCCL_E_RUNTIME;
    }
    CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rank, &resources.comm));
    ScopedDomainCapability capability(resources.comm);
    CHK_RET(capability.Set(prepareTopology ? false : test.capability));
    if (!prepareTopology) {
        CHK_RET(TopologyContext(resources.comm, op, test, topology, false));
    }
    const u64 outputBytes = test.count * ops_hccl::DATATYPE_SIZE_TABLE[test.dataType];
    const u64 inputBytes = op == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ? outputBytes * rankSize : outputBytes;
    if (aclrtMalloc(&resources.send, inputBytes, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK)) != ACL_SUCCESS
        || aclrtMalloc(&resources.recv, outputBytes, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK))
               != ACL_SUCCESS) {
        return HCCL_E_MEMORY;
    }
    CHK_RET(ExecuteOperator(op, test, resources));
    if (prepareTopology) {
        CHK_RET(TopologyContext(resources.comm, op, test, topology, true));
    }
    return HCCL_SUCCESS;
}

void CheckCollective(HcclCMDType op, const FlowCase& test, u32 rankSize)
{
    auto& queues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    switch (op) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            EXPECT_EQ(CheckAllReduce(queues, rankSize, test.dataType, test.count, test.reduceOp), HCCL_SUCCESS);
            break;
        case HcclCMDType::HCCL_CMD_REDUCE:
            EXPECT_EQ(CheckReduce(queues, rankSize, test.dataType, test.count, test.reduceOp, test.root), HCCL_SUCCESS);
            break;
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            EXPECT_EQ(CheckReduceScatter(queues, rankSize, test.dataType, test.count, test.reduceOp), HCCL_SUCCESS);
            break;
        default:
            FAIL() << "Unexpected collective";
    }
    // This is an integration witness that the public API reached the remote DPU reduction path.
    // Primitive arguments, ordering and call counts are covered by wrapper UTs.
    bool remoteWriteReduce = false;
    for (const auto& pod : queues) {
        for (const auto& server : pod.second) {
            for (const auto& rank : server.second) {
                for (const auto& stream : rank.second) {
                    for (const auto& task : stream) {
                        remoteWriteReduce |= task->GetType() == TaskTypeStub::WRITE_REDUCE
                                             && task->GetLinkType() == LinkProtoStub::RDMA;
                    }
                }
            }
        }
    }
    EXPECT_EQ(remoteWriteReduce, test.expectInline);
}

void RunDomain(HcclCMDType op, const FlowCase& test, bool prepareTopology, std::vector<std::vector<char>>& topologies)
{
    struct WorldGuard {
        ~WorldGuard() { SimWorld::Global()->Deinit(); }
    } world;
    SimWorld::Global()->Init(test.topology, HcclDevType::DEV_TYPE_950);
    const u32 rankSize = static_cast<u32>(topologies.size());
    for (u32 rank = 0; rank < rankSize; ++rank) {
        SimWorld::Global()->GetSimNpuByRankId(rank).AllocMemory(::BufferType::CCL, test.cclBufferBytes);
    }
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    ASSERT_EQ(InitEnvConfig(), HCCL_SUCCESS);
    std::vector<HcclResult> results(rankSize, HCCL_E_INTERNAL);
    std::vector<std::string> errors(rankSize);
    std::vector<std::thread> threads;
    for (u32 rank = 0; rank < rankSize; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                results[rank] = RunRank(op, test, rank, rankSize, prepareTopology, topologies[rank]);
            } catch (const std::exception& error) {
                errors[rank] = error.what();
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (u32 rank = 0; rank < rankSize; ++rank) {
        ASSERT_EQ(results[rank], HCCL_SUCCESS) << "rank=" << rank << " " << errors[rank];
    }
    if (!prepareTopology) {
        CheckCollective(op, test, rankSize);
    }
}

void RunCollective(HcclCMDType op, const FlowCase& test)
{
    ScopedEnvironment environment(test.deterministic);
    ScopedWriteReduceSupport symbols;
    ASSERT_TRUE(symbols.Ready());
    u32 rankSize = 0;
    for (const auto& pod : test.topology) {
        for (const auto& server : pod) {
            rankSize += static_cast<u32>(server.size());
        }
    }
    std::vector<std::vector<char>> topologies(rankSize);
    // First capture a genuine model topology using the default unsupported capability.
    // Replay its serialized capability in a fresh domain without reusing warm-up task queues or resources.
    RunDomain(op, test, true, topologies);
    if (testing::Test::HasFailure()) {
        return;
    }
    RunDomain(op, test, false, topologies);
}

} // namespace hostdpu_st
#endif // HOSTDPU_INLINE_REDUCE_ST_COMMON_H

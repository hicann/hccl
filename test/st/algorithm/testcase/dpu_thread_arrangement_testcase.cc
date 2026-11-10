/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * DPU Thread 算法编排 ST。
 *
 * 独立目标：覆盖新 DPU Thread 流程（用例设置 ST_DPU_THREAD_NEW_FLOW 强制走新流程），
 * 并将 HCOMM 版本桩打到 9.2.0.2（ST_HCOMM_VERSION）以满足其他版本相关判断；
 * 未设置这些变量的既有用例行为不受影响。
 *
 * 每条用例校验：RUN_INFO 日志中的 algTag（选路）与 hccl-verifier 的任务队列语义。
 *
 * 注：以下编排未纳入本用例：
 *   - DpuReduce/Broadcast/ScatterPipeLineMeshNHRNHR、DpuBarrierSequenceMeshNHR：仿真/verifier 暂不支持；
 *   - DpuSend/RecvSoleHost：准入要求 HOST→DEVICE 非对称链路（IsHostNicToDeviceNicLink），
 *     仿真拓扑只会生成 HOST/HOST 或 DEVICE/DEVICE 同构链路，无法构造该场景。
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include "alg_env_config.h"
#include "testcase_common.h"

// libhccl-model-proxy 中 log_stub 的日志级别（0x03=ERROR），用例临时下调以打印 RUN_INFO
extern uint32_t logLevel;

namespace {

// CANN_VERSION(9, 2, 0, 2) = 9*10000000 + 2*100000 + 0*1000 - 200 + 2
constexpr const char* ST_HCOMM_VERSION_9_2_0_2 = "90199802";

struct StubEnvSetup {
    StubEnvSetup()
    {
        setenv("ST_HCOMM_VERSION", ST_HCOMM_VERSION_9_2_0_2, 1);
        setenv("ST_DPU_THREAD_NEW_FLOW", "1", 1);
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        ResetAlgEnvConfigInitState();
    }
};
const StubEnvSetup g_stubEnvSetup;

constexpr u64 DATA_COUNT = 1024;
constexpr u32 DATA_TYPE_SIZE = sizeof(float);
constexpr HcclDataType DATA_TYPE = HcclDataType::HCCL_DATA_TYPE_FP32;
constexpr HcclReduceOp REDUCE_OP = HcclReduceOp::HCCL_REDUCE_SUM;

using OpLauncher
    = std::function<HcclResult(u32 rankId, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream)>;
using OpChecker = std::function<HcclResult(HcclSim::AllRankTaskQueues& taskQueues, u32 rankSize)>;

// 从日志中提取第一条 algTag（<op>_<comm>_<algName>_<engine>），仅用于失败信息
std::string FindAlgTagLine(const std::string& logContent)
{
    const size_t pos = logContent.find("algTag[");
    if (pos == std::string::npos) {
        return "";
    }
    const size_t end = logContent.find(']', pos);
    return (end == std::string::npos) ? logContent.substr(pos, 120) : logContent.substr(pos, end - pos + 1);
}

class ST_DPU_THREAD_ARRANGEMENT_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_ALGO");
        ResetAlgEnvConfigInitState();
    }
};

void RunArrangement(
    const TopoMeta& topoMeta, const std::string& hcclAlgo, const std::string& expectedAlgName, u64 sendBufSize,
    u64 recvBufSize, const OpLauncher& launcher, const OpChecker& checker)
{
    const u32 rankSize = HcclSim::CalRankSize(topoMeta);
    ASSERT_GT(rankSize, 0U);

    if (hcclAlgo.empty()) {
        unsetenv("HCCL_ALGO");
    } else {
        setenv("HCCL_ALGO", hcclAlgo.c_str(), 1);
    }
    ResetAlgEnvConfigInitState();

    SimWorld::Global()->Init(topoMeta, HcclDevType::DEV_TYPE_950);

    // 算子执行期间把 stdout 重定向到临时文件，并临时放开 INFO 级日志，用于校验选路 algTag
    const std::string logFile = "./st_dpu_arrangement_stdout.log";
    fflush(stdout);
    const int savedStdout = dup(STDOUT_FILENO);
    FILE* logFp = fopen(logFile.c_str(), "w");
    if (logFp != nullptr && savedStdout >= 0) {
        dup2(fileno(logFp), STDOUT_FILENO);
    }
    logLevel = 0x01;

    std::vector<HcclResult> results(rankSize, HCCL_E_INTERNAL);
    std::vector<std::thread> threads;
    threads.reserve(rankSize);
    for (u32 rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([&, rankId]() {
            aclrtSetDevice(rankId);
            aclrtStream stream = nullptr;
            if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
                results[rankId] = HCCL_E_INTERNAL;
                return;
            }
            HcclComm comm = nullptr;
            results[rankId] = HcclCommInitClusterInfo("./ranktable.json", rankId, &comm);
            if (results[rankId] != HCCL_SUCCESS) {
                return;
            }
            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
            const u64 sendSize = (sendBufSize == 0) ? 1 : sendBufSize;
            const u64 recvSize = (recvBufSize == 0) ? 1 : recvBufSize;
            // 打桩实现，仿真运行需标记内存是 INPUT 和 OUTPUT
            aclrtMalloc(&sendBuf, sendSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            results[rankId] = launcher(rankId, sendBuf, recvBuf, comm, stream);
            HcclCommDestroy(comm);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    fflush(stdout);
    logLevel = 0x03;
    if (savedStdout >= 0) {
        dup2(savedStdout, STDOUT_FILENO);
        close(savedStdout);
    }
    if (logFp != nullptr) {
        fclose(logFp);
    }

    for (u32 rankId = 0; rankId < rankSize; ++rankId) {
        EXPECT_EQ(results[rankId], HCCL_SUCCESS)
            << "rank[" << rankId << "] failed, algorithm[" << expectedAlgName << "], HCCL_ALGO[" << hcclAlgo << "]";
    }

    // 1. 选路校验：HcclGetAlgRes 的 RUN_INFO 记录最终 algTag（<op>_<comm>_<algName>_<engine>）
    std::ifstream logStream(logFile);
    const std::string logContent((std::istreambuf_iterator<char>(logStream)), std::istreambuf_iterator<char>());
    std::remove(logFile.c_str());
    const std::string algKey = "_" + expectedAlgName + "_";
    EXPECT_NE(logContent.find(algKey), std::string::npos)
        << "arrangement[" << expectedAlgName << "] not selected, HCCL_ALGO[" << hcclAlgo << "], first algTag: ["
        << FindAlgTagLine(logContent) << "]";

    // 2. 语义校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    EXPECT_EQ(checker(taskQueues, rankSize), HCCL_SUCCESS)
        << "semantics check failed, algorithm[" << expectedAlgName << "]";

    SimWorld::Global()->Deinit();
    unsetenv("HCCL_ALGO");
    ResetAlgEnvConfigInitState();
}

OpLauncher AllReduceLauncher()
{
    return [](u32, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
        return HcclAllReduce(sendBuf, recvBuf, DATA_COUNT, DATA_TYPE, REDUCE_OP, comm, stream);
    };
}

OpLauncher AllGatherLauncher()
{
    return [](u32, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
        return HcclAllGather(sendBuf, recvBuf, DATA_COUNT, DATA_TYPE, comm, stream);
    };
}

OpLauncher ReduceScatterLauncher()
{
    return [](u32, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
        return HcclReduceScatter(sendBuf, recvBuf, DATA_COUNT, DATA_TYPE, REDUCE_OP, comm, stream);
    };
}

OpLauncher ReduceLauncher()
{
    return [](u32, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
        return HcclReduce(sendBuf, recvBuf, DATA_COUNT, DATA_TYPE, REDUCE_OP, 0, comm, stream);
    };
}

OpLauncher BroadcastLauncher()
{
    return [](u32, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
        return HcclBroadcast(sendBuf, DATA_COUNT, DATA_TYPE, 0, comm, stream);
    };
}

OpLauncher ScatterLauncher()
{
    return [](u32, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
        return HcclScatter(sendBuf, recvBuf, DATA_COUNT, DATA_TYPE, 0, comm, stream);
    };
}

// 三级拓扑：2 超节点 x 2 server x 2 卡（x*y*z 记法：2*2*2）
TopoMeta ThreeLevelTopo()
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    return topoMeta;
}

// 两级拓扑：1 超节点 x 2 server x 2 卡
TopoMeta TwoLevelTopo()
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 2);
    return topoMeta;
}

// 跨 server 拓扑：1 超节点 x 2 server x 每 server 1 卡（rank0 在 server0，rank1 在 server1）
TopoMeta CrossServerTopo()
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 1);
    return topoMeta;
}

} // namespace

// ============================== AllGather ==============================
TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, all_gather_pipeline_mesh_nhr_nhr)
{
    TopoMeta topoMeta = ThreeLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    RunArrangement(
        topoMeta, "allgather:pipeline{mesh,nhr,nhr}", "DpuAllGatherPipeLineMeshNHRNHR", DATA_COUNT * DATA_TYPE_SIZE,
        DATA_COUNT * DATA_TYPE_SIZE * rankSize, AllGatherLauncher(), [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckAllGather(tq, rs, DATA_TYPE, DATA_COUNT);
        });
}

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, all_gather_sequence_mesh_nhr)
{
    TopoMeta topoMeta = TwoLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    RunArrangement(
        topoMeta, "allgather:sequence{mesh,nhr}", "DpuAllGatherSequenceMeshNHR", DATA_COUNT * DATA_TYPE_SIZE,
        DATA_COUNT * DATA_TYPE_SIZE * rankSize, AllGatherLauncher(), [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckAllGather(tq, rs, DATA_TYPE, DATA_COUNT);
        });
}

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, all_gather_sequence_meshconcur_nhr_nhr)
{
    TopoMeta topoMeta = ThreeLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    RunArrangement(
        topoMeta, "allgather:sequence{meshconcur,nhr,nhr}", "DpuAllGatherSequenceMeshConcurNHRNHR",
        DATA_COUNT * DATA_TYPE_SIZE, DATA_COUNT * DATA_TYPE_SIZE * rankSize, AllGatherLauncher(),
        [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckAllGather(tq, rs, DATA_TYPE, DATA_COUNT);
        });
}

// ============================== AllReduce ==============================
TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, all_reduce_pipeline_mesh_nhr_nhr)
{
    TopoMeta topoMeta = ThreeLevelTopo();
    RunArrangement(
        topoMeta, "allreduce:pipeline{mesh,nhr,nhr}", "DpuAllReducePipeLineMeshNHRNHR", DATA_COUNT * DATA_TYPE_SIZE,
        DATA_COUNT * DATA_TYPE_SIZE, AllReduceLauncher(), [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckAllReduce(tq, rs, DATA_TYPE, DATA_COUNT, REDUCE_OP);
        });
}

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, all_reduce_sequence_mesh_nhr_fallback_mesh)
{
    TopoMeta topoMeta = TwoLevelTopo();
    RunArrangement(
        topoMeta, "allreduce:sequence{mesh,nhr}", "DpuAllReduceSequenceMeshMesh", DATA_COUNT * DATA_TYPE_SIZE,
        DATA_COUNT * DATA_TYPE_SIZE, AllReduceLauncher(), [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckAllReduce(tq, rs, DATA_TYPE, DATA_COUNT, REDUCE_OP);
        });
}

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, all_reduce_sequence_meshconcur_nhr_mesh)
{
    TopoMeta topoMeta = ThreeLevelTopo();
    RunArrangement(
        topoMeta, "allreduce:sequence{meshconcur,nhr,mesh}", "DpuAllReduceSequenceMeshConcurNHRMesh",
        DATA_COUNT * DATA_TYPE_SIZE, DATA_COUNT * DATA_TYPE_SIZE, AllReduceLauncher(),
        [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckAllReduce(tq, rs, DATA_TYPE, DATA_COUNT, REDUCE_OP);
        });
}

// ============================== ReduceScatter ==============================
TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, reduce_scatter_pipeline_mesh_nhr_mesh)
{
    TopoMeta topoMeta = ThreeLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    RunArrangement(
        topoMeta, "reducescatter:pipeline{mesh,nhr,mesh}", "DpuReduceScatterPipeLineMeshNHRMesh",
        DATA_COUNT * DATA_TYPE_SIZE * rankSize, DATA_COUNT * DATA_TYPE_SIZE, ReduceScatterLauncher(),
        [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckReduceScatter(tq, rs, DATA_TYPE, DATA_COUNT, REDUCE_OP);
        });
}

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, reduce_scatter_sequence_mesh_mesh)
{
    TopoMeta topoMeta = TwoLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    RunArrangement(
        topoMeta, "reducescatter:sequence{mesh,mesh}", "DpuReduceScatterSequenceMeshMesh",
        DATA_COUNT * DATA_TYPE_SIZE * rankSize, DATA_COUNT * DATA_TYPE_SIZE, ReduceScatterLauncher(),
        [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckReduceScatter(tq, rs, DATA_TYPE, DATA_COUNT, REDUCE_OP);
        });
}

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, reduce_scatter_sequence_meshconcur_nhr_mesh)
{
    TopoMeta topoMeta = ThreeLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    RunArrangement(
        topoMeta, "reducescatter:sequence{meshconcur,nhr,mesh}", "DpuReduceScatterSequenceMeshConcurNHRMesh",
        DATA_COUNT * DATA_TYPE_SIZE * rankSize, DATA_COUNT * DATA_TYPE_SIZE, ReduceScatterLauncher(),
        [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckReduceScatter(tq, rs, DATA_TYPE, DATA_COUNT, REDUCE_OP);
        });
}

// ============================== Reduce ==============================
TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, reduce_sequence_mesh_nhr)
{
    TopoMeta topoMeta = TwoLevelTopo();
    RunArrangement(
        topoMeta, "reduce:sequence{mesh,nhr}", "DpuReduceSequenceMeshMesh", DATA_COUNT * DATA_TYPE_SIZE,
        DATA_COUNT * DATA_TYPE_SIZE, ReduceLauncher(), [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckReduce(tq, rs, DATA_TYPE, DATA_COUNT, REDUCE_OP, 0);
        });
}

// ============================== Broadcast ==============================
TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, broadcast_sequence_mesh_nhr)
{
    TopoMeta topoMeta = TwoLevelTopo();
    RunArrangement(
        topoMeta, "broadcast:sequence{mesh,nhr}", "DpuBroadcastSequenceMeshNHR", DATA_COUNT * DATA_TYPE_SIZE,
        DATA_COUNT * DATA_TYPE_SIZE, BroadcastLauncher(), [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckBroadcast(tq, rs, DATA_TYPE, DATA_COUNT, 0);
        });
}

// ============================== Scatter ==============================
TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, scatter_sequence_mesh_nhr)
{
    TopoMeta topoMeta = TwoLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    RunArrangement(
        topoMeta, "scatter:sequence{mesh,nhr}", "DpuScatterSequenceMeshNHR", DATA_COUNT * DATA_TYPE_SIZE * rankSize,
        DATA_COUNT * DATA_TYPE_SIZE, ScatterLauncher(), [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckScatter(tq, rs, DATA_TYPE, DATA_COUNT, 0);
        });
}

// ============================== AllToAll 系列 ==============================
TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, all_to_all_sole_mesh)
{
    TopoMeta topoMeta = TwoLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    RunArrangement(
        topoMeta, "", "DpuAllToAllSoleMesh", DATA_COUNT * DATA_TYPE_SIZE * rankSize,
        DATA_COUNT * DATA_TYPE_SIZE * rankSize,
        [](u32, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
            return HcclAlltoAll(sendBuf, DATA_COUNT, DATA_TYPE, recvBuf, DATA_COUNT, DATA_TYPE, comm, stream);
        },
        [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckAll2All(tq, rs, DATA_TYPE, DATA_COUNT);
        });
}

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, all_to_all_v_sole_mesh)
{
    TopoMeta topoMeta = TwoLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    std::vector<u64> sendCounts(rankSize, DATA_COUNT);
    std::vector<u64> sdispls(rankSize, 0);
    std::vector<u64> recvCounts(rankSize, DATA_COUNT);
    std::vector<u64> rdispls(rankSize, 0);
    for (u32 i = 0; i < rankSize; ++i) {
        sdispls[i] = i * DATA_COUNT;
        rdispls[i] = i * DATA_COUNT;
    }
    std::vector<u64> sendCountMatrix(rankSize * rankSize, DATA_COUNT);
    RunArrangement(
        topoMeta, "", "DpuAllToAllVSoleMesh", DATA_COUNT * DATA_TYPE_SIZE * rankSize,
        DATA_COUNT * DATA_TYPE_SIZE * rankSize,
        [=](u32, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
            return HcclAlltoAllV(
                sendBuf, sendCounts.data(), sdispls.data(), DATA_TYPE, recvBuf, recvCounts.data(), rdispls.data(),
                DATA_TYPE, comm, stream);
        },
        [=](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckAll2AllV(tq, rs, DATA_TYPE, sendCountMatrix);
        });
}

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, all_to_all_vc_sole_mesh)
{
    TopoMeta topoMeta = TwoLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    std::vector<u64> sendCountMatrix(rankSize * rankSize, DATA_COUNT);
    RunArrangement(
        topoMeta, "", "DpuAllToAllVCSoleMesh", DATA_COUNT * DATA_TYPE_SIZE * rankSize,
        DATA_COUNT * DATA_TYPE_SIZE * rankSize,
        [=](u32, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
            return HcclAlltoAllVC(sendBuf, sendCountMatrix.data(), DATA_TYPE, recvBuf, DATA_TYPE, comm, stream);
        },
        [=](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckAll2AllVC(tq, rs, DATA_TYPE, sendCountMatrix);
        });
}

// ============================== Send / Recv ==============================
// Send/Recv 必须成对下发：仿真建链会等待对端，单边下发会报 "wait channel ready timeout"。
// TwoLevelTopo 下 rank0↔rank1 同 server，channel 落在 L0/DEVICE，走老路径；
// CrossServerTopo 下 rank0↔rank1 跨 server，channel 落在 L1/HOST，覆盖 DPU Thread 新路径。
namespace {
OpLauncher SendRecvLauncher(u32 srcRank, u32 dstRank)
{
    return [srcRank, dstRank](u32 rankId, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
        if (rankId == srcRank) {
            return HcclSend(sendBuf, DATA_COUNT, DATA_TYPE, dstRank, comm, stream);
        }
        if (rankId == dstRank) {
            return HcclRecv(recvBuf, DATA_COUNT, DATA_TYPE, srcRank, comm, stream);
        }
        return HCCL_SUCCESS;
    };
}

OpChecker SendRecvChecker(u32 srcRank, u32 dstRank)
{
    return [srcRank, dstRank](HcclSim::AllRankTaskQueues& tq, u32 rankSize) {
        HcclResult sendRet = CheckSend(tq, rankSize, DATA_TYPE, DATA_COUNT, srcRank, dstRank);
        HcclResult recvRet = CheckRecv(tq, rankSize, DATA_TYPE, DATA_COUNT, srcRank, dstRank);
        return (sendRet == HCCL_SUCCESS && recvRet == HCCL_SUCCESS) ? HCCL_SUCCESS : HCCL_E_INTERNAL;
    };
}
} // namespace

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, send_sole_mesh)
{
    TopoMeta topoMeta = TwoLevelTopo();
    RunArrangement(
        topoMeta, "", "DpuSendSoleMesh", DATA_COUNT * DATA_TYPE_SIZE, DATA_COUNT * DATA_TYPE_SIZE,
        SendRecvLauncher(0, 1), SendRecvChecker(0, 1));
}

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, recv_sole_mesh)
{
    TopoMeta topoMeta = TwoLevelTopo();
    RunArrangement(
        topoMeta, "", "DpuRecvSoleMesh", DATA_COUNT * DATA_TYPE_SIZE, DATA_COUNT * DATA_TYPE_SIZE,
        SendRecvLauncher(0, 1), SendRecvChecker(0, 1));
}

// 跨 server 配对：channel 命中 L1/HOST，进入 isDpuThreadEnabled 的 DPU Thread 新路径
TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, send_sole_mesh_cross_server)
{
    TopoMeta topoMeta = CrossServerTopo();
    RunArrangement(
        topoMeta, "", "DpuSendSoleMesh", DATA_COUNT * DATA_TYPE_SIZE, DATA_COUNT * DATA_TYPE_SIZE,
        SendRecvLauncher(0, 1), SendRecvChecker(0, 1));
}

TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, recv_sole_mesh_cross_server)
{
    TopoMeta topoMeta = CrossServerTopo();
    RunArrangement(
        topoMeta, "", "DpuRecvSoleMesh", DATA_COUNT * DATA_TYPE_SIZE, DATA_COUNT * DATA_TYPE_SIZE,
        SendRecvLauncher(0, 1), SendRecvChecker(0, 1));
}

// ============================== BatchSendRecv ==============================
TEST_F(ST_DPU_THREAD_ARRANGEMENT_TEST, batch_send_recv_sole_mesh)
{
    TopoMeta topoMeta = TwoLevelTopo();
    u32 rankSize = HcclSim::CalRankSize(topoMeta);
    RunArrangement(
        topoMeta, "", "DpuBatchSendRecvSoleMesh", DATA_COUNT * DATA_TYPE_SIZE * rankSize,
        DATA_COUNT * DATA_TYPE_SIZE * rankSize,
        [=](u32, void* sendBuf, void* recvBuf, HcclComm comm, aclrtStream stream) {
            std::vector<HcclSendRecvItem> sendRecvInfo;
            sendRecvInfo.reserve(rankSize * 2);
            for (u32 i = 0; i < rankSize; ++i) {
                void* curSendBuf = static_cast<void*>(static_cast<u8*>(sendBuf) + DATA_COUNT * DATA_TYPE_SIZE * i);
                void* curRecvBuf = static_cast<void*>(static_cast<u8*>(recvBuf) + DATA_COUNT * DATA_TYPE_SIZE * i);
                sendRecvInfo.push_back({HcclSendRecvType::HCCL_SEND, curSendBuf, DATA_COUNT, DATA_TYPE, i});
                sendRecvInfo.push_back({HcclSendRecvType::HCCL_RECV, curRecvBuf, DATA_COUNT, DATA_TYPE, i});
            }
            return HcclBatchSendRecv(sendRecvInfo.data(), sendRecvInfo.size(), comm, stream);
        },
        [](HcclSim::AllRankTaskQueues& tq, u32 rs) {
            return CheckBatchSendRecv(tq, rs, DATA_TYPE, DATA_COUNT);
        });
}

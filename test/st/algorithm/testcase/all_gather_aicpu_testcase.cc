/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include <thread>
#include "alg_env_config.h"
#include "selected_algo_log_stub.h"

using namespace HcclSim;
using namespace ops_hccl;

constexpr uint32_t DATATYPE_SIZE_TABLE_ALL_GATHER_ST[HCCL_DATA_TYPE_RESERVED]
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

class ST_ALL_GATHER_AICPU_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};
u32 AnalyseRankSize(const TopoMeta& topoInfo)
{
    u32 rankSize = 0;
    for (const auto& superPod : topoInfo) {
        for (const auto& podIdx : superPod) {
            rankSize += podIdx.size();
        }
    }
    return rankSize;
}

void RunAllGatherAicpuA5(const TopoMeta& topoInfo, const u64& sendCount, const HcclDataType& dataType)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoInfo, HcclDevType::DEV_TYPE_950);

    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_ALL_GATHER_ST[dataType];
    auto rankSize = AnalyseRankSize(topoInfo);
    // 算子执行参数设置,多线程运行SCATTER算子
    std::vector<std::thread> threads;
    for (auto rankIdx = 0; rankIdx < rankSize; ++rankIdx) {
        threads.emplace_back([=]() {
            // 1.SetDevice
            aclrtSetDevice(rankIdx);
            // 2.创建流
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);
            // 3.初始化通信域
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankIdx, &comm));

            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
            u64 sendBufSize = sendCount * dataTypeSize; // 数据量转化为字节数
            u64 recvBufSize = sendCount * dataTypeSize * rankSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            HCCL_INFO("[ST_ALL_GATHER_AICPU_TEST]Run HcclAllGather");

            // 4.算子下发
            CHK_RET(HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_2rank_int64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_3rank_int64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100; // 1GB单卡数据量，scratch单rank分区为100M（200M/2）, 有24M尾块
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_4rank_int32_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_6rank_int16_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 5}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_8rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_2rank_fp64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP64; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_3rank_fp32_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_4rank_fp16_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_8rank_bfp16_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_6rank_fp8e5m2_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                                 // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_6rank_fp8e8m0_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 1024 * 1024 * 1024; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x2rank_small_data_test)
{
    // 1GB较大数据量（多loop），同时包含尾块，数据宽度8B
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT64; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x3rank_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT32; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x3rank_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 300 * 1024 * 1024; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT16; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_3x3rank_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 300 * 1024 * 1024; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_meshnhr_2x2x4rank_int8_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto sendCount = 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_meshnhr_2x1x8rank_int8_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 8);
    auto sendCount = 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

// asymmetric topology
TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_7server_asymmetric_fp8e8m0_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{
        {{0, 1, 2, 3, 4, 5, 6, 7},
         {0, 1, 2, 3, 4, 5, 6},
         {0, 1, 2, 3, 4, 5},
         {0, 1, 2, 3, 4},
         {0, 1, 2, 3},
         {0, 1, 2},
         {1, 2}}};

    // 算子执行参数设置
    auto sendCount = 100; // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

// ========== RHD 算法强制选路（任务 ag-rhd-sequence-20261104/06，TEST_PLAN §6）==========
// RHD（相邻配对非 2^幂 + 整段合并单 WQE）仅经 HCCL_ALGO 显式启用：
// 二级 allgather:sequence{meshconcur,rhd} → AicpuAllGatherSequenceMeshConcurRHD（TopoMatchTwoLevel，
// L0=server 内 meshconcur、L1=server 间 RHD）；三级 allgather:sequence{meshconcur,rhd,rhd} →
// AicpuAllGatherSequenceMeshConcurRHDRHD（TopoMatchThreeLevel，L1=超节点内 server 间、L2=超节点间，
// 执行序 L2→L1→L0）。topoCustomCheck 恒 false（preferred-only）：未配置时被软过滤排除，
// 默认选路零变化；显式配置时 hcclAlgoPriority 跳过软检查为唯一候选。非幂层内维度是前后处理的
// 目标场景（无幂守卫），2^幂拓扑退化为空前后处理步骤（A-1）。
// 本套件每条用例 RunAllGatherAicpuA5 后经 ExpectAlgoHit 断言选路真实命中（PR#3933 评审
// 21ec67：防"组网/过滤问题导致静默回退默认算法、用例照常绿"——dev-r13 pow2 组网误写实证）。
class ST_ALL_GATHER_AICPU_RHD_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        // 清空选路命中记录（PR#3933 评审 21ec67：防上一用例残留，见 ExpectAlgoHit）
        ResetSelectedAlgoLog();
    }
    void TearDown() override
    {
        unsetenv("HCCL_ALGO");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
    }

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// A-1 二级 RHD 2^幂退化：1 超节点 × 2 server × 2 rank（RHD 域 p=2，σ 恒等=HD 等价路径），
// 前后处理空步骤，totalWqeNum==⌊log₂2⌋==1（域内 p 口径）
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_sequence_2x2rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-2 二级非幂（d1=3，前/后处理生效）：1 超节点 × 3 server × 2 rank=6 rank（RHD 域 p=3、r=1，
// 相邻配对 (0,1)、代表 0/被合并 1/block 2，WQE 角色矩阵 2/1/1）
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_sequence_3x2rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-3 二级非对称非幂（12_2_4 原生形态）：1 超节点 × 2 server 非对称（server 卡数 2/4=6 rank，
// GCD=2→d0=2、d1=3），RHD 域 p=3（WQE 同 A-2 口径）。rev4 F3 裁决（TEST_PLAN §3 A-3 行）：
// TopoMeta 显式 {{0,1},{0,1,2,3}}（server0=2 卡+server1=4 卡，与冻结 TEST_PLAN:57 的 12_2_4 mock
// 语义一致）；禁止 3/3/2 形态——GLOBAL 层 gcd=CalcGcd(instSizeListByLayer)，gcd==1 且
// engine!=HOSTCPU 时 TopoMatchTwoLevel 拒绝（topo_match_two_level.cc:58-62）→ RHD 无法命中、
// 静默回退默认算法，轨道 A 命中断言失效（REVIEW F3 实测）。轨道 A 命中断言强化
// （--expect-algo + 批次日志 grep selected algName 含 RHD）由 Checker 轨道 A 承载（TEST_PLAN rev4）。
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_sequence_2server_asymmetric_int8_small_data_test)
{
    // 仿真模型初始化（12_2_4 mock 的 TopoMeta 等价：非对称 2 server 2/4 卡）
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2, 3}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-12 二级 r≥2 最小例（D-1 证据原型 p=6=DESIGN §3.4 数值例）：1 超节点 × 6 server × 2 rank=12 rank
// （自建 2level_2_6 的 TopoMeta 等价，RHD 域 p=6、r=2，配对 (0,1)(2,3)、角色矩阵 3/1/2）
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_sequence_6x2rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-11 二级 r≥2 缺陷区回归（D-1 核心：p=11 σ 4-环区）：1 超节点 × 11 server × 2 rank=22 rank
// （自建 2level_2_11 的 TopoMeta 等价，RHD 域 p=11、r=3，配对 (0,1)(2,3)(4,5)、角色矩阵 4/1/3）
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_sequence_11x2rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{
        {{0, 1},
         {0, 1},
         {0, 1},
         {0, 1},
         {0, 1},
         {0, 1},
         {0, 1},
         {0, 1},
         {0, 1},
         {0, 1},
         {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-4 三级 RHD 2^幂退化：2 超节点 × 2 server × 2 rank=8 rank（L1 域 d1=2、L2 域 d2=2 均幂，
// L1 的 RPT=d2=2 不计入 WQE 计数，L1/L2 各 totalWqeNum==1；执行序 L2→L1→L0）
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_3level_2x2x2rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-5 三级非幂（d1=3）+ L1 RPT>1（d2=2）：2 超节点 × 3 server × 2 rank=12 rank，
// L1 域 p=3（非幂前后处理）×RPT=2（每 rpt 份独立恒等直写，rpt·ORS 寻址）、L2 域 p=2 幂
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_3level_2x3x2rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 2);
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-13 三级 L1 r≥2（D-1 三级 L1 绑定点 IN≡OUT≡SCR 同网格）：2 超节点 × 6 server × 2 rank=24 rank
// （自建 3level_2_6_2 的 TopoMeta 等价，L1 域 p=6 r=2 角色矩阵 3/1/2、L2 域 p=2 幂）
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_3level_2x6x2rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 6, 2);
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-14 三级 L2 非幂 + d2>d1（rev3 形态 3level_3_2_2 的 TopoMeta 等价）：3 超节点 × 2 server ×
// 2 rank=12 rank，L2 域 p=3 r=1（配对 (0,1)，角色矩阵 2/1/1）+ d2=3>d1=2 跨步 OUT 网格
// （OSS=S·d1）直写几何；L1 域 p=2 幂。L2 r≥2 为登记未验证范围（TEST_PLAN §3.2）
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_3level_3x2x2rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 2);
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-6 dtype 抽样（需求 4.3 全表的 ST 抽样，二级非幂 6 rank 形态；AG 无归约排除）
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_sequence_3x2rank_dtype_table_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd}", 1);

    const std::vector<HcclDataType> dataTypeTable = {
        HcclDataType::HCCL_DATA_TYPE_INT8,   HcclDataType::HCCL_DATA_TYPE_UINT8, HcclDataType::HCCL_DATA_TYPE_INT16,
        HcclDataType::HCCL_DATA_TYPE_INT32,  HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT32,
        HcclDataType::HCCL_DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_FP16,  HcclDataType::HCCL_DATA_TYPE_BFP16,
        HcclDataType::HCCL_DATA_TYPE_FP32,   HcclDataType::HCCL_DATA_TYPE_FP64,
    };

    // 算子执行参数设置
    auto sendCount = 100; // 单卡数据量
    for (const auto& dataType : dataTypeTable) {
        RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
        ExpectAlgoHit("RHD");
    }
}

// A-7 大 count 多 loop（loopTimes>1，三相位与每 loop 重算叠加；repeat（RPT）与 loop 双层独立不混淆）：
// 2level_2_6 形态 12 rank，单卡 600MB（300M 元素 × UINT16）确保超过 maxCountPerLoop
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_sequence_6x2rank_uint16_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 300 * 1024 * 1024;                  // 单卡数据量（多 loop）
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT16; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-8/A-9 图模式幂/非幂（enableRemoteMemAccess_ OUT 直写分支，幂/非幂统一路径）：
// 环境无图模式入口——HCCL_OP_EXPANSION_MODE 合法词表（CCU_MS/CCU_SCHED/AI_CPU/AICPU_TS/AIV/
// HOST/HOST_TS）无 OFFLOAD 图模式值（基线轮已登记，runner 恒置 AI_CPU），按 TEST_PLAN A-8
// 登记口径：同配置默认运行（不判图模式 PASS/FAIL），图模式执行态覆盖为未验证范围，
// 由 TEST_REPORT 记录（直写分支逻辑由源码对照 Review 保证）
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_graphmode_power_2x2rank_int8_test)
{
    // 仿真模型初始化（图模式形态锚点：幂）
    TopoMeta topoMeta{{{0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// A-9 图模式非幂（形态锚点：非幂 d1=3，登记口径同上）
TEST_F(ST_ALL_GATHER_AICPU_RHD_TEST, st_all_gather_a5_aicpu_rhd_graphmode_nonpower_3x2rank_int8_test)
{
    // 仿真模型初始化（图模式形态锚点：非幂）
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    setenv("HCCL_ALGO", "allgather:sequence{meshconcur,rhd}", 1);

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

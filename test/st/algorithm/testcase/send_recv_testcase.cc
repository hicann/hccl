/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_param.h"
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

class ST_SEND_RECV_TEST : public ::testing::Test {
protected:
    void SetUp() override {
        ResetAlgEnvConfigInitState();
    }

    void TearDown() override {
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }

    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {
    }
};

using RankId = uint32_t;

void SendRecvTest(
    TopoMeta &topoMeta, const std::map<RankId, RankId> &sendRecvMap, u32 dataCount, HcclDataType dataType) {
    u32 rankSize = 0;
    for (const auto &superPod: topoMeta)
        for (const auto &server: superPod)
            rankSize += server.size();
    // 仿真模型初始化
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_910_95);

    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);

    // 多线程运行send&recv算子
    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            // 1.SetDevice
            aclrtSetDevice(rankId);

            // 2.创建流
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            // 3.初始化通信域
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            u64 bufSize = dataCount * DATATYPE_SIZE_TABLE[dataType]; // 数据量转化为字节数

            // 4.算子下发
            void *sendBuf = nullptr;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, bufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            void *recvBuf = nullptr;
            aclrtMalloc(&recvBuf, bufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            for (const auto &kv: sendRecvMap) {
                RankId srcRankId = kv.first;
                RankId dstRankId = kv.second;
                if (srcRankId == rankId) {
                    CHK_RET(HcclSend(sendBuf, dataCount, dataType, dstRankId, comm, stream));
                } else if (dstRankId == rankId) {
                    CHK_RET(HcclRecv(recvBuf, dataCount, dataType, srcRankId, comm, stream));
                }
            }

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto &thread: threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    for (const auto &kv: sendRecvMap) {
        RankId srcRankId = kv.first;
        RankId dstRankId = kv.second;
        HcclResult sendRes = CheckSend(taskQueues, rankSize, dataType, dataCount, srcRankId, dstRankId);
        EXPECT_TRUE(sendRes == HCCL_SUCCESS);
        HcclResult recvRes = CheckRecv(taskQueues, rankSize, dataType, dataCount, srcRankId, dstRankId);
        EXPECT_TRUE(recvRes == HCCL_SUCCESS);
    }

    // 资源清理
    SimWorld::Global()->Deinit();
};

// 2卡1机100个int32
TEST_F(ST_SEND_RECV_TEST, st_send_recv_1d_2r_int32_c100_test) {
    // 三维数组指定超节点-Server-Device信息，数字是deviceId
    TopoMeta topoMeta{{{0, 1}}};
    // 键是srcRankId，值是dstRankId，注意rankId和deviceId不一定相等
    // rankId相当于TopoMeta中每一项按顺序的index
    std::map<RankId, RankId> sendRecvMap = {{0, 1}};

    // 算子执行参数设置
    auto dataCount = 100; // 传输数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 2卡1机100个int32，同一通信域多次调用算子用例
TEST_F(ST_SEND_RECV_TEST, st_send_recv_1d_3r_int32_100_repeat_test) {
    TopoMeta topoMeta{{{0, 1}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 1}, {1, 0}, {0, 1}};
    auto dataCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 3卡1机100个int32
TEST_F(ST_SEND_RECV_TEST, st_send_recv_1d_3r_int32_c100_test) {
    TopoMeta topoMeta{{{0, 1, 2}}};
    std::map<RankId, RankId> sendRecvMap = {{2, 0}, {1, 2}};
    auto dataCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 1卡2机100个int32
TEST_F(ST_SEND_RECV_TEST, st_send_recv_2d_1r_int32_c100_test) {
    TopoMeta topoMeta{{{0}, {1}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 1}};
    auto dataCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 1卡3机100个int32
TEST_F(ST_SEND_RECV_TEST, st_send_recv_3d_1r_int32_c100_test) {
    TopoMeta topoMeta{{{0}, {1}, {2}}};
    std::map<RankId, RankId> sendRecvMap = {{2, 0}, {1, 2}};
    auto dataCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 1卡1机2超节点100个int32
TEST_F(ST_SEND_RECV_TEST, st_send_recv_2s_1d_1r_int32_c100_test) {
    TopoMeta topoMeta{{{0}}, {{1}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 1}};
    auto dataCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 1卡1机3超节点100个int32
TEST_F(ST_SEND_RECV_TEST, st_send_recv_3s_1d_1r_int32_c100_test) {
    TopoMeta topoMeta{{{0}}, {{1}}, {{2}}};
    std::map<RankId, RankId> sendRecvMap = {{2, 0}, {0, 1}};
    auto dataCount = 100;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 八卡单机256M个int32
TEST_F(ST_SEND_RECV_TEST, st_send_recv_1d_8r_int32_c256m_test) {
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 6}, {5, 1}, {7, 2}, {3, 4}};
    auto dataCount = 256 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个int32
TEST_F(ST_SEND_RECV_TEST, st_send_recv_2d_4r_int32_c256m_test) {
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};
    auto dataCount = 256 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个UINT8
TEST_F(ST_SEND_RECV_TEST, st_send_recv_2d_4r_uint8_c256m_test) {
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};
    auto dataCount = 256 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个UINT64
TEST_F(ST_SEND_RECV_TEST, st_send_recv_2d_4r_uint64_c256m_test) {
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};
    auto dataCount = 256 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个FP16
TEST_F(ST_SEND_RECV_TEST, st_send_recv_2d_4r_fp16_c256m_test) {
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};
    auto dataCount = 256 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个FP64
TEST_F(ST_SEND_RECV_TEST, st_send_recv_2d_4r_fp64_c256m_test) {
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};
    auto dataCount = 256 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP64;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个HIF8
TEST_F(ST_SEND_RECV_TEST, st_send_recv_2d_4r_hif8_c256m_test) {
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};
    auto dataCount = 256 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个FP8E4M3
TEST_F(ST_SEND_RECV_TEST, st_send_recv_2d_4r_fp8e4m3_c256m_test) {
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};
    auto dataCount = 256 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E4M3;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个FP8E5M2
TEST_F(ST_SEND_RECV_TEST, st_send_recv_2d_4r_fp8e5m2_c256m_test) {
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};
    auto dataCount = 256 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

// 四卡两机256M个FP8E8M0
TEST_F(ST_SEND_RECV_TEST, st_send_recv_s2r4fp8e8m0c256m_test) {
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    std::map<RankId, RankId> sendRecvMap = {{0, 4}, {6, 1}, {2, 3}, {5, 7}};
    auto dataCount = 256 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0;

    SendRecvTest(topoMeta, sendRecvMap, dataCount, dataType);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 对称内存公共层门禁矩阵 UT：覆盖 SRS-SD「质量保障-UT：公共层白名单/input 侧探测策略门禁矩阵」
// 被测对象：RefreshSymmetricMemory / CheckAndSetSymmetricMemory / IsReduceOpSymMemDeterministic（op_common.cc）
// 覆盖点：
//   1. opType 白名单（Scatter 放行 / RSV+MeshChunk 放行 / RSV+SoleMesh 拒绝 / 未知算子拒绝）
//   2. 数据类型门禁（RSV 走 vDataDes.dataType，RS 走 DataDes.dataType；INT64/UINT64/FP64/PROD 拦截）
//   3. in-place / 多级拓扑 / 版本 / 展开模式 / 引擎 / 一级拓扑门禁
//   4. input 侧探测策略（仅 Scatter 跳过 input 侧；RSV 双侧探测）
//   5. 探测失败回退（返回失败 / 空句柄）；symMemChecked 窗口有效性恢复
//   6. IsReduceOpSymMemDeterministic 算法白名单矩阵（SoleNHR 不受级数限制、MeshChunk 单级放行）

#include <gtest/gtest.h>

#include <string>

#include "hcomm_stub.h"
#include "op_common.h"
#include "securec.h"

using namespace ops_hccl;
using namespace ut_stub;

namespace {
constexpr u32 ALG_NAME_LEN = OP_ALG_LENGTH;

class SymMemGateTest : public testing::Test {
protected:
    void SetUp() override
    {
        UtSymWinStubReset();
        param_ = OpParam{};
        topo_ = TopoInfoWithNetLayerDetails{};
        // 默认满足全部基础门禁：版本/OPBASE/AICPU_TS/MESH_1D/单级
        param_.opMode = OpMode::OPBASE;
        param_.engine = CommEngine::COMM_ENGINE_AICPU_TS;
        param_.hcclComm = reinterpret_cast<void*>(0x1);
        param_.inputPtr = reinterpret_cast<void*>(0x1000);
        param_.inputSize = 4096;
        param_.outputPtr = reinterpret_cast<void*>(0x2000);
        param_.outputSize = 512;
        param_.reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
        param_.vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
        param_.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
        topo_.level0Topo = Level0Shape::MESH_1D;
        topo_.topoLevelNums = TOPO_LEVEL_NUM_1;
    }

    // 按 opType 填充默认合法算法名
    void SetOp(HcclCMDType opType)
    {
        param_.opType = opType;
        const char* algName = "AicpuReduceScatterVSoleMeshChunk";
        if (opType == HcclCMDType::HCCL_CMD_SCATTER) {
            algName = "AicpuScatterSoleMesh";
        } else if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
            algName = "AicpuAllReduceSoleNHR";
        } else if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
            algName = "AicpuReduceScatterSoleMeshChunk";
        }
        ASSERT_EQ(strcpy_s(param_.algName, sizeof(param_.algName), algName), EOK);
    }

    OpParam param_;
    TopoInfoWithNetLayerDetails topo_;
};

// ---------------- IsReduceOpSymMemDeterministic 算法白名单矩阵 ----------------

TEST_F(SymMemGateTest, DeterministicSoleNhrNotLimitedByLevelNums)
{
    // SoleNHR 为单模板单棵全 rank 蝶形树，多级仅改变链路层，先于级数检查放行
    topo_.topoLevelNums = 3;
    EXPECT_TRUE(IsReduceOpSymMemDeterministic("AicpuAllReduceSoleNHR", &topo_));
    EXPECT_TRUE(IsReduceOpSymMemDeterministic("AicpuReduceScatterSoleNHR", &topo_));
}

TEST_F(SymMemGateTest, DeterministicWhitelistOnSingleLevel)
{
    EXPECT_TRUE(IsReduceOpSymMemDeterministic("AicpuReduceScatterPipeLineMeshNHR", &topo_));
    EXPECT_TRUE(IsReduceOpSymMemDeterministic("AicpuAllReducePipeLineMeshNHR", &topo_));
    EXPECT_TRUE(IsReduceOpSymMemDeterministic("AicpuAllReduceSoleMeshChunkTwoShot", &topo_));
    EXPECT_TRUE(IsReduceOpSymMemDeterministic("AicpuReduceScatterSoleMeshChunk", &topo_));
    EXPECT_TRUE(IsReduceOpSymMemDeterministic("AicpuReduceScatterVSoleMeshChunk", &topo_));
}

TEST_F(SymMemGateTest, DeterministicRejectsMultiLevelAndUnknown)
{
    topo_.topoLevelNums = 2;
    EXPECT_FALSE(IsReduceOpSymMemDeterministic("AicpuReduceScatterVSoleMeshChunk", &topo_));
    EXPECT_FALSE(IsReduceOpSymMemDeterministic("AicpuReduceScatterSoleMeshChunk", &topo_));
    // 既有 RSV SoleMesh 不在白名单（确定性折叠依赖 CCL 每源一槽，维持拒绝）
    EXPECT_FALSE(IsReduceOpSymMemDeterministic("AicpuReduceScatterVSoleMesh", &topo_));
    EXPECT_FALSE(IsReduceOpSymMemDeterministic("AicpuUnknownAlg", &topo_));
}

// ---------------- RefreshSymmetricMemory：opType 白名单与门禁矩阵 ----------------

TEST_F(SymMemGateTest, ScatterAllowedAndSkipsInputSideProbe)
{
    // Scatter 纯搬运无归约：白名单直接放行；input 侧探测跳过（API 结构原因）
    SetOp(HcclCMDType::HCCL_CMD_SCATTER);
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_TRUE(param_.supportSymmetricMemory);
    // 仅 output 侧被探测：input 窗口不应被赋值，探测次数为 1
    EXPECT_EQ(g_utSymWinGetCalls, 1U);
    EXPECT_EQ(param_.inputSymWindow, nullptr);
    EXPECT_NE(param_.outputSymWindow, nullptr);
}

TEST_F(SymMemGateTest, ScatterAllowedEvenIfInputSideUnregistered)
{
    // Scatter 跳过 input 侧探测：不存在 input 窗口注册，仍应放行（仅 output 探测成功即置位）。
    // 对偶场景见 ReduceScatterVInputProbeFailRejected：RSV input 侧失败则整体回退。
    SetOp(HcclCMDType::HCCL_CMD_SCATTER);
    param_.inputSymWindow = nullptr; // Scatter 无 input 窗口契约
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_TRUE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 1U); // 只查 output
}

TEST_F(SymMemGateTest, ReduceScatterVAllowedWithBothSideProbe)
{
    // RSV：MeshChunk + 单级 + 非64位 + 非PROD + 非 in-place → 放行，双侧探测
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_TRUE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 2U);
    EXPECT_NE(param_.inputSymWindow, nullptr);
    EXPECT_NE(param_.outputSymWindow, nullptr);
}

TEST_F(SymMemGateTest, ReduceScatterVSoleMeshRejected)
{
    // 既有 SoleMesh 不在确定性白名单：不探测、不置位
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    ASSERT_EQ(strcpy_s(param_.algName, sizeof(param_.algName), "AicpuReduceScatterVSoleMesh"), EOK);
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

// RSV 64 位门禁回归：IsReduceOpSymMemAllowed 必须读 vDataDes.dataType（DataDes 对 V 类恒为 RESERVED）
TEST_F(SymMemGateTest, ReduceScatterV64BitTypesRejectedViaVDataDes)
{
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    // DataDes 保持 FP32，仅 vDataDes 切 64 位：仍必须拦截
    for (HcclDataType dataType :
         {HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_FP64}) {
        OpParam p = param_;
        p.vDataDes.dataType = dataType;
        p.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
        EXPECT_EQ(RefreshSymmetricMemory(p, &topo_), HCCL_SUCCESS);
        EXPECT_FALSE(p.supportSymmetricMemory) << "dataType=" << static_cast<int>(dataType);
    }
}

TEST_F(SymMemGateTest, ReduceScatterDataTypeViaDataDes)
{
    // RS（非 V 类）类型门禁走 DataDes.dataType
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    param_.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT64;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
}

TEST_F(SymMemGateTest, ReduceOpsRejectProd)
{
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    param_.reduceType = HcclReduceOp::HCCL_REDUCE_PROD;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
}

TEST_F(SymMemGateTest, ReduceScatterVInplaceRejected)
{
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    param_.outputPtr = param_.inputPtr;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
}

TEST_F(SymMemGateTest, ReduceScatterVMultiLevelRejected)
{
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    topo_.topoLevelNums = 2;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
}

// ---------------- RefreshSymmetricMemory：基础门禁矩阵 ----------------

TEST_F(SymMemGateTest, BaseGateRejectsPcieMixTopology)
{
    // 公共层基础门禁排除 PCIe 组网（level0PcieMix 覆盖混合与纯 PCIe）：
    // 覆盖 CCU→AICPU 回退场景的补探测——入口未探测（symMemChecked=false）时 Refresh 不得开启
    SetOp(HcclCMDType::HCCL_CMD_SCATTER);
    topo_.level0PcieMix = true;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U); // 不补探测（门禁前置否决）

    // RSV 同理（pcieMix 下即使白名单算法也不放行）
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    ASSERT_EQ(strcpy_s(param_.algName, sizeof(param_.algName), "AicpuReduceScatterVSoleMeshChunk"), EOK);
    param_.symMemChecked = false;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

TEST_F(SymMemGateTest, BaseGateVersionModeEngineTopo)
{
    SetOp(HcclCMDType::HCCL_CMD_SCATTER);

    g_utHcommVersion = CANN_VERSION(9, 0, 0);
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);

    g_utHcommVersion = CANN_VERSION(9, 1, 0);
    param_.opMode = OpMode::OFFLOAD;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);

    param_.opMode = OpMode::OPBASE;
    param_.engine = CommEngine::COMM_ENGINE_CCU;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);

    param_.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    topo_.level0Topo = Level0Shape::CLOS;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);

    topo_.level0Topo = Level0Shape::MESH_1D;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_TRUE(param_.supportSymmetricMemory);
}

TEST_F(SymMemGateTest, UnknownOpTypeRejected)
{
    param_.opType = HcclCMDType::HCCL_CMD_INVALID;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);
}

// ---------------- 探测失败回退与 symMemChecked 恢复 ----------------

TEST_F(SymMemGateTest, ProbeFailureFallsBack)
{
    SetOp(HcclCMDType::HCCL_CMD_SCATTER);
    g_utSymWinGetRet = HCCL_E_INTERNAL;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);

    UtSymWinStubReset();
    SetOp(HcclCMDType::HCCL_CMD_SCATTER);
    param_.symMemChecked = false; // 清掉上段失败探测置位的标记，确保本段真正发起探测
    g_utSymWinReturnNull = true;  // 探测"成功"但句柄为空：同样回退
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 1U); // 确认本段真实走了探测路径（空句柄分支被覆盖）
}

TEST_F(SymMemGateTest, ReduceScatterVInputProbeFailRejected)
{
    // RSV 双侧契约：仅 input 侧探测失败（第 1 次），output 侧成功 → 整体回退
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    g_utSymWinFailFirstN = 1;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
}

TEST_F(SymMemGateTest, SymMemCheckedRestoresByWindowValidity)
{
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    // 入口已探测过（symMemChecked）：按窗口有效性恢复开关，不再重复查询
    param_.symMemChecked = true;
    param_.inputSymWindow = reinterpret_cast<void*>(0x11);
    param_.outputSymWindow = reinterpret_cast<void*>(0x22);
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_TRUE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 0U);

    // 任一侧窗口失效：不恢复
    param_.outputSymWindow = nullptr;
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_FALSE(param_.supportSymmetricMemory);
}

TEST_F(SymMemGateTest, SymMemCheckedScatterIgnoresInputWindow)
{
    // Scatter 的 symMemChecked 恢复只看 output 窗口
    SetOp(HcclCMDType::HCCL_CMD_SCATTER);
    param_.symMemChecked = true;
    param_.inputSymWindow = nullptr;
    param_.outputSymWindow = reinterpret_cast<void*>(0x22);
    EXPECT_EQ(RefreshSymmetricMemory(param_, &topo_), HCCL_SUCCESS);
    EXPECT_TRUE(param_.supportSymmetricMemory);
}

// ---------------- CheckAndSetSymmetricMemory：探测侧策略 ----------------

TEST_F(SymMemGateTest, CheckAndSetScatterSkipsInputSide)
{
    SetOp(HcclCMDType::HCCL_CMD_SCATTER);
    CheckAndSetSymmetricMemory(param_);
    EXPECT_TRUE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 1U);
    EXPECT_EQ(param_.inputSymWindow, nullptr);
    EXPECT_NE(param_.outputSymWindow, nullptr);
}

TEST_F(SymMemGateTest, CheckAndSetReduceScatterVProbesBothSides)
{
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    CheckAndSetSymmetricMemory(param_);
    EXPECT_TRUE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 2U);
    EXPECT_NE(param_.inputSymWindow, nullptr);
    EXPECT_NE(param_.outputSymWindow, nullptr);
    EXPECT_TRUE(param_.symMemChecked);
}

TEST_F(SymMemGateTest, CheckAndSetSkipsEmptySide)
{
    // inputSize=0：不查 input 侧；outputSize=0：不查 output 侧（RSV 0 分片回退的探测前提）
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    param_.inputSize = 0;
    CheckAndSetSymmetricMemory(param_);
    EXPECT_TRUE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 1U);

    UtSymWinStubReset();
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    param_.inputSize = 4096;
    param_.outputSize = 0;
    param_.inputSymWindow = nullptr;
    param_.outputSymWindow = nullptr; // 清掉上段残留，验证 output 侧确实未被探测
    CheckAndSetSymmetricMemory(param_);
    EXPECT_TRUE(param_.supportSymmetricMemory);
    EXPECT_EQ(g_utSymWinGetCalls, 1U);
    EXPECT_EQ(param_.outputSymWindow, nullptr);
}

TEST_F(SymMemGateTest, CheckAndSetInputProbeFailKeepsZeroCopyOff)
{
    // RSV input 侧探测失败：整体回退（双侧契约）
    SetOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    g_utSymWinGetRet = HCCL_E_INTERNAL;
    CheckAndSetSymmetricMemory(param_);
    EXPECT_FALSE(param_.supportSymmetricMemory);
    EXPECT_TRUE(param_.symMemChecked);
}
} // namespace

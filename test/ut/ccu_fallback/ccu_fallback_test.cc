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
#include <memory>
#include <string>
#include "ccu_fallback.h"
#include "ccu_fallback_stub.h"
#include "dlhcomm_function.h"
#include "securec.h"

// 白盒源码包含：直调ccu_fallback.cc内的static函数（CompareCcuParam）与内部类型（CheckParamInfo）；
// 因此本目标不再单独编译ccu_fallback.cc（见CMakeLists），include须置于桩头文件之后
#include "../../../src/ops/op_common/ccu_fallback.cc"

using namespace ops_hccl;
using namespace ut_stub;

namespace {
constexpr char UT_ALG_NAME[] = "ut_alg_name";
constexpr char UT_TAG[] = "ut_nego_tag";
constexpr char UT_COMM_NAME[] = "ut_comm";
constexpr u64 UT_DATA_COUNT = 100;

// 构造跨rank协商入参：非send/recv/batch的算子tag不携带SelectAlg_前缀，确保可进入CCU协商链路
void FillNegotiationParam(OpParam& param, HcclCMDType opType)
{
    param = OpParam{};
    param.opType = opType;
    param.opExecuteConfig = OpExecuteConfig::CCU_MS;
    param.DataDes.count = UT_DATA_COUNT;
    param.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    (void)strcpy_s(param.tag, sizeof(param.tag), UT_TAG);
    (void)strcpy_s(param.commName, sizeof(param.commName), UT_COMM_NAME);
}
} // namespace

class CcuFallbackTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetCcuFallbackStub();
        // 注入HcclConfigGetInfo桩：回写可配置的展开模式并计数，模拟CCU展开模式通信域；
        // dlopen(libhcomm.so)成功与否均不影响该注入（DlHcommFunctionInit成功后handle短路，失败则不会重写）
        DlHcommFunction::GetInstance().dlHcclConfigGetInfo
            = [](HcclComm comm, HcclConfigType cfgType, uint32_t infoLen, void* info) -> HcclResult {
            (void)comm;
            (void)cfgType;
            g_ccuFallbackStubCounters.configGetInfoCalled++;
            if (info == nullptr || infoLen < sizeof(HcclOpExpansionMode)) {
                return HCCL_E_PARA;
            }
            *static_cast<HcclOpExpansionMode*>(info) = g_ccuFallbackStubConfig.expansionMode;
            return HCCL_SUCCESS;
        };
    }

    void TearDown() override {}

    // 断言未进入跨rank协商链路：模式查询/rankSize查询/协商缓存/集合通信/子通信域创建/回退重选均未发生
    void AssertNoCrossRankNegotiation() const
    {
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.configGetInfoCalled);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.getRankSizeCalled);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.engineCtxGetCalled);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.engineCtxCreateCalled);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.allGatherCalled);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.allReduceCalled);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.createSubCommCalled);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.submitCalled);
    }
};

// ---- CheckCcuParamAndFallback：BATCH_SEND_RECV在CCU展开模式下整段跳过跨rank协商 ----

// batch在CCU_MS模式通信域下直接放行：不查询展开模式、不发起参数协商AllGather、不做算法回退
TEST_F(CcuFallbackTest, BatchSendRecvSkipsParamCheckInCcuMsMode)
{
    g_ccuFallbackStubConfig.expansionMode = HCCL_OP_EXPANSION_MODE_CCU_MS;
    OpParam param{};
    FillNegotiationParam(param, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
    param.batchSendRecvDataDes.sendRecvItemsPtr = nullptr;
    param.batchSendRecvDataDes.itemNum = 0;

    std::string algName = UT_ALG_NAME;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo;
    HcclResult ret = CheckCcuParamAndFallback(nullptr, param, topoInfo, algName);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    AssertNoCrossRankNegotiation();
    // 参数原样返回：opExecuteConfig/algName/入参tag均未被协商回退改写
    EXPECT_EQ(OpExecuteConfig::CCU_MS, param.opExecuteConfig);
    EXPECT_STREQ(UT_TAG, param.tag);
    EXPECT_STREQ(UT_ALG_NAME, algName.c_str());
    EXPECT_EQ(nullptr, topoInfo.get());
    EXPECT_EQ(0U, g_ccuFallbackStubCounters.reSelectorCalled);
    EXPECT_EQ(0U, g_ccuFallbackStubCounters.setCommEngineCalled);
    EXPECT_EQ(0U, g_ccuFallbackStubCounters.setOpParamAlgTagCalled);
}

// batch在CCU_SCHED模式通信域下行为一致：同样整段跳过
TEST_F(CcuFallbackTest, BatchSendRecvSkipsParamCheckInCcuSchedMode)
{
    g_ccuFallbackStubConfig.expansionMode = HCCL_OP_EXPANSION_MODE_CCU_SCHED;
    OpParam param{};
    FillNegotiationParam(param, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);

    std::string algName = UT_ALG_NAME;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo;
    HcclResult ret = CheckCcuParamAndFallback(nullptr, param, topoInfo, algName);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    AssertNoCrossRankNegotiation();
    EXPECT_EQ(OpExecuteConfig::CCU_MS, param.opExecuteConfig);
    EXPECT_STREQ(UT_ALG_NAME, algName.c_str());
}

// dlHcclConfigGetInfo不可用（dlsym失败）时batch同样跳过：修复前后行为一致，固化该兼容语义
TEST_F(CcuFallbackTest, BatchSendRecvSkipsParamCheckWithoutConfigQuery)
{
    DlHcommFunction::GetInstance().dlHcclConfigGetInfo = nullptr;
    OpParam param{};
    FillNegotiationParam(param, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);

    std::string algName = UT_ALG_NAME;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo;
    HcclResult ret = CheckCcuParamAndFallback(nullptr, param, topoInfo, algName);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    AssertNoCrossRankNegotiation();
    EXPECT_STREQ(UT_ALG_NAME, algName.c_str());
}

// 对照组：集合算子在CCU_MS模式下确实进入参数协商链路（桩在EngineCtxCreate处短路报错），
// 证明若batch跳过缺失，上述batch用例的计数断言必然失败，测试具备回归判别力
TEST_F(CcuFallbackTest, CollectOpEntersParamNegotiationInCcuMsMode)
{
    g_ccuFallbackStubConfig.expansionMode = HCCL_OP_EXPANSION_MODE_CCU_MS;
    OpParam param{};
    FillNegotiationParam(param, HcclCMDType::HCCL_CMD_ALLREDUCE);

    std::string algName = UT_ALG_NAME;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo;
    HcclResult ret = CheckCcuParamAndFallback(nullptr, param, topoInfo, algName);
    EXPECT_NE(HCCL_SUCCESS, ret);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.configGetInfoCalled);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.getRankSizeCalled);
    // 一次缓存查询 + 一次GetNegotiationCtx查询，ctx未命中后创建失败短路
    EXPECT_EQ(2U, g_ccuFallbackStubCounters.engineCtxGetCalled);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.engineCtxCreateCalled);
    EXPECT_EQ(0U, g_ccuFallbackStubCounters.allGatherCalled);
}

// 对照组：非CCU模式（AI_CPU）在展开模式查询后跳过协商，修复前后行为一致
TEST_F(CcuFallbackTest, CollectOpSkipsParamNegotiationInAicpuMode)
{
    g_ccuFallbackStubConfig.expansionMode = HCCL_OP_EXPANSION_MODE_AI_CPU;
    OpParam param{};
    FillNegotiationParam(param, HcclCMDType::HCCL_CMD_ALLREDUCE);

    std::string algName = UT_ALG_NAME;
    std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo;
    HcclResult ret = CheckCcuParamAndFallback(nullptr, param, topoInfo, algName);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.configGetInfoCalled);
    EXPECT_EQ(0U, g_ccuFallbackStubCounters.getRankSizeCalled);
    EXPECT_EQ(0U, g_ccuFallbackStubCounters.allGatherCalled);
}

// 存量行为回归：send/recv仍整段跳过
TEST_F(CcuFallbackTest, SendRecvSkipsParamCheck)
{
    g_ccuFallbackStubConfig.expansionMode = HCCL_OP_EXPANSION_MODE_CCU_MS;
    const HcclCMDType skipOpTypes[] = {HcclCMDType::HCCL_CMD_SEND, HcclCMDType::HCCL_CMD_RECEIVE};
    for (HcclCMDType opType : skipOpTypes) {
        SCOPED_TRACE("opType=" + std::to_string(static_cast<u32>(opType)));
        OpParam param{};
        FillNegotiationParam(param, opType);
        std::string algName = UT_ALG_NAME;
        std::unique_ptr<TopoInfoWithNetLayerDetails> topoInfo;
        HcclResult ret = CheckCcuParamAndFallback(nullptr, param, topoInfo, algName);
        EXPECT_EQ(HCCL_SUCCESS, ret);
        AssertNoCrossRankNegotiation();
        EXPECT_STREQ(UT_ALG_NAME, algName.c_str());
    }
}

// ---- CheckCcuResNegotiation：BATCH_SEND_RECV整段跳过资源协商 ----

// batch直接放行：不查询rankSize、不创建协商子通信域、不发起资源协商AllReduce
TEST_F(CcuFallbackTest, BatchSendRecvSkipsResNegotiation)
{
    g_ccuFallbackStubConfig.expansionMode = HCCL_OP_EXPANSION_MODE_CCU_MS;
    OpParam param{};
    FillNegotiationParam(param, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);

    HcclResult ret = CheckCcuResNegotiation(nullptr, param, true);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    AssertNoCrossRankNegotiation();
    EXPECT_EQ(0U, g_ccuFallbackStubCounters.createSubCommCalled);
}

// 对照组：集合算子确实进入资源协商链路（rankSize=8，桩在EngineCtxCreate处短路报错）
TEST_F(CcuFallbackTest, CollectOpEntersResNegotiation)
{
    OpParam param{};
    FillNegotiationParam(param, HcclCMDType::HCCL_CMD_ALLREDUCE);

    HcclResult ret = CheckCcuResNegotiation(nullptr, param, true);
    EXPECT_NE(HCCL_SUCCESS, ret);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.getRankSizeCalled);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.engineCtxGetCalled);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.engineCtxCreateCalled);
    EXPECT_EQ(0U, g_ccuFallbackStubCounters.allReduceCalled);
}

// 对照组：单rank场景跳过协商，直接以本端资源结论返回
TEST_F(CcuFallbackTest, CollectOpSingleRankSkipsResNegotiation)
{
    g_ccuFallbackStubConfig.rankSize = 1;
    OpParam param{};
    FillNegotiationParam(param, HcclCMDType::HCCL_CMD_ALLREDUCE);

    HcclResult ret = CheckCcuResNegotiation(nullptr, param, true);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.getRankSizeCalled);
    EXPECT_EQ(0U, g_ccuFallbackStubCounters.engineCtxGetCalled);
    EXPECT_EQ(0U, g_ccuFallbackStubCounters.allReduceCalled);
}

// 存量行为回归：send/recv在资源协商中仍整段跳过
TEST_F(CcuFallbackTest, SendRecvSkipsResNegotiation)
{
    const HcclCMDType skipOpTypes[] = {HcclCMDType::HCCL_CMD_SEND, HcclCMDType::HCCL_CMD_RECEIVE};
    for (HcclCMDType opType : skipOpTypes) {
        SCOPED_TRACE("opType=" + std::to_string(static_cast<u32>(opType)));
        OpParam param{};
        FillNegotiationParam(param, opType);
        HcclResult ret = CheckCcuResNegotiation(nullptr, param, false);
        EXPECT_EQ(HCCL_SUCCESS, ret);
        AssertNoCrossRankNegotiation();
    }
}

// ---- CompareCcuParam（白盒直调）：isVariableCountOp豁免名单行为 ----
// batch当前经两处入口提前返回不可达本函数（见上方入口用例），此处直调固化第三处变更：
// batch从isVariableCountOp豁免名单移除后，若因未来新增调用点到达本函数，将参与count比较。

namespace {
// 构造CompareCcuParam的各rank协商数据（对应ExecuteParamCheckOp的AllGather收集结果）
CheckParamInfo MakeCheckParamInfo(uint64_t count, OpExecuteConfig config, HcclDataType dataType, HcclCMDType opType)
{
    return CheckParamInfo{
        count, static_cast<uint32_t>(config), static_cast<uint32_t>(dataType), static_cast<uint32_t>(opType)};
}
} // namespace

// 判别用例：batch不在isVariableCountOp豁免名单，count不一致时参与count比较并报错；
// 修复前（isVariableCountOp含batch）此处整段豁免返回SUCCESS且零调用，本用例必失败。
// 注意：count不一致是FillNegotiationParam人为注入DataDes的合成场景，并非batch真实输入形态：
// 真实batch入参仅填batchSendRecvDataDes（batch_send_recv.cc），而其与DataDes同属一个union
// （alg_param.h），真实形态下DataDes.count实为sendRecvItemsPtr堆地址别名、dataType实为
// itemNum别名，不承载per-rank数据量；本用例仅固化batch不在豁免名单这一代码事实。
TEST_F(CcuFallbackTest, BatchSendRecvUnequalCountIsCompared)
{
    OpParam param{};
    FillNegotiationParam(param, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
    constexpr u32 rankSize = 2;
    const CheckParamInfo recvInfos[rankSize] = {
        MakeCheckParamInfo(
            UT_DATA_COUNT, OpExecuteConfig::CCU_MS, HcclDataType::HCCL_DATA_TYPE_INT32,
            HcclCMDType::HCCL_CMD_BATCH_SEND_RECV),
        MakeCheckParamInfo(
            UT_DATA_COUNT + 1, OpExecuteConfig::CCU_MS, HcclDataType::HCCL_DATA_TYPE_INT32,
            HcclCMDType::HCCL_CMD_BATCH_SEND_RECV),
    };

    OpExecuteConfig lowestConfig = OpExecuteConfig::DEFAULT;
    HcclResult ret = CompareCcuParam(recvInfos, rankSize, param, lowestConfig);
    EXPECT_NE(HCCL_SUCCESS, ret);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.inconsistentCheckOpTypeCalled);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.reportCheckFailedCalled);
}

// 保护剩余豁免名单：AlltoAllV/AlltoAllVC/AllGatherV/ReduceScatterV/HalfAlltoAllV的count不一致
// 仍整段豁免，不触发比较与报错；名单被误删时本用例失败
TEST_F(CcuFallbackTest, VariableCountOpsStillExemptFromCountCompare)
{
    const HcclCMDType variableCountOpTypes[]
        = {HcclCMDType::HCCL_CMD_ALLTOALLV, HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclCMDType::HCCL_CMD_ALLGATHER_V,
           HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, HcclCMDType::HCCL_CMD_HALF_ALLTOALLV};
    for (HcclCMDType opType : variableCountOpTypes) {
        SCOPED_TRACE("opType=" + std::to_string(static_cast<u32>(opType)));
        OpParam param{};
        FillNegotiationParam(param, opType);
        constexpr u32 rankSize = 2;
        const CheckParamInfo recvInfos[rankSize] = {
            MakeCheckParamInfo(UT_DATA_COUNT, OpExecuteConfig::CCU_MS, HcclDataType::HCCL_DATA_TYPE_INT32, opType),
            MakeCheckParamInfo(UT_DATA_COUNT + 1, OpExecuteConfig::CCU_MS, HcclDataType::HCCL_DATA_TYPE_INT32, opType),
        };

        OpExecuteConfig lowestConfig = OpExecuteConfig::DEFAULT;
        HcclResult ret = CompareCcuParam(recvInfos, rankSize, param, lowestConfig);
        EXPECT_EQ(HCCL_SUCCESS, ret);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.inconsistentCheckOpTypeCalled);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.reportCheckFailedCalled);
        EXPECT_EQ(OpExecuteConfig::CCU_MS, lowestConfig);
    }
}

// 基线：非变量数算子（AllReduce）count不一致时比较链路触发，证明比较机制本身工作
TEST_F(CcuFallbackTest, FixedCountOpUnequalCountIsCompared)
{
    OpParam param{};
    FillNegotiationParam(param, HcclCMDType::HCCL_CMD_ALLREDUCE);
    constexpr u32 rankSize = 2;
    const CheckParamInfo recvInfos[rankSize] = {
        MakeCheckParamInfo(
            UT_DATA_COUNT, OpExecuteConfig::CCU_MS, HcclDataType::HCCL_DATA_TYPE_INT32,
            HcclCMDType::HCCL_CMD_ALLREDUCE),
        MakeCheckParamInfo(
            UT_DATA_COUNT + 1, OpExecuteConfig::CCU_MS, HcclDataType::HCCL_DATA_TYPE_INT32,
            HcclCMDType::HCCL_CMD_ALLREDUCE),
    };

    OpExecuteConfig lowestConfig = OpExecuteConfig::DEFAULT;
    HcclResult ret = CompareCcuParam(recvInfos, rankSize, param, lowestConfig);
    EXPECT_NE(HCCL_SUCCESS, ret);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.inconsistentCheckOpTypeCalled);
    EXPECT_EQ(1U, g_ccuFallbackStubCounters.reportCheckFailedCalled);
}

// 全rank参数一致（含batch）：零报告且最低opExecuteConfig正确传播（CCU_SCHED级别低于CCU_MS）
TEST_F(CcuFallbackTest, EqualParamsNoReportAndLowestConfigPropagates)
{
    const HcclCMDType equalOpTypes[] = {HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, HcclCMDType::HCCL_CMD_ALLREDUCE};
    for (HcclCMDType opType : equalOpTypes) {
        SCOPED_TRACE("opType=" + std::to_string(static_cast<u32>(opType)));
        OpParam param{};
        FillNegotiationParam(param, opType);
        constexpr u32 rankSize = 2;
        const CheckParamInfo recvInfos[rankSize] = {
            MakeCheckParamInfo(UT_DATA_COUNT, OpExecuteConfig::CCU_MS, HcclDataType::HCCL_DATA_TYPE_INT32, opType),
            MakeCheckParamInfo(UT_DATA_COUNT, OpExecuteConfig::CCU_SCHED, HcclDataType::HCCL_DATA_TYPE_INT32, opType),
        };

        OpExecuteConfig lowestConfig = OpExecuteConfig::DEFAULT;
        HcclResult ret = CompareCcuParam(recvInfos, rankSize, param, lowestConfig);
        EXPECT_EQ(HCCL_SUCCESS, ret);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.inconsistentCheckOpTypeCalled);
        EXPECT_EQ(0U, g_ccuFallbackStubCounters.reportCheckFailedCalled);
        EXPECT_EQ(OpExecuteConfig::CCU_SCHED, lowestConfig);
    }
}

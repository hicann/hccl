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
 * CCU 协商降级策略层（ccu_fallback）UT：
 * 1. 资源协商（CheckCcuResNegotiation）：协商基础设施失败按"全域资源不足"同语义返回
 *    HCCL_E_UNAVAIL 禁止上抛（环境抖动上抛会放大为算子执行失败）；send/recv/batch 跳过；
 * 2. 参数检视协商降级（CheckCcuParamAndFallback）：协商基础设施失败跳过参数比对、统一降级
 *    AICPU_TS，复用"配置不一致"回退通道（ReSelector + 结果缓存）；协商原语静默假成功
 *    （载荷 magic 校验失败）同语义降级，禁止垃圾参数被当合法协商结果；
 *    CCU only 模式保持拒绝语义（HCCL_E_NOT_SUPPORT 不静默降级）；
 * 3. 既有路径回归：全域一致不回退、存在低配 rank 回退、缓存命中跳过协商/直接应用回退结果、
 *    非 ccu 模式/单 rank/send/recv/SelectAlg_ tag 跳过。
 */
#include <gtest/gtest.h>
#include <string>
#include <utility>

#include "ccu_fallback.h"
#include "op_common.h"
#include "dlhcomm_function.h"
#include "ut_ccu_stub.h"

using ops_hccl::CheckCcuParamAndFallback;
using ops_hccl::CheckCcuResNegotiation;
using ops_hccl::FallbackCtxData;
using ops_hccl::OpExecuteConfig;
using ops_hccl::OpParam;
using ut::g_ccuStub;
using ut::g_negStub;

namespace {
HcclComm FakeComm(uintptr_t id) { return reinterpret_cast<HcclComm>(id); }

// 构造被测参数：默认 ALLREDUCE + CCU_MS + 多 rank 通信域（rankSize/模式由桩配置控制）
OpParam MakeParam(
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE, OpExecuteConfig cfg = OpExecuteConfig::CCU_MS,
    const char* tag = "ut_tag")
{
    OpParam param{};
    (void)strcpy_s(param.tag, sizeof(param.tag), tag);
    (void)strcpy_s(param.commName, sizeof(param.commName), "ut_comm");
    param.opType = opType;
    param.opExecuteConfig = cfg;
    param.DataDes.count = 1024;
    return param;
}

// 参数检视协商调用包装：topoInfo 由 ReSelector 桩消费，被测函数与桩均不解引用，空指针即可
HcclResult CallParamCheck(HcclComm comm, OpParam& param, std::string& algName)
{
    std::unique_ptr<ops_hccl::TopoInfoWithNetLayerDetails> topoInfo;
    return CheckCcuParamAndFallback(comm, param, topoInfo, algName);
}

// 参数协商缓存 key：algName + "_" + 调用时 opExecuteConfig + "_ccuParamCache"（与被测逻辑锁定）
std::string CacheKey(const std::string& algName, OpExecuteConfig cfg)
{
    return algName + "_" + std::to_string(static_cast<uint32_t>(cfg)) + ut::CCU_PARAM_CACHE_SUFFIX;
}
} // namespace

class CcuFallbackTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ut::CcuStubReset();
        // 默认 ccu 模式：IsCommExpansionModeCcu 经单例 dlHcclConfigGetInfo 查询展开模式（桩注入）
        ops_hccl::DlHcommFunction::GetInstance().dlHcclConfigGetInfo
            = [](HcclComm comm, HcclConfigType type, uint32_t len, void* info) -> HcclResult {
            (void)comm;
            (void)type;
            if (info == nullptr || len < sizeof(HcclOpExpansionMode)) {
                return HCCL_E_PARA;
            }
            *static_cast<HcclOpExpansionMode*>(info) = static_cast<HcclOpExpansionMode>(g_ccuStub.expansionMode);
            return HCCL_SUCCESS;
        };
    }
};

/* ==================== 资源协商：CheckCcuResNegotiation ==================== */

TEST_F(CcuFallbackTest, ResNegotiationInfraFailureReturnsUnavail)
{
    // 协商基础设施失败（子通信域建链/原语执行/环境异常）：按"全域资源不足"同语义返回
    // HCCL_E_UNAVAIL 交由上层释放已申请的 CCU 通道并回退，禁止原样上抛
    HcclComm comm = FakeComm(0x2001);
    OpParam param = MakeParam();
    g_negStub.reduceMinRet = HCCL_E_INTERNAL;

    ASSERT_EQ(CheckCcuResNegotiation(comm, param, true), HCCL_E_UNAVAIL);
    EXPECT_EQ(g_negStub.reduceMinCnt, 1);
}

TEST_F(CcuFallbackTest, ResNegotiationAllRanksCapableSuccess)
{
    // 全域资源充足：返回 SUCCESS，本端资源可用性透传给协商原语
    HcclComm comm = FakeComm(0x2002);
    OpParam param = MakeParam();

    ASSERT_EQ(CheckCcuResNegotiation(comm, param, true), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.reduceMinCnt, 1);
    EXPECT_TRUE(g_negStub.lastLocalCapability);
}

TEST_F(CcuFallbackTest, ResNegotiationSomeRankInsufficientUnavail)
{
    // 协商成功但存在资源不足的 rank：与基础设施失败同语义返回 HCCL_E_UNAVAIL
    HcclComm comm = FakeComm(0x2003);
    OpParam param = MakeParam();
    g_negStub.allRanksCapable = false;

    ASSERT_EQ(CheckCcuResNegotiation(comm, param, true), HCCL_E_UNAVAIL);
}

TEST_F(CcuFallbackTest, ResNegotiationSendRecvSkip)
{
    // send/recv/batch_send_recv：不跨 rank 协商直接返回 SUCCESS
    HcclComm comm = FakeComm(0x2004);
    ASSERT_EQ(CheckCcuResNegotiation(comm, MakeParam(HcclCMDType::HCCL_CMD_SEND), true), HCCL_SUCCESS);
    ASSERT_EQ(CheckCcuResNegotiation(comm, MakeParam(HcclCMDType::HCCL_CMD_RECEIVE), true), HCCL_SUCCESS);
    ASSERT_EQ(CheckCcuResNegotiation(comm, MakeParam(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV), true), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.reduceMinCnt, 0);
}

/* ==================== 参数检视协商降级：CheckCcuParamAndFallback ==================== */

TEST_F(CcuFallbackTest, ParamNegotiationInfraFailureFallsBackToAicpu)
{
    // 协商基础设施失败：跳过参数比对统一降级 AICPU_TS，复用配置不一致的 ReSelector 回退通道，
    // 并写入结果缓存（同 key 后续算子直接应用降级结果不再重试）
    HcclComm comm = FakeComm(0x3001);
    OpParam param = MakeParam();
    std::string algName = "ut_alg";
    g_negStub.gatherRet = HCCL_E_INTERNAL;

    ASSERT_EQ(CallParamCheck(comm, param, algName), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.gatherCnt, 1);
    EXPECT_EQ(g_ccuStub.reSelectorCnt, 1);
    EXPECT_EQ(static_cast<uint32_t>(g_ccuStub.lastReSelectorConfig), static_cast<uint32_t>(OpExecuteConfig::AICPU_TS));
    // ReSelector 桩保真首步行为：回退目标配置写入 param
    EXPECT_EQ(static_cast<uint32_t>(param.opExecuteConfig), static_cast<uint32_t>(OpExecuteConfig::AICPU_TS));
    // 回退终态写缓存：key 按调用时 algName/config 生成，缓存值为回退后 algName 与 AICPU_TS
    FallbackCtxData cache{};
    EXPECT_TRUE(ut::CcuStubGetCtx(comm, CacheKey("ut_alg", OpExecuteConfig::CCU_MS), cache));
    EXPECT_EQ(static_cast<uint32_t>(cache.opExecuteConfig), static_cast<uint32_t>(OpExecuteConfig::AICPU_TS));
    EXPECT_STREQ(cache.algName, "ut_alg_fallback");
}

TEST_F(CcuFallbackTest, ParamNegotiationInfraFailureCcuOnlyRejects)
{
    // CCU only 模式（强制 CCU 不静默降级）：协商基础设施失败与配置不一致同语义，
    // 返回 HCCL_E_NOT_SUPPORT 且不写缓存
    HcclComm comm = FakeComm(0x3002);
    OpParam param = MakeParam();
    param.commOpExpansionMode = static_cast<HcclOpExpansionMode>(HCCL_OP_EXPANSION_EXT_CCU_MS_ONLY);
    std::string algName = "ut_alg";
    g_negStub.gatherRet = HCCL_E_INTERNAL;

    ASSERT_EQ(CallParamCheck(comm, param, algName), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(g_ccuStub.reSelectorCnt, 0);
    EXPECT_FALSE(ut::CcuStubHasCtx(comm, CacheKey("ut_alg", OpExecuteConfig::CCU_MS)));
}

TEST_F(CcuFallbackTest, ParamNegotiationConsistentNoFallback)
{
    // 全域配置一致：不回退，缓存最终 algName 与配置供后续算子跳过协商
    HcclComm comm = FakeComm(0x3003);
    OpParam param = MakeParam();
    std::string algName = "ut_alg";
    ut::NegCAddGatherOverride(
        g_negStub, offsetof(ut::CheckParamInfoAbi, opExecuteConfig),
        static_cast<uint32_t>(OpExecuteConfig::CCU_MS)); // 全域 CCU_MS

    ASSERT_EQ(CallParamCheck(comm, param, algName), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.gatherCnt, 1);
    EXPECT_EQ(g_ccuStub.reSelectorCnt, 0);
    EXPECT_EQ(static_cast<uint32_t>(param.opExecuteConfig), static_cast<uint32_t>(OpExecuteConfig::CCU_MS));
    FallbackCtxData cache{};
    EXPECT_TRUE(ut::CcuStubGetCtx(comm, CacheKey("ut_alg", OpExecuteConfig::CCU_MS), cache));
    EXPECT_EQ(static_cast<uint32_t>(cache.opExecuteConfig), static_cast<uint32_t>(OpExecuteConfig::CCU_MS));
    EXPECT_STREQ(cache.algName, "ut_alg");
}

TEST_F(CcuFallbackTest, ParamNegotiationLowestRankConfigFallsBack)
{
    // 存在低配 rank（配置不一致）：取全域最低配置回退，与基础设施失败降级共用 ReSelector 通道
    HcclComm comm = FakeComm(0x3004);
    OpParam param = MakeParam();
    std::string algName = "ut_alg";
    ut::NegCAddGatherOverride(
        g_negStub, offsetof(ut::CheckParamInfoAbi, opExecuteConfig),
        static_cast<uint32_t>(OpExecuteConfig::AICPU_TS)); // 其余 rank 为 AICPU_TS

    ASSERT_EQ(CallParamCheck(comm, param, algName), HCCL_SUCCESS);
    EXPECT_EQ(g_ccuStub.reSelectorCnt, 1);
    EXPECT_EQ(static_cast<uint32_t>(g_ccuStub.lastReSelectorConfig), static_cast<uint32_t>(OpExecuteConfig::AICPU_TS));
    EXPECT_EQ(static_cast<uint32_t>(param.opExecuteConfig), static_cast<uint32_t>(OpExecuteConfig::AICPU_TS));
}

TEST_F(CcuFallbackTest, ParamNegotiationSilentFakeSuccessFallsBackToAicpu)
{
    // 协商原语静默假成功（设备侧任务失败但 host 全链路返回 SUCCESS，收包为协商前清零的
    // 全 0 垃圾）：magic 校验拦截垃圾载荷，按基础设施失败同语义降级 AICPU_TS 并写缓存，
    // 禁止垃圾 opExecuteConfig 被当合法协商结果（变长算子误回退 / CCU only 模式误拒绝）
    HcclComm comm = FakeComm(0x3005);
    OpParam param = MakeParam();
    std::string algName = "ut_alg";
    ut::NegCAddGatherOverride(
        g_negStub, offsetof(ut::CheckParamInfoAbi, opExecuteConfig),
        static_cast<uint32_t>(OpExecuteConfig::CCU_MS));                              // 垃圾载荷伪装全域一致
    ut::NegCAddGatherOverride(g_negStub, offsetof(ut::CheckParamInfoAbi, magic), 0u); // 全 0，magic 必不符

    ASSERT_EQ(CallParamCheck(comm, param, algName), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.gatherCnt, 1);
    EXPECT_EQ(g_ccuStub.reSelectorCnt, 1);
    EXPECT_EQ(static_cast<uint32_t>(g_ccuStub.lastReSelectorConfig), static_cast<uint32_t>(OpExecuteConfig::AICPU_TS));
    EXPECT_EQ(static_cast<uint32_t>(param.opExecuteConfig), static_cast<uint32_t>(OpExecuteConfig::AICPU_TS));
    FallbackCtxData cache{};
    EXPECT_TRUE(ut::CcuStubGetCtx(comm, CacheKey("ut_alg", OpExecuteConfig::CCU_MS), cache));
    EXPECT_EQ(static_cast<uint32_t>(cache.opExecuteConfig), static_cast<uint32_t>(OpExecuteConfig::AICPU_TS));
    EXPECT_STREQ(cache.algName, "ut_alg_fallback");
}

/* ==================== 跳过/缓存路径回归 ==================== */

TEST_F(CcuFallbackTest, ParamSkipSendRecv)
{
    HcclComm comm = FakeComm(0x4001);
    std::string algName = "ut_alg";
    OpParam sendParam = MakeParam(HcclCMDType::HCCL_CMD_SEND);
    OpParam recvParam = MakeParam(HcclCMDType::HCCL_CMD_RECEIVE);
    OpParam batchParam = MakeParam(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
    ASSERT_EQ(CallParamCheck(comm, sendParam, algName), HCCL_SUCCESS);
    ASSERT_EQ(CallParamCheck(comm, recvParam, algName), HCCL_SUCCESS);
    ASSERT_EQ(CallParamCheck(comm, batchParam, algName), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.gatherCnt, 0);
}

TEST_F(CcuFallbackTest, ParamSkipSelectAlgTag)
{
    // aiv sk selector 内部校验调用（SelectAlg_ 前缀 tag）：跳过跨 rank 协商
    HcclComm comm = FakeComm(0x4002);
    OpParam param = MakeParam(HcclCMDType::HCCL_CMD_ALLREDUCE, OpExecuteConfig::CCU_MS, "SelectAlg_xx");
    std::string algName = "ut_alg";

    ASSERT_EQ(CallParamCheck(comm, param, algName), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.gatherCnt, 0);
}

TEST_F(CcuFallbackTest, ParamSkipNonCcuMode)
{
    // 非 ccu 展开模式：0 = HCCL_OP_EXPANSION_MODE_AI_CPU（枚举形态随 CANN 版本不同，参照
    // hccl_host_comm_dl.h 惯例统一按数值引用）
    HcclComm comm = FakeComm(0x4003);
    OpParam param = MakeParam();
    std::string algName = "ut_alg";
    g_ccuStub.expansionMode = 0;

    ASSERT_EQ(CallParamCheck(comm, param, algName), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.gatherCnt, 0);
}

TEST_F(CcuFallbackTest, ParamSkipSingleRank)
{
    // 单 rank 通信域：无跨 rank 协商必要
    HcclComm comm = FakeComm(0x4004);
    OpParam param = MakeParam();
    std::string algName = "ut_alg";
    g_ccuStub.rankSize = 1;

    ASSERT_EQ(CallParamCheck(comm, param, algName), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.gatherCnt, 0);
}

TEST_F(CcuFallbackTest, ParamCacheHitApplyCachedFallback)
{
    // 此前该 key 已回退过（协商失败/不一致）：缓存命中后同 key 后续算子直接应用降级结果，
    // 不再发起协商，SetCommEngine/SetOpParamAlgTag 按既有回退路径设置引擎与算法 tag
    HcclComm comm = FakeComm(0x4005);
    OpParam param = MakeParam();
    std::string algName = "ut_alg";
    ASSERT_EQ(
        ut::CcuStubPresetCtx(
            comm, CacheKey("ut_alg", OpExecuteConfig::CCU_MS), "ut_alg_aicpu", OpExecuteConfig::AICPU_TS),
        HCCL_SUCCESS);

    ASSERT_EQ(CallParamCheck(comm, param, algName), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.gatherCnt, 0);
    EXPECT_EQ(static_cast<uint32_t>(param.opExecuteConfig), static_cast<uint32_t>(OpExecuteConfig::AICPU_TS));
    EXPECT_EQ(algName, "ut_alg_aicpu");
    EXPECT_EQ(g_ccuStub.setCommEngineCnt, 1);
    EXPECT_EQ(g_ccuStub.setAlgTagCnt, 1);
}

TEST_F(CcuFallbackTest, ParamCacheHitNoFallbackNeeded)
{
    // 缓存命中且配置一致（此前协商全域一致）：直接返回 SUCCESS，不重复协商也不改配置
    HcclComm comm = FakeComm(0x4006);
    OpParam param = MakeParam();
    std::string algName = "ut_alg";
    ASSERT_EQ(
        ut::CcuStubPresetCtx(comm, CacheKey("ut_alg", OpExecuteConfig::CCU_MS), "ut_alg", OpExecuteConfig::CCU_MS),
        HCCL_SUCCESS);

    ASSERT_EQ(CallParamCheck(comm, param, algName), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.gatherCnt, 0);
    EXPECT_EQ(static_cast<uint32_t>(param.opExecuteConfig), static_cast<uint32_t>(OpExecuteConfig::CCU_MS));
    EXPECT_EQ(algName, "ut_alg");
    EXPECT_EQ(g_ccuStub.setCommEngineCnt, 0);
}

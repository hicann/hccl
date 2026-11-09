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
 * hostDPU inline reduce 能力协商功能 UT（Part2，与公共协商模块UT解耦）：
 * 1. 门控：P2P算子（子集调用语义）/ 非hostDPU组网 / 非950及后续代际设备 / 协商子通信域（防递归）
 *    / 图模式捕获态（录制态跳过且不写终态缓存），任一命中即跳过
 * 2. per-comm三态能力缓存：命中还原（ALL_RANKS_SUPPORTED支持，
 *    SOME_RANK_UNSUPPORTED/NEGOTIATION_DEGRADED/非法值一律按不支持）
 * 3. 首次协商：本端查询→全域聚合→写终态缓存；协商失败降级不上抛；缓存写失败/并发竞争同名已存在不阻断；
 *    写前二次Get并发收敛：并发拒绝假失败采纳并发者已写入的成功终态，迟到成功覆盖假失败终态
 * 4. dlsym能力查询层：空指针校验 / 符号缺失降级 / 复合符号判定 / 查询失败降级 / 结果透传
 */
#include <gtest/gtest.h>
#include "hostdpu_inline_reduce.h"
#include "hcomm_primitives_dl.h"
#include "ut_ir_stub.h"

using namespace ops_hccl;
using ut::g_irStub;

namespace {
HcclComm FakeComm(uintptr_t id) { return reinterpret_cast<HcclComm>(id); }

// 构造被测 topoInfo：默认 hostDPU组网 + 950 + 已落存host侧代表端点
void SetupTopo(
    TopoInfoWithNetLayerDetails& topo, bool hostDpuOnly = true, HcclDevType devType = HcclDevType::DEV_TYPE_950)
{
    topo = TopoInfoWithNetLayerDetails{};
    topo.hostDpuOnly = hostDpuOnly;
    topo.deviceType = devType;
    EndpointDesc ep{};
    ep.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    topo.hostDpuEndpoint = ep;
}
} // namespace

class HostDpuInlineReduceTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ut::IrStubReset();
        InitDl(); // 默认两个必需符号可用（能力查询 + write-reduce传输）
    }
    // 按当前 dlsymSymbols 集合重放符号探测（控制 INIT_SUPPORT_FLAG 真实流程）
    static void InitDl() { HcommPrimitivesDlInit(reinterpret_cast<void*>(0x1)); }
};

/* ---------------- 门控：三者任一不满足即跳过（不查询/不协商/不建缓存） ---------------- */

TEST_F(HostDpuInlineReduceTest, SkipWhenNotHostDpuOnly)
{
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo, false); // 非 hostDPU 组网
    HcclComm comm = FakeComm(0x1001);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_FALSE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.negotiateCnt, 0);
    EXPECT_EQ(g_irStub.checkFeatureCnt, 0);
    EXPECT_FALSE(ut::IrStubHasCtx(comm, ut::HOSTDPU_INLINE_CAP_TAG));
}

TEST_F(HostDpuInlineReduceTest, SkipWhenNotDev950)
{
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo, true, HcclDevType::DEV_TYPE_910B); // hostDPU组网但非A5(950)及后续代际
    HcclComm comm = FakeComm(0x1002);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_FALSE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.negotiateCnt, 0);
    EXPECT_FALSE(ut::IrStubHasCtx(comm, ut::HOSTDPU_INLINE_CAP_TAG));
}

TEST_F(HostDpuInlineReduceTest, GatePassOnDev960)
{
    // 后续代际（A6/960）同样放行：进入本端查询+全域协商
    HcclComm comm = FakeComm(0x1004);
    g_irStub.checkFeatureValue = true;
    g_irStub.negotiateAllRanksCapable = true;
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo, true, HcclDevType::DEV_TYPE_960);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_TRUE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.checkFeatureCnt, 1);
    EXPECT_EQ(g_irStub.negotiateCnt, 1);
}

TEST_F(HostDpuInlineReduceTest, SkipWhenNegotiationComm)
{
    // 防递归主防线：协商子通信域上的算子（协商AllReduce重入Selector）不得再次进入协商
    g_irStub.isNegotiationComm = true;
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    HcclComm comm = FakeComm(0x1003);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_FALSE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.negotiateCnt, 0);
    EXPECT_FALSE(ut::IrStubHasCtx(comm, ut::HOSTDPU_INLINE_CAP_TAG));
}

TEST_F(HostDpuInlineReduceTest, SkipWhenP2pOpType)
{
    // P2P算子（SEND/RECEIVE/BATCH_SEND_RECV）不参与全rank集合协商：P2P语义仅要求配对rank调用，
    // 仅参与方进入协商会让全域AllReduce等待永不到达的对端（挂死）；且P2P路径不消费能力位
    for (HcclCMDType opType :
         {HcclCMDType::HCCL_CMD_SEND, HcclCMDType::HCCL_CMD_RECEIVE, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV}) {
        HcclComm comm = FakeComm(0x1005);
        TopoInfoWithNetLayerDetails topo;
        SetupTopo(topo);
        ASSERT_EQ(NegotiateHostDpuInlineReduceCapability(comm, opType, nullptr, &topo), HCCL_SUCCESS);
        EXPECT_FALSE(topo.allDpuSupportInlineReduce);
        EXPECT_EQ(g_irStub.checkFeatureCnt, 0);                           // 不做本端查询
        EXPECT_EQ(g_irStub.negotiateCnt, 0);                              // 不做全域协商
        EXPECT_FALSE(ut::IrStubHasCtx(comm, ut::HOSTDPU_INLINE_CAP_TAG)); // 不写缓存
    }
}

TEST_F(HostDpuInlineReduceTest, SkipWhenStreamInCapture)
{
    // 图模式捕获态：缓存miss时跳过协商且不写终态缓存（能力位保持默认false）；
    // 缓存命中读取不受捕获态影响（命中还原先于捕获态门控）
    g_irStub.streamInCapture = true;
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    HcclComm comm = FakeComm(0x1006);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_FALSE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.checkFeatureCnt, 0);
    EXPECT_EQ(g_irStub.negotiateCnt, 0);
    EXPECT_FALSE(ut::IrStubHasCtx(comm, ut::HOSTDPU_INLINE_CAP_TAG)); // 不写终态缓存
    g_irStub.streamInCapture = false;

    // 捕获态下的缓存命中路径：预置SUPPORTED仍正常还原，零查询零协商
    ASSERT_EQ(ut::IrStubPresetCap(comm, ut::CAP_ALL_RANKS_SUPPORTED, sizeof(int8_t)), HCCL_SUCCESS);
    g_irStub.streamInCapture = true;
    TopoInfoWithNetLayerDetails topo2;
    SetupTopo(topo2);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo2), HCCL_SUCCESS);
    EXPECT_TRUE(topo2.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.checkFeatureCnt, 0);
    EXPECT_EQ(g_irStub.negotiateCnt, 0);
    g_irStub.streamInCapture = false;
}

/* ---------------- per-comm 三态能力缓存命中 ---------------- */

TEST_F(HostDpuInlineReduceTest, CacheHitNegotiatedOk)
{
    HcclComm comm = FakeComm(0x2001);
    ASSERT_EQ(ut::IrStubPresetCap(comm, ut::CAP_ALL_RANKS_SUPPORTED, sizeof(int8_t)), HCCL_SUCCESS);
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    // 命中即零开销：不再本端查询、不再协商
    EXPECT_TRUE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.checkFeatureCnt, 0);
    EXPECT_EQ(g_irStub.negotiateCnt, 0);
}

TEST_F(HostDpuInlineReduceTest, CacheHitNegotiatedNo)
{
    HcclComm comm = FakeComm(0x2002);
    ASSERT_EQ(ut::IrStubPresetCap(comm, ut::CAP_SOME_RANK_UNSUPPORTED, sizeof(int8_t)), HCCL_SUCCESS);
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_FALSE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.negotiateCnt, 0);
}

TEST_F(HostDpuInlineReduceTest, CacheHitDegraded)
{
    HcclComm comm = FakeComm(0x2003);
    ASSERT_EQ(ut::IrStubPresetCap(comm, ut::CAP_NEGOTIATION_DEGRADED, sizeof(int8_t)), HCCL_SUCCESS);
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_FALSE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.negotiateCnt, 0);
}

TEST_F(HostDpuInlineReduceTest, CacheHitInvalidCapValueNotSupported)
{
    // 命中但缓存值为非法枚举值（缓存ABI演进/位翻转）：非ALL_RANKS_SUPPORTED一律按不支持，不重新协商
    HcclComm comm = FakeComm(0x2004);
    ASSERT_EQ(ut::IrStubPresetCap(comm, static_cast<int8_t>(99), sizeof(int8_t)), HCCL_SUCCESS);
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_FALSE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.negotiateCnt, 0);
}

/* ---------------- 首次协商：本端查询 → 全域聚合 → 写终态缓存 ---------------- */

TEST_F(HostDpuInlineReduceTest, FirstNegotiationAllSupport)
{
    HcclComm comm = FakeComm(0x3001);
    g_irStub.checkFeatureValue = true;        // 本端支持
    g_irStub.negotiateAllRanksCapable = true; // 全域支持
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_TRUE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.checkFeatureCnt, 1);
    EXPECT_EQ(g_irStub.negotiateCnt, 1);
    EXPECT_TRUE(g_irStub.lastLocalCapability);                   // 本端查询结果透传给协商原语
    EXPECT_EQ(g_irStub.lastEndpointDesc, &topo.hostDpuEndpoint); // 以落存的host侧代表端点查询
    EXPECT_EQ(ut::IrStubGetCapValue(comm, ut::HOSTDPU_INLINE_CAP_TAG), ut::CAP_ALL_RANKS_SUPPORTED);
}

TEST_F(HostDpuInlineReduceTest, FirstNegotiationAnyRankNotSupport)
{
    HcclComm comm = FakeComm(0x3002);
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    // 本端不支持（远端全支持）→ 不支持
    g_irStub.checkFeatureValue = false;
    g_irStub.negotiateAllRanksCapable = false;
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_FALSE(topo.allDpuSupportInlineReduce);
    EXPECT_FALSE(g_irStub.lastLocalCapability);
    EXPECT_EQ(ut::IrStubGetCapValue(comm, ut::HOSTDPU_INLINE_CAP_TAG), ut::CAP_SOME_RANK_UNSUPPORTED);
    // 本端支持，任一远端不支持 → 不支持（换新comm重走协商，避免命中comm缓存致断言空转）
    HcclComm comm2 = FakeComm(0x3006);
    TopoInfoWithNetLayerDetails topo2;
    SetupTopo(topo2);
    g_irStub.checkFeatureValue = true;
    g_irStub.negotiateAllRanksCapable = false;
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm2, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo2), HCCL_SUCCESS);
    EXPECT_FALSE(topo2.allDpuSupportInlineReduce);
    EXPECT_TRUE(g_irStub.lastLocalCapability); // 本端支持已透传协商原语
    EXPECT_EQ(g_irStub.checkFeatureCnt, 2);    // 新comm重新走本端查询（两次调用各一次）
    EXPECT_EQ(g_irStub.negotiateCnt, 2);       // 且真正发起了全域协商（两次调用各一次）
    EXPECT_EQ(ut::IrStubGetCapValue(comm2, ut::HOSTDPU_INLINE_CAP_TAG), ut::CAP_SOME_RANK_UNSUPPORTED);
}

TEST_F(HostDpuInlineReduceTest, NegotiationFailDegrade)
{
    // 协商失败：按不支持降级、写 DEGRADED 终态缓存（避免逐op重试），错误禁止上抛
    HcclComm comm = FakeComm(0x3003);
    g_irStub.negotiateRet = HCCL_E_INTERNAL;
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_FALSE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(ut::IrStubGetCapValue(comm, ut::HOSTDPU_INLINE_CAP_TAG), ut::CAP_NEGOTIATION_DEGRADED);
    // DEGRADED 缓存命中后不再重试协商
    TopoInfoWithNetLayerDetails topo2;
    SetupTopo(topo2);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo2), HCCL_SUCCESS);
    EXPECT_FALSE(topo2.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.negotiateCnt, 1);
}

TEST_F(HostDpuInlineReduceTest, CacheCreateFailTolerant)
{
    // 缓存写失败：不影响本次op（topoInfo字段已直接设置），仅损失缓存收益
    HcclComm comm = FakeComm(0x3004);
    g_irStub.engineCtxCreateFail = true;
    g_irStub.negotiateAllRanksCapable = true;
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_TRUE(topo.allDpuSupportInlineReduce);
    EXPECT_FALSE(ut::IrStubHasCtx(comm, ut::HOSTDPU_INLINE_CAP_TAG));
}

TEST_F(HostDpuInlineReduceTest, CacheCreateConcurrentRaceTolerant)
{
    // 并发首算子竞争：Create撞同名已存在（HCOMM语义返回HCCL_E_PARA）不告警不阻断，
    // topoInfo按本地协商结果设置（协商幂等，与先写线程终态必然一致）
    HcclComm comm = FakeComm(0x3005);
    g_irStub.engineCtxCreateExist = true;
    g_irStub.checkFeatureValue = true;
    g_irStub.negotiateAllRanksCapable = true;
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_TRUE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.negotiateCnt, 1);
}

TEST_F(HostDpuInlineReduceTest, ConcurrentRejectFakeFailureAdoptsPeerSuccess)
{
    // 并发拒绝假失败收敛：同comm并发算子命中缓存miss窗口，本端协商被CommWorker并发拒绝
    // （negotiateRet失败→NEGOTIATION_DEGRADED），写前二次Get时并发算子已写入ALL_RANKS_SUPPORTED：
    // 采纳并发成功结果（能力为comm级静态属性且AllReduce-MIN全域一致，正确协商结果必然相同），
    // 本op恢复启用且不重建缓存（Create不被调用，保留并发者写入的成功终态）
    HcclComm comm = FakeComm(0x3007);
    g_irStub.negotiateRet = HCCL_E_INTERNAL; // 并发拒绝/协商失败
    g_irStub.injectCapOnSecondGet = true;    // 写前二次Get注入并发成功条目
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_EQ(g_irStub.negotiateCnt, 1);         // 确实发起了协商（非缓存命中路径）
    EXPECT_EQ(g_irStub.engineCtxGetCnt, 2);      // 首轮miss + 写前二次Get
    EXPECT_EQ(g_irStub.engineCtxCreateCnt, 0);   // 采纳路径不重建缓存
    EXPECT_TRUE(topo.allDpuSupportInlineReduce); // 采纳并发成功，不再降级
    EXPECT_EQ(ut::IrStubGetCapValue(comm, ut::HOSTDPU_INLINE_CAP_TAG), ut::CAP_ALL_RANKS_SUPPORTED);
}

TEST_F(HostDpuInlineReduceTest, LateSuccessOverwritesPeerDegraded)
{
    // 迟到成功覆盖假失败终态：并发算子先写入NEGOTIATION_DEGRADED（并发拒绝假象），
    // 本端协商随后成功——写前二次Get发现旧条目，销毁后覆盖为ALL_RANKS_SUPPORTED，
    // 防止假失败终态固化导致该comm生命周期内inline reduce永久降级
    HcclComm comm = FakeComm(0x3008);
    g_irStub.checkFeatureValue = true;
    g_irStub.negotiateAllRanksCapable = true; // 本端协商成功
    g_irStub.injectCapOnSecondGet = true;
    g_irStub.injectCapValue = ut::CAP_NEGOTIATION_DEGRADED; // 并发者先写入的假失败终态
    TopoInfoWithNetLayerDetails topo;
    SetupTopo(topo);
    ASSERT_EQ(
        NegotiateHostDpuInlineReduceCapability(comm, HcclCMDType::HCCL_CMD_ALLREDUCE, nullptr, &topo), HCCL_SUCCESS);
    EXPECT_TRUE(topo.allDpuSupportInlineReduce);
    EXPECT_EQ(g_irStub.engineCtxCreateCnt, 1); // 覆盖写入
    EXPECT_EQ(ut::IrStubGetCapValue(comm, ut::HOSTDPU_INLINE_CAP_TAG), ut::CAP_ALL_RANKS_SUPPORTED);
}

/* ---------------- dlsym 能力查询层（HcclCheckDpuInlineReduceLocal / 复合符号判定） ---------------- */

TEST_F(HostDpuInlineReduceTest, CheckLocalNullOutParam)
{
    EndpointDesc ep{};
    ep.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    EXPECT_EQ(HcclCheckDpuInlineReduceLocal(ep, nullptr), HCCL_E_PARA);
}

TEST_F(HostDpuInlineReduceTest, CheckLocalSymbolMissingDegrade)
{
    // 老版本HCOMM：能力查询符号缺失 → 复合判定不支持，按不支持降级且不调用查询
    g_irStub.dlsymSymbols = {};
    InitDl();
    EXPECT_FALSE(IsHcommDpuInlineReduceSupported());
    EndpointDesc ep{};
    ep.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    bool supported = true;
    ASSERT_EQ(HcclCheckDpuInlineReduceLocal(ep, &supported), HCCL_SUCCESS);
    EXPECT_FALSE(supported);
    EXPECT_EQ(g_irStub.checkFeatureCnt, 0);
}

TEST_F(HostDpuInlineReduceTest, CheckLocalWriteReduceSymbolMissingDegrade)
{
    // 复合判定：仅能力查询符号可用、write-reduce传输符号缺失 → 仍按不支持处理
    g_irStub.dlsymSymbols = {"HcommEndpointCheckFeature"};
    InitDl();
    EXPECT_FALSE(IsHcommDpuInlineReduceSupported());
    EndpointDesc ep{};
    ep.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    bool supported = true;
    ASSERT_EQ(HcclCheckDpuInlineReduceLocal(ep, &supported), HCCL_SUCCESS);
    EXPECT_FALSE(supported);
}

TEST_F(HostDpuInlineReduceTest, CheckLocalQueryFailDegrade)
{
    // 查询失败（老HCOMM配新HCCL/未知featureType）：显式按不支持降级，禁止上抛
    g_irStub.checkFeatureRet = HCCL_E_NOT_SUPPORT;
    EndpointDesc ep{};
    ep.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    bool supported = true;
    ASSERT_EQ(HcclCheckDpuInlineReduceLocal(ep, &supported), HCCL_SUCCESS);
    EXPECT_FALSE(supported);
}

TEST_F(HostDpuInlineReduceTest, CheckLocalPassThrough)
{
    // 查询成功：结果透传，且featureType/端点按约定传给HCOMM
    EndpointDesc ep{};
    ep.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    bool supported = false;
    g_irStub.checkFeatureValue = true;
    ASSERT_EQ(HcclCheckDpuInlineReduceLocal(ep, &supported), HCCL_SUCCESS);
    EXPECT_TRUE(supported);
    EXPECT_EQ(g_irStub.lastFeatureType, static_cast<int32_t>(HCCL_HCOMM_FEATURE_DPU_INLINE_REDUCE));
    EXPECT_EQ(g_irStub.lastEndpointDesc, &ep);

    g_irStub.checkFeatureValue = false;
    ASSERT_EQ(HcclCheckDpuInlineReduceLocal(ep, &supported), HCCL_SUCCESS);
    EXPECT_FALSE(supported);
}

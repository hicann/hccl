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
 * HcclCreateSubCommConfig 弱符号缺失降级 UT（模拟老版本 HCOMM 未提供该接口）：
 * 本 target 链接时桩不定义该符号（-DUT_NEG_NO_SUBCOMM_SYMBOL），SUT 的弱引用在链接期
 * 解析为空地址，验证协商按不可用降级（HCCL_E_UNAVAIL）且资源回滚、可重复进入。
 */
#include <gtest/gtest.h>
#include "comm_negotiation.h"
#include "ut_neg_stub.h"

using namespace ops_hccl;
using ut::g_negStub;

namespace {
HcclComm FakeComm(uintptr_t id) { return reinterpret_cast<HcclComm>(id); }
} // namespace

class CommNegotiationSymbolMissingTest : public ::testing::Test {
protected:
    void SetUp() override { ut::NegStubReset(); }
};

TEST_F(CommNegotiationSymbolMissingTest, SubCommSymbolMissingDegrade)
{
    // 老版本HCOMM无HcclCreateSubCommConfig：协商按不可用降级（HCCL_E_UNAVAIL），
    // 不建子通信域/不发起原语/不注册回调，且EngineCtx回滚（下次调用可重试）
    HcclComm comm = FakeComm(0x6001);
    bool allRanksCapable = true;
    EXPECT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_E_UNAVAIL);
    EXPECT_EQ(g_negStub.createSubCommCnt, 0);
    EXPECT_EQ(g_negStub.allReduceCnt, 0);
    EXPECT_EQ(g_negStub.regCbCnt, 0);
    EXPECT_FALSE(ut::NegStubHasCtx(comm, "ut_world_ut_negotiation"));
    // 同comm重试：降级路径可重复进入，不残留半建状态
    EXPECT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_E_UNAVAIL);
}

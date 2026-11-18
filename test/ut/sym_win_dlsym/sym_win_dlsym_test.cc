/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测点：RefreshSymWinRankToMember 的 rank→LSA memberId 登记、
// GetSymWinRankToMemberTable 的下发结果，以及 GetSymWinRemoteMem 的逐 peer 取址透传语义。

#include <cstddef>
#include <cstdint>

#include "gtest/gtest.h"

#include "hccl_sym_win_dl.h"

void SymWinMockReset();
void SymWinMockSetMemberIdOffset(uint32_t offset);
void SymWinMockSetRankToMemberFailRank(int32_t rank);
void SymWinMockSetGetLsaTeamFail(bool fail);
void SymWinMockSetRemoteAddrDlsymMissing(bool missing);
void SymWinMockSetLegacyResult(HcclResult result);
bool SymWinMockLegacyCalled();
bool SymWinMockRemoteAddrCalled();
bool SymWinMockPeerPointerCalled();
uint32_t SymWinMockLegacyPeerId();
size_t SymWinMockLegacyOffset();
void* SymWinMockLegacyPtrBase();

namespace {

constexpr uint32_t RANK_SIZE = 4;
constexpr uint32_t MEMBER_ID_OFFSET = 10;
constexpr uint32_t UNFILLED = 0xFFFFFFFFU;

class SymWinRankMapTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        SymWinMockReset();
        // 打开 legacy 取址支持位：HcclSymWinDlInit 只探测符号是否存在，取址实现由本 UT 的强符号提供
        HcclSymWinDlInit(reinterpret_cast<void*>(0x1));
        comm_ = reinterpret_cast<HcclComm>(0x100);
    }

    HcclComm comm_{nullptr};
};

TEST_F(SymWinRankMapTest, TableMissReportsFailureWithoutGuessingIdentity)
{
    // 未登记窗口不得按 identity 猜测：多层/子团队下 rank 与 memberId 不等价，猜出的映射会指向错误 rank
    void* win = reinterpret_cast<void*>(0x2003);
    uint32_t members[RANK_SIZE] = {UNFILLED, UNFILLED, UNFILLED, UNFILLED};
    EXPECT_FALSE(GetSymWinRankToMemberTable(win, RANK_SIZE, members));
    for (uint32_t rank = 0; rank < RANK_SIZE; rank++) {
        EXPECT_EQ(members[rank], UNFILLED);
    }
}

TEST_F(SymWinRankMapTest, TableCarriesRegisteredMemberId)
{
    SymWinMockSetMemberIdOffset(MEMBER_ID_OFFSET);
    void* win = reinterpret_cast<void*>(0x2000);
    ASSERT_TRUE(RefreshSymWinRankToMember(comm_, win, RANK_SIZE));

    uint32_t members[RANK_SIZE] = {};
    EXPECT_TRUE(GetSymWinRankToMemberTable(win, RANK_SIZE, members));
    for (uint32_t rank = 0; rank < RANK_SIZE; rank++) {
        EXPECT_EQ(members[rank], rank + MEMBER_ID_OFFSET);
    }
}

TEST_F(SymWinRankMapTest, IncompleteRankMappingReturnsFalseAndIsNotRegistered)
{
    SymWinMockSetMemberIdOffset(MEMBER_ID_OFFSET);
    SymWinMockSetRankToMemberFailRank(2);
    void* win = reinterpret_cast<void*>(0x2001);
    // rank 2 不在 LSA WorldTeam 内：整体判定未覆盖且不登记半成品映射
    EXPECT_FALSE(RefreshSymWinRankToMember(comm_, win, RANK_SIZE));

    uint32_t members[RANK_SIZE] = {UNFILLED, UNFILLED, UNFILLED, UNFILLED};
    EXPECT_FALSE(GetSymWinRankToMemberTable(win, RANK_SIZE, members));
    for (uint32_t rank = 0; rank < RANK_SIZE; rank++) {
        EXPECT_EQ(members[rank], UNFILLED);
    }
}

TEST_F(SymWinRankMapTest, PartialMappingFailedRankRecoversAfterFullCoverage)
{
    SymWinMockSetMemberIdOffset(MEMBER_ID_OFFSET);
    SymWinMockSetRankToMemberFailRank(1);
    void* win = reinterpret_cast<void*>(0x2002);
    EXPECT_FALSE(RefreshSymWinRankToMember(comm_, win, RANK_SIZE));

    SymWinMockSetRankToMemberFailRank(-1);
    ASSERT_TRUE(RefreshSymWinRankToMember(comm_, win, RANK_SIZE));

    uint32_t members[RANK_SIZE] = {};
    EXPECT_TRUE(GetSymWinRankToMemberTable(win, RANK_SIZE, members));
    EXPECT_EQ(members[1], 1 + MEMBER_ID_OFFSET);
}

TEST_F(SymWinRankMapTest, GetLsaTeamFailFallsBackToNonSymmetric)
{
    // LSA team 接口存在但查询失败：不得按 identity 猜测（多层/子团队会静默用错 memberId），整体回退
    SymWinMockSetMemberIdOffset(MEMBER_ID_OFFSET);
    SymWinMockSetGetLsaTeamFail(true);
    void* win = reinterpret_cast<void*>(0x2008);
    EXPECT_FALSE(RefreshSymWinRankToMember(comm_, win, RANK_SIZE));

    // 失败时不登记映射，查表必须报未命中，不得退化为 identity
    uint32_t members[RANK_SIZE] = {UNFILLED, UNFILLED, UNFILLED, UNFILLED};
    EXPECT_FALSE(GetSymWinRankToMemberTable(win, RANK_SIZE, members));
    for (uint32_t rank = 0; rank < RANK_SIZE; rank++) {
        EXPECT_EQ(members[rank], UNFILLED);
    }
}

TEST_F(SymWinRankMapTest, ShortRegisteredTableMissesInsteadOfTruncating)
{
    // 先以较小 rankSize 登记，再用更大 rankSize 查表：不得读取短表前缀，应整体报未命中
    SymWinMockSetMemberIdOffset(MEMBER_ID_OFFSET);
    void* win = reinterpret_cast<void*>(0x2009);
    ASSERT_TRUE(RefreshSymWinRankToMember(comm_, win, 2));

    uint32_t members[RANK_SIZE] = {UNFILLED, UNFILLED, UNFILLED, UNFILLED};
    EXPECT_FALSE(GetSymWinRankToMemberTable(win, RANK_SIZE, members));
    for (uint32_t rank = 0; rank < RANK_SIZE; rank++) {
        EXPECT_EQ(members[rank], UNFILLED);
    }
}

TEST_F(SymWinRankMapTest, RemoteMemPassesCommRankToLegacy)
{
    SymWinMockSetMemberIdOffset(MEMBER_ID_OFFSET);
    void* win = reinterpret_cast<void*>(0x2005);
    ASSERT_TRUE(RefreshSymWinRankToMember(comm_, win, RANK_SIZE));

    void* ptr = nullptr;
    ASSERT_EQ(GetSymWinRemoteMem(win, 256, 3, &ptr), HCCL_SUCCESS);
    EXPECT_TRUE(SymWinMockLegacyCalled());
    EXPECT_TRUE(SymWinMockRemoteAddrCalled());
    // 逐 peer 取址的入参语义是通信域 rank，不能复用 rank→memberId 换算结果
    EXPECT_EQ(SymWinMockLegacyPeerId(), 3U);
    EXPECT_EQ(SymWinMockLegacyOffset(), 256U);
    EXPECT_EQ(ptr, reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(SymWinMockLegacyPtrBase()) + 256));
}

TEST_F(SymWinRankMapTest, RemoteMemFallsBackToPeerPointerWhenRemoteAddrUnavailable)
{
    // 老版 hcomm 只导出 HcclSymWinGetPeerPointer：重新探测后应回退到该符号
    SymWinMockSetRemoteAddrDlsymMissing(true);
    HcclSymWinDlInit(reinterpret_cast<void*>(0x1));
    void* win = reinterpret_cast<void*>(0x200A);
    void* ptr = nullptr;
    ASSERT_EQ(GetSymWinRemoteMem(win, 128, 1, &ptr), HCCL_SUCCESS);
    EXPECT_FALSE(SymWinMockRemoteAddrCalled());
    EXPECT_TRUE(SymWinMockPeerPointerCalled());
    EXPECT_EQ(ptr, reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(SymWinMockLegacyPtrBase()) + 128));
}

TEST_F(SymWinRankMapTest, RemoteMemPropagatesLegacyError)
{
    // legacy 取址失败时须原样透传错误码，不得伪造成功
    SymWinMockSetLegacyResult(HCCL_E_PARA);
    void* win = reinterpret_cast<void*>(0x200B);
    void* ptr = nullptr;
    EXPECT_EQ(GetSymWinRemoteMem(win, 64, 2, &ptr), HCCL_E_PARA);
    EXPECT_TRUE(SymWinMockRemoteAddrCalled());
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(SymWinRankMapTest, NullInputsReturnTrueWithoutRegistration)
{
    void* win = reinterpret_cast<void*>(0x2006);
    EXPECT_TRUE(RefreshSymWinRankToMember(nullptr, win, RANK_SIZE));
    EXPECT_TRUE(RefreshSymWinRankToMember(comm_, nullptr, RANK_SIZE));
    EXPECT_TRUE(RefreshSymWinRankToMember(comm_, win, 0));

    uint32_t members[RANK_SIZE] = {UNFILLED, UNFILLED, UNFILLED, UNFILLED};
    EXPECT_FALSE(GetSymWinRankToMemberTable(win, RANK_SIZE, members));
    for (uint32_t rank = 0; rank < RANK_SIZE; rank++) {
        EXPECT_EQ(members[rank], UNFILLED);
    }
}

TEST_F(SymWinRankMapTest, ZeroRankSizeKeepsExistingMapping)
{
    SymWinMockSetMemberIdOffset(MEMBER_ID_OFFSET);
    void* win = reinterpret_cast<void*>(0x2007);
    ASSERT_TRUE(RefreshSymWinRankToMember(comm_, win, RANK_SIZE));
    ASSERT_TRUE(RefreshSymWinRankToMember(comm_, win, 0));

    uint32_t members[RANK_SIZE] = {};
    EXPECT_TRUE(GetSymWinRankToMemberTable(win, RANK_SIZE, members));
    EXPECT_EQ(members[1], 1 + MEMBER_ID_OFFSET);
}

} // namespace

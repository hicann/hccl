/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UT 桩：替换 hccl_sym_win_dl.cc 的运行期依赖，使 host 侧 rank→memberId 登记与逐 peer 取址可被独立驱动。
// 1) HcclDlopen/HcclDlsym/HcclDlclose 返回伪句柄与桩函数指针，避免真实 dlopen("libhcomm.so")；
// 2) 以强符号覆盖 hccl_sym_win_dl.cc 内的 weak 取址实现，记录入参供断言；
// 3) DlogRecord 提供桩实现，满足链接依赖。

#include <cstddef>
#include <cstdint>
#include <string>

#include "hccl/hccl_types.h"
#include "hccl_dl.h"

namespace {

struct MockCallRecord {
    bool called{false};
    void* winHandle{nullptr};
    size_t offset{0};
    uint32_t peerId{0};
};

MockCallRecord g_legacyCall;
uint32_t g_memberIdOffset = 0;
int32_t g_rankToMemberFailRank = -1;
bool g_getLsaTeamFail = false;
bool g_remoteAddrDlsymMissing = false;
bool g_remoteAddrUsed = false;
bool g_peerPointerUsed = false;
HcclResult g_legacyResult = HCCL_SUCCESS;

constexpr uintptr_t MOCK_LEGACY_PTR_BASE = 0x4000;
constexpr uintptr_t MOCK_LSA_TEAM = 0x5000;

HcclResult MockLegacyGetRemoteAddr(void* winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    g_legacyCall.called = true;
    g_legacyCall.winHandle = winHandle;
    g_legacyCall.offset = offset;
    g_legacyCall.peerId = peerRank;
    if (g_legacyResult != HCCL_SUCCESS) {
        return g_legacyResult;
    }
    if (ptr != nullptr) {
        *ptr = reinterpret_cast<void*>(MOCK_LEGACY_PTR_BASE + offset);
    }
    return HCCL_SUCCESS;
}

HcclResult MockGetRemoteAddr(void* winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    g_remoteAddrUsed = true;
    return MockLegacyGetRemoteAddr(winHandle, offset, peerRank, ptr);
}

HcclResult MockGetPeerPointer(void* winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    g_peerPointerUsed = true;
    return MockLegacyGetRemoteAddr(winHandle, offset, peerRank, ptr);
}

HcclResult MockGetLsaTeam(HcclComm comm, void** lsaTeam)
{
    (void)comm;
    if (g_getLsaTeamFail || lsaTeam == nullptr) {
        return HCCL_E_PARA;
    }
    *lsaTeam = reinterpret_cast<void*>(MOCK_LSA_TEAM);
    return HCCL_SUCCESS;
}

HcclResult MockRankToMember(HcclComm comm, void* lsaTeam, uint32_t rankId, uint32_t* memberId)
{
    (void)comm;
    (void)lsaTeam;
    if (memberId == nullptr) {
        return HCCL_E_PARA;
    }
    if (g_rankToMemberFailRank >= 0 && rankId == static_cast<uint32_t>(g_rankToMemberFailRank)) {
        return HCCL_E_PARA;
    }
    *memberId = rankId + g_memberIdOffset;
    return HCCL_SUCCESS;
}

} // namespace

void SymWinMockReset()
{
    g_legacyCall = MockCallRecord();
    g_memberIdOffset = 0;
    g_rankToMemberFailRank = -1;
    g_getLsaTeamFail = false;
    g_remoteAddrDlsymMissing = false;
    g_remoteAddrUsed = false;
    g_peerPointerUsed = false;
    g_legacyResult = HCCL_SUCCESS;
}

void SymWinMockSetMemberIdOffset(uint32_t offset) { g_memberIdOffset = offset; }

void SymWinMockSetRankToMemberFailRank(int32_t rank) { g_rankToMemberFailRank = rank; }

void SymWinMockSetGetLsaTeamFail(bool fail) { g_getLsaTeamFail = fail; }

void SymWinMockSetRemoteAddrDlsymMissing(bool missing) { g_remoteAddrDlsymMissing = missing; }

void SymWinMockSetLegacyResult(HcclResult result) { g_legacyResult = result; }

bool SymWinMockLegacyCalled() { return g_remoteAddrUsed || g_peerPointerUsed; }

bool SymWinMockRemoteAddrCalled() { return g_remoteAddrUsed; }

bool SymWinMockPeerPointerCalled() { return g_peerPointerUsed; }

uint32_t SymWinMockLegacyPeerId() { return g_legacyCall.peerId; }

size_t SymWinMockLegacyOffset() { return g_legacyCall.offset; }

void* SymWinMockLegacyPtrBase() { return reinterpret_cast<void*>(MOCK_LEGACY_PTR_BASE); }

extern "C" {

void* HcclDlopen(const char* libName, int mode)
{
    (void)libName;
    (void)mode;
    return reinterpret_cast<void*>(0x1);
}

// hccl_sym_win_dl.cc 复用 HcommDlInit 已加载的 libhcomm 句柄；UT 以伪句柄满足链接，符号解析仍走 HcclDlsym 桩
void* HcommGetLibHandle(void) { return reinterpret_cast<void*>(0x1); }

void* HcclDlsym(void* handle, const char* funcName)
{
    (void)handle;
    if (funcName == nullptr) {
        return nullptr;
    }
    const std::string name(funcName);
    if (name == "HcclTeamGetLsaTeam") {
        return reinterpret_cast<void*>(&MockGetLsaTeam);
    }
    if (name == "HcclTeamRankToMember") {
        return reinterpret_cast<void*>(&MockRankToMember);
    }
    if (name == "HcclSymWinGetRemoteAddr") {
        return g_remoteAddrDlsymMissing ? nullptr : reinterpret_cast<void*>(&MockGetRemoteAddr);
    }
    if (name == "HcclSymWinGetPeerPointer") {
        return reinterpret_cast<void*>(&MockGetPeerPointer);
    }
    return nullptr;
}

int HcclDlclose(void* handle)
{
    (void)handle;
    return 0;
}

// 强符号覆盖 hccl_sym_win_dl.cc 内的 weak 默认实现；两者入参语义均为通信域 rank
HcclResult HcclSymWinGetPeerPointer(void* winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    return MockGetPeerPointer(winHandle, offset, peerRank, ptr);
}

HcclResult HcclSymWinGetRemoteAddr(void* winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    return MockGetRemoteAddr(winHandle, offset, peerRank, ptr);
}

void DlogRecord(int32_t moduleId, int32_t level, const char* fmt, ...)
{
    (void)moduleId;
    (void)level;
    (void)fmt;
}

} // extern "C"

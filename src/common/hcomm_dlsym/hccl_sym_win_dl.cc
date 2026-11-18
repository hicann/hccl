/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_sym_win_dl.h"
#include "hcomm_dlsym.h"
#include <algorithm>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

DEFINE_WEAK_FUNC(
    HcclResult, HcclSymWinGetPeerPointer, HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);
DEFINE_WEAK_FUNC(
    HcclResult, HcclSymWinGetRemoteAddr, HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);
DEFINE_WEAK_FUNC(
    HcclResult, HcclSymWinGetPeerDevicePointer, HcclCommSymWindow winHandle, size_t offset, uint32_t peerMemberId,
    void** ptr);

// 弱符号默认实现：未链接 hcomm_dlsym.cc 的裁剪/UT 构建拿不到库句柄，按「本次解析失败」处理（不固化，可重试）；
// 完整构建由 hcomm_dlsym.cc 中的强符号覆盖。
extern "C" __attribute__((weak)) void* HcommGetLibHandle(void) { return nullptr; }

void HcclSymWinDlInit(void* libHcommHandle)
{
    INIT_SUPPORT_FLAG(libHcommHandle, HcclSymWinGetPeerPointer);
    INIT_SUPPORT_FLAG(libHcommHandle, HcclSymWinGetRemoteAddr);
}

void HcclSymWinHostDlInit(void* libHcommHandle) { INIT_SUPPORT_FLAG(libHcommHandle, HcclSymWinGetPeerDevicePointer); }

// device 侧兜底：x86 host 上通常不可用，保留以兼容 aarch64/仿真环境。
// 只探测一次且失败不重试，避免每次取对端地址都重复 dlopen。
static std::once_flag gSymWinDeviceInitOnce;

static void EnsureDeviceSymWinDlInit(void)
{
    if (HcommIsSupportHcclSymWinGetPeerPointer() || HcommIsSupportHcclSymWinGetRemoteAddr()) {
        return;
    }
    void* handle = HcclDlopen("libccl_kernel.so", RTLD_NOW);
    if (handle == nullptr) {
        HCCL_COMPAT_ERROR("[HcclWrapper] no symmetric-memory peer pointer provider available");
        return;
    }
    HcclSymWinDlInit(handle);
}

// hcomm 的 team 转换接口：把通信域 rank 换算成 LSA WorldTeam member ID；
// HcommTeamHandle 为不透明句柄，统一按 void* 传递。
typedef HcclResult (*HcclTeamGetLsaTeamFn)(HcclComm comm, void** lsaTeam);
typedef HcclResult (*HcclTeamRankToMemberFn)(HcclComm comm, void* lsaTeam, uint32_t rankId, uint32_t* memberId);

struct SymWinTeamApi {
    HcclTeamGetLsaTeamFn getLsaTeam{nullptr};
    HcclTeamRankToMemberFn rankToMember{nullptr};
};

enum class SymWinTeamApiState {
    READY,      // 接口已就绪
    ABSENT,     // 句柄有效但库中确实没有该接口（旧版 hcomm），rank==memberId 为历史兼容行为
    UNAVAILABLE // 本次解析失败（库句柄未就绪等），不固化结论，下次调用重试
};

static std::mutex gSymWinTeamApiMutex;
static SymWinTeamApi gSymWinTeamApi;
static bool gSymWinTeamApiReady = false;
static bool gSymWinTeamApiAbsent = false;

// 复用 HcommDlInit 已持有的 libhcomm 句柄：另开 dlopen 在加载路径不同/静态链接时可能失败，
// 且失败一旦被 call_once 固化，后续所有窗口都会永久退化为 identity 映射。
static SymWinTeamApiState ResolveSymWinTeamApi(SymWinTeamApi& api)
{
    std::lock_guard<std::mutex> lock(gSymWinTeamApiMutex);
    if (gSymWinTeamApiReady) {
        api = gSymWinTeamApi;
        return SymWinTeamApiState::READY;
    }
    if (gSymWinTeamApiAbsent) {
        return SymWinTeamApiState::ABSENT;
    }
    void* handle = HcommGetLibHandle();
    if (handle == nullptr) {
        HCCL_COMPAT_ERROR("[HcclWrapper] libhcomm handle is not ready, retry LSA team API resolution next time");
        return SymWinTeamApiState::UNAVAILABLE;
    }
    void* getLsaTeam = HcclDlsym(handle, "HcclTeamGetLsaTeam");
    void* rankToMember = HcclDlsym(handle, "HcclTeamRankToMember");
    if (getLsaTeam == nullptr || rankToMember == nullptr) {
        gSymWinTeamApiAbsent = true;
        HCCL_LOG_PRINT(
            HCCL, HCCL_LOG_WARN,
            "[HcclWrapper] libhcomm has no LSA team rank/member API, fallback to identity mapping (rank==memberId)");
        return SymWinTeamApiState::ABSENT;
    }
    gSymWinTeamApi.getLsaTeam = reinterpret_cast<HcclTeamGetLsaTeamFn>(getLsaTeam);
    gSymWinTeamApi.rankToMember = reinterpret_cast<HcclTeamRankToMemberFn>(rankToMember);
    gSymWinTeamApiReady = true;
    api = gSymWinTeamApi;
    return SymWinTeamApiState::READY;
}

// 窗口句柄 → 按通信域 rank 索引的 memberId 表。hccl 侧拿不到窗口销毁通知，条目可能滞后，
// 故每次算子入口都用当前通信域重刷；句柄复用时新映射在使用前覆盖旧表。
constexpr size_t SYM_WIN_RANK_MAP_MAX = 64;

struct SymWinRankMapEntry {
    std::vector<uint32_t> memberIds;
    uint64_t seq{0};
};

static std::mutex gSymWinRankMapMutex;
static std::map<void*, SymWinRankMapEntry> gSymWinRankMaps;
static uint64_t gSymWinRankMapSeq = 0;

bool RefreshSymWinRankToMember(HcclComm comm, HcclCommSymWindow winHandle, uint32_t rankSize)
{
    if (winHandle == nullptr || comm == nullptr || rankSize == 0) {
        return true; // 无需换算，视为 rank 已覆盖
    }
    std::vector<uint32_t> memberIds(rankSize);
    SymWinTeamApi teamApi;
    const SymWinTeamApiState teamApiState = ResolveSymWinTeamApi(teamApi);
    void* lsaTeam = nullptr;
    bool covered = true;
    if (teamApiState == SymWinTeamApiState::ABSENT) {
        // 旧版 hcomm 无 LSA team 接口：按 rank == memberId 处理（历史行为）
        for (uint32_t rank = 0; rank < rankSize; rank++) {
            memberIds[rank] = rank;
        }
    } else if (teamApiState != SymWinTeamApiState::READY) {
        // 本次解析失败（库句柄未就绪等）：不能固化为 identity，整体回退非对称，下次算子入口会重试解析
        HCCL_COMPAT_ERROR("[HcclWrapper] LSA team API unresolved for window[%p], fallback to non-symmetric", winHandle);
        return false;
    } else if (teamApi.getLsaTeam(comm, &lsaTeam) != HCCL_SUCCESS || lsaTeam == nullptr) {
        // 接口存在但查询失败/无 team：不能按 identity 猜测，多层/子团队下会静默用错 memberId，整体回退
        HCCL_COMPAT_ERROR("[HcclWrapper] get LSA team failed for window[%p], fallback to non-symmetric", winHandle);
        return false;
    } else {
        for (uint32_t rank = 0; rank < rankSize; rank++) {
            uint32_t memberId = rank;
            if (teamApi.rankToMember(comm, lsaTeam, rank, &memberId) != HCCL_SUCCESS) {
                // 该 rank 不在 LSA WorldTeam 内（多层/子团队），映射不完整
                HCCL_COMPAT_DEBUG("[HcclWrapper] rank[%u]/[%u] not in LSA team, win[%p]", rank, rankSize, winHandle);
                covered = false;
                break;
            }
            memberIds[rank] = memberId;
        }
    }
    if (!covered) {
        return false; // 不登记半成品映射，避免取址侧拿到部分转换结果
    }
    std::lock_guard<std::mutex> lock(gSymWinRankMapMutex);
    if (gSymWinRankMaps.size() >= SYM_WIN_RANK_MAP_MAX && gSymWinRankMaps.find(winHandle) == gSymWinRankMaps.end()) {
        // 表项无销毁通知，超限时淘汰最久未刷新的条目：活跃窗口每次算子入口都会刷新，不会被并发 churn
        // 立刻挤掉；被淘汰的窗口取址时按未命中整体回退非对称，不会再静默退化成 identity。
        auto oldest
            = std::min_element(gSymWinRankMaps.begin(), gSymWinRankMaps.end(), [](const auto& lhs, const auto& rhs) {
                  return lhs.second.seq < rhs.second.seq;
              });
        HCCL_COMPAT_DEBUG(
            "[HcclWrapper] rank map exceeds %zu entries, evict window[%p]", SYM_WIN_RANK_MAP_MAX, oldest->first);
        gSymWinRankMaps.erase(oldest);
    }
    gSymWinRankMaps[winHandle] = SymWinRankMapEntry{std::move(memberIds), ++gSymWinRankMapSeq};
    return true;
}

bool GetSymWinRankToMemberTable(HcclCommSymWindow winHandle, uint32_t rankSize, uint32_t* memberIds)
{
    if (memberIds == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(gSymWinRankMapMutex);
    auto it = gSymWinRankMaps.find(winHandle);
    // 未登记/登记不完整（入口未刷新、窗口表淘汰等）不能按 identity 猜测：多层/子团队下 rank 与 memberId
    // 不等价，猜出的 memberId 可能仍是合法值，取址会成功但指向错误 rank，造成静默写错对端 buffer。
    if (it == gSymWinRankMaps.end() || it->second.memberIds.size() < rankSize) {
        return false;
    }
    for (uint32_t rank = 0; rank < rankSize; rank++) {
        memberIds[rank] = it->second.memberIds[rank];
    }
    return true;
}

HcclResult GetSymWinRemoteMem(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    std::call_once(gSymWinDeviceInitOnce, EnsureDeviceSymWinDlInit);
    if (HcommIsSupportHcclSymWinGetRemoteAddr()) {
        return HcclSymWinGetRemoteAddr(winHandle, offset, peerRank, ptr);
    }
    return HcclSymWinGetPeerPointer(winHandle, offset, peerRank, ptr);
}

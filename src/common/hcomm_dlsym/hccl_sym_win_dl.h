/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SYM_WIN_DL_H
#define HCCL_SYM_WIN_DL_H

#include <cstddef>
#include <cstdint>
#include "dlsym_common.h"
#include "hccl_comm.h" // 原始头文件，包含所有类型和声明

/* 8.5.0 桩: HcclCommSymWindow (来自 hccl_types.h) */
#if CANN_VERSION_NUM < CANN_VERSION(9, 0, 0)
typedef void* HcclCommSymWindow;
#endif

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(
    HcclResult, HcclSymWinGetPeerPointer, HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);
DECL_SUPPORT_FLAG(HcclSymWinGetPeerPointer);
DECL_WEAK_FUNC(
    HcclResult, HcclSymWinGetRemoteAddr, HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);
DECL_SUPPORT_FLAG(HcclSymWinGetRemoteAddr);
/* hcomm host 侧 UB 对称窗口取址（libhcomm.so 导出）：peerMemberId 为 LSA WorldTeam memberId，
 * 返回 offset 处可直接访问的对端 device VA；非 UB 窗口返回 HCCL_E_NOT_SUPPORT。 */
DECL_WEAK_FUNC(
    HcclResult, HcclSymWinGetPeerDevicePointer, HcclCommSymWindow winHandle, size_t offset, uint32_t peerMemberId,
    void** ptr);
DECL_SUPPORT_FLAG(HcclSymWinGetPeerDevicePointer);

void HcclSymWinDlInit(void* libHcommHandle);
/* 初始化 host 侧对称窗口取址接口（libhcomm.so）。与 device 侧 HcclSymWinDlInit 分开调用，避免支持标记互相复位。 */
void HcclSymWinHostDlInit(void* libHcommHandle);
/* peerRank 为通信域 rank：URMA / legacy 远端地址查询接口的入参语义，调用方直接透传。
 * 仅用于非 UB 对称窗口的逐 peer 取址（UB 窗口走 HcclSymWinGetPeerDevicePointer）。 */
HcclResult GetSymWinRemoteMem(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);

/*
 * UB 对称窗口的「通信域 rank → LSA WorldTeam memberId」映射。
 * 对端 VA 由 hcomm 按 memberId 计算，因此需要在算子入口按窗口刷新该映射（GetSymWinRankToMemberTable 只查表）。
 * 单层 LSA 下 rank 与 memberId 等价；多层/子团队（lsaTeamSize < rankSize）下不等价。
 * 旧版 hcomm（无 LSA team 接口）退化为 rank 即 memberId；接口存在但查询失败时返回 false。
 * 返回 false 表示映射不可信（存在不在 LSA WorldTeam 内的 rank，或 team 查询失败），
 * 此时需整体关闭对称内存，否则对端取址必然失败。
 */
bool RefreshSymWinRankToMember(HcclComm comm, HcclCommSymWindow winHandle, uint32_t rankSize);

/* 窗口的「通信域 rank → LSA memberId」表。仅当该窗口已按不小于 rankSize 的规模成功登记时返回 true 并填充；
 * 未登记或登记不完整（如窗口表淘汰、入口未刷新）返回 false 且不写 memberIds。调用方不得按 identity 猜测：
 * 多层/子团队下 rank 与 memberId 不等价，猜出的 memberId 可能仍是合法值，取址会成功但指向错误 rank，
 * 此时应整体回退非对称。rankSize 由调用方限界。 */
bool GetSymWinRankToMemberTable(HcclCommSymWindow winHandle, uint32_t rankSize, uint32_t* memberIds);

#ifdef __cplusplus
}
#endif

#endif // HCCL_SYM_WIN_DL_H

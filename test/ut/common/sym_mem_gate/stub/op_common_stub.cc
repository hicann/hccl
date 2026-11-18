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

#include "adapter_acl.h"
#include "auto_selector_base.h"
#include "hccl_aiv_utils.h"
#include "hccl_sym_win_dl.h"

// op_common.cc IsSymWinLsaCovered 引用，真实实现定义在 hccl_sym_win_dl.cc（本 UT 不链接）。
// LSA 覆盖校验仅在 AIV 引擎分支生效，本 UT 只覆盖 AICPU_TS；桩按旧版 hcomm（rank == memberId）视为已覆盖。
extern "C" bool RefreshSymWinRankToMember(HcclComm comm, HcclCommSymWindow winHandle, uint32_t rankSize)
{
    (void)comm;
    (void)winHandle;
    (void)rankSize;
    return true;
}

namespace ops_hccl {
// op_common.cc 同时引用 AIV/CCU 分支的依赖；对称内存门禁用例不应触发，仅用于链接。
HcclResult haclrtMemcpy(void*, size_t, const void*, size_t, aclrtMemcpyKind)
{
    ADD_FAILURE() << "对称内存门禁用例不应调用haclrtMemcpy";
    return HCCL_E_NOT_SUPPORT;
}

HcclResult haclrtMemset(void*, size_t, int32_t, size_t)
{
    ADD_FAILURE() << "对称内存门禁用例不应调用haclrtMemset";
    return HCCL_E_NOT_SUPPORT;
}

HcclResult ClearAivTagCb(HcclComm, HcclCommStatePhase, void*)
{
    ADD_FAILURE() << "对称内存门禁用例不应调用ClearAivTagCb";
    return HCCL_E_NOT_SUPPORT;
}

// op_common.cc CheckNdaMainboardSupported 引用，定义在 adapter_acl.cc（本 UT 不链接），按取不到主板信息处理
HcclResult hcalrtGetDeviceInfo(u32 deviceId, aclrtDevAttr devAttr, s64& val, bool quiet)
{
    (void)deviceId;
    (void)devAttr;
    (void)val;
    (void)quiet;
    return HCCL_E_NOT_SUPPORT;
}

// op_common.cc IsUbxAicpuTsAllToAllSymAlg 引用，真实实现定义在 auto_selector_base.cc（本 UT 不链接）。
// 门禁用例不构造 UBX alltoall 拓扑，按非 concurrent 拓扑处理。
HcclResult AutoSelectorBase::CheckMeshNumEqualToClosNum(const TopoInfoWithNetLayerDetails* topoInfo, bool& isEqual)
{
    (void)topoInfo;
    isEqual = false;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

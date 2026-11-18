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
#include "hccl_aiv_utils.h"
#include "hccl_sym_win_dl.h"

// op_common.cc 的对端对称 VA 取址引用，真实实现定义在 hccl_sym_win_dl.cc（本 UT 不链接）；
// CCU 资源申请用例不走 AIV 对称路径，此处恒视为已登记并按 identity 填充，避免影响用例。
extern "C" bool GetSymWinRankToMemberTable(HcclCommSymWindow, uint32_t rankSize, uint32_t* memberIds)
{
    if (memberIds == nullptr) {
        return false;
    }
    for (uint32_t rank = 0; rank < rankSize; rank++) {
        memberIds[rank] = rank;
    }
    return true;
}

namespace ops_hccl {
// 对称 VA 表签名缓存，真实实现定义在 hccl_aiv_utils.cc（本 UT 不链接），此处恒视为未填表。
bool IsAivSymPeerTableFilled(const void*, const AivSymSign&) { return false; }

void RecordAivSymPeerTable(const void*, const AivSymSign&) {}

// GetAlgResWithEngine同时引用非CCU分支；这些依赖仅用于链接，CCU用例不应调用。
HcclResult haclrtMemcpy(void*, size_t, const void*, size_t, aclrtMemcpyKind)
{
    ADD_FAILURE() << "CCU资源申请用例不应调用haclrtMemcpy";
    return HCCL_E_NOT_SUPPORT;
}

HcclResult haclrtMemset(void*, size_t, int32_t, size_t)
{
    ADD_FAILURE() << "CCU资源申请用例不应调用haclrtMemset";
    return HCCL_E_NOT_SUPPORT;
}

HcclResult ClearAivTagCb(HcclComm, HcclCommStatePhase, void*)
{
    ADD_FAILURE() << "CCU资源申请用例不应调用ClearAivTagCb";
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
} // namespace ops_hccl

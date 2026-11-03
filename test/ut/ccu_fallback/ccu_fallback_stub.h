/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_FALLBACK_UT_STUB_H
#define CCU_FALLBACK_UT_STUB_H

#include <cstdint>
#include "hccl.h"

namespace ut_stub {
// 桩调用计数：观测被测函数是否进入跨rank协商链路（AllGather/AllReduce/协商子通信域创建等）
struct CcuFallbackStubCounters {
    uint32_t configGetInfoCalled{0}; // 由测试注入的DlHcommFunction::dlHcclConfigGetInfo桩计数
    uint32_t getRankSizeCalled{0};
    uint32_t getRankIdCalled{0};
    uint32_t getCommNameCalled{0};
    uint32_t engineCtxGetCalled{0};
    uint32_t engineCtxCreateCalled{0};
    uint32_t engineCtxDestroyCalled{0};
    uint32_t allGatherCalled{0};
    uint32_t allReduceCalled{0};
    uint32_t createSubCommCalled{0};
    uint32_t commDestroyCalled{0};
    uint32_t commResumeCalled{0};
    uint32_t regStateCbCalled{0};
    uint32_t submitCalled{0};
    uint32_t removeWorkerCalled{0};
    uint32_t reSelectorCalled{0};
    uint32_t setCommEngineCalled{0};
    uint32_t setOpParamAlgTagCalled{0};
    uint32_t inconsistentCheckOpTypeCalled{0};
    uint32_t reportCheckFailedCalled{0};
};

// 桩行为配置：展开模式回写值、rankSize与EngineCtx查询/创建返回值可按用例覆写
struct CcuFallbackStubConfig {
    HcclOpExpansionMode expansionMode{HCCL_OP_EXPANSION_MODE_INVALID};
    uint32_t rankSize{8};
    HcclResult engineCtxGetRet{HCCL_E_NOT_FOUND};
    HcclResult engineCtxCreateRet{HCCL_E_INTERNAL};
};

extern CcuFallbackStubCounters g_ccuFallbackStubCounters;
extern CcuFallbackStubConfig g_ccuFallbackStubConfig;

// 复位全部计数与桩行为配置，供每个用例SetUp调用
void ResetCcuFallbackStub();
} // namespace ut_stub

#endif // CCU_FALLBACK_UT_STUB_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 打桩HCCL依赖so所需空文件

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hccl/hccl_types.h>

#ifdef __cplusplus
extern "C" {
#endif

// HcclDedicatedThreadAcquire 桩：ST仿真不需要保序流，设置 *thread = 0 触发"OrderLaunch not required"降级路径
int HcclDedicatedThreadAcquire(void* comm, unsigned int useType, uint32_t notifyNumPerThread, uint64_t* thread)
{
    if (thread != nullptr) {
        *thread = 0;
    }
    return 0;
}

// HcommEndpointCheckFeature 桩：ST仿真不支持NDA特性，设置 *isSupport = false 走 host 建链降级路径
int HcommEndpointCheckFeature(int featureType, void* endpoint, bool* isSupport)
{
    if (isSupport != nullptr) {
        *isSupport = false;
    }
    return 0;
}

// AICPU 控核查询接口桩（kernel_launch.h 弱符号）：默认取值永不触发门控（idle 远大于 reserved+1）。
// 注意：当前 ST 保序线程被上面的桩置零，门控在"无保序线程"条件即短路，本桩不会被咨询；
// 定义它保证 ST 二进制内符号已解析（而非 null 降级），为 ST 未来开启保序仿真预留。
// 如需 ST 级"开启控核完整下发"用例，需另行评估：HcclDedicatedThreadAcquire 桩返回非零、
// 链接 order_launch.cc、补 HcclThreadExportToCommEngine / HcommThreadNotifyRecord / Wait 桩 —— 涉及
// ST 保序仿真语义，建议与 ST 维护者对齐后单独交付。
uint32_t AicpuGetIdleCpuCnt() { return 8; }

uint32_t AicpuGetReservedCpuCnt() { return 0; }

// HcclThreadResGetInfo 桩：仅 HCCL_STUB_CAPTURE_ACTIVE=1 时回填占位流（判值对齐 aclrt_stub.cc），
// 否则返回 HCCL_E_NOT_SUPPORT——GetUnfoldStream 回退 param.stream，存量 ST 语义不变
static char g_simPlaceholderStream = 0;
int HcclThreadResGetInfo(void* comm, uint64_t thread, void* key, uint32_t size, void** out)
{
    if (out == nullptr || thread == 0 || size != sizeof(void*)) {
        return 1;
    }
    const char* captureActive = getenv("HCCL_STUB_CAPTURE_ACTIVE");
    if ((captureActive != nullptr) && (std::strcmp(captureActive, "1") == 0)) {
        *out = &g_simPlaceholderStream;
        return 0;
    }
    return static_cast<int>(HCCL_E_NOT_SUPPORT);
}

#ifdef __cplusplus
}
#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_TEMPLATE_UTILS_H
#define BSR_HOST_TEST_TEMPLATE_UTILS_H

#include <map>
#include <vector>

#include "alg_param.h"

namespace ops_hccl {

// 生产 BuffInfo 字段远多于此，被测代码只读 hcclBuff.size 与 inputPtr/outputPtr 及两个 base 偏移。
struct BuffInfo {
    void* inputPtr = nullptr;
    void* outputPtr = nullptr;
    HcclMem hcclBuff;
    BufferType inBuffType = BufferType::DEFAULT;
    BufferType outBuffType = BufferType::DEFAULT;
    BufferType hcclBuffType = BufferType::DEFAULT;
    u64 inBuffBaseOff = 0;
    u64 outBuffBaseOff = 0;
};

struct TemplateDataParams {
    BuffInfo buffInfo;
    u64 count = 0;
    u64 sliceSize = 0;
    u64 inputSliceStride = 0;
    u64 outputSliceStride = 0;
    u64 repeatNum = 0;
    u64 inputRepeatStride = 0;
    u64 outputRepeatStride = 0;
};

struct TemplateResource {
    std::vector<ThreadHandle> threads;
    void* aivCommInfoPtr = nullptr;
};

inline bool
IsAllConnetedWithTopo(const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType)
{
    (void)topoInfo;
    (void)netLayer;
    (void)topoType;
    return false;
}

} // namespace ops_hccl

#endif // BSR_HOST_TEST_TEMPLATE_UTILS_H

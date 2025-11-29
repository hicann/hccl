/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils.h"

namespace ops_hccl {

HcclMem HcclMemRange(HcclMem inMem, u64 offset, u64 size)
{
    HcclMem outMem;
    if (inMem.addr == nullptr) {
        HCCL_ERROR("HcclMem addr is null");
        return outMem;
    }
    if (offset + size > inMem.size){
        HCCL_ERROR("HcclMem request range[%llu] is out of size[%llu]", offset + size, inMem.size);
        return outMem;
    }
    outMem.type = inMem.type;
    outMem.addr = static_cast<void *>(static_cast<u8 *>(inMem.addr) + offset);
    outMem.size = size;
    return outMem;
}

u32 CalcCeilLog2(const u32 num)
{
    u32 ans = 0;
    for (u32 tmp = num - 1; tmp != 0; tmp >>= 1, ++ans) {}
    return ans;
}
}

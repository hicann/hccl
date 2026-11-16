/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_STUCK_DETECT_HCOMM_PRIMITIVES_H
#define UT_STUCK_DETECT_HCOMM_PRIMITIVES_H

#include <cstdint>

using ThreadHandle = uint64_t;

enum HcommDataType { HCOMM_DATA_TYPE_INT32 = 0 };

enum HcommReduceOp { HCOMM_REDUCE_SUM = 0 };

int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp);

#endif // UT_STUCK_DETECT_HCOMM_PRIMITIVES_H

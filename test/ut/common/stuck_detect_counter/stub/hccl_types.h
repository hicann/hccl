/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_STUCK_DETECT_HCCL_TYPES_H
#define UT_STUCK_DETECT_HCCL_TYPES_H

#include <cstdint>

using HcclResult = int;
using HcclComm = void*;

using u32 = uint32_t;
using u64 = uint64_t;

constexpr HcclResult HCCL_SUCCESS = 0;
constexpr HcclResult HCCL_E_INTERNAL = -4;

namespace ops_hccl {
// 与真实定义（op_common/inc/alg_param.h）保持一致：通信域名称最大长度
constexpr uint32_t COMM_INDENTIFIER_MAX_LENGTH = 128;
} // namespace ops_hccl

#endif // UT_STUCK_DETECT_HCCL_TYPES_H

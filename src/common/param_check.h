/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_COMMON_PARAM_CHECK
#define OPS_HCCL_SRC_COMMON_PARAM_CHECK

#include <hccl/hccl_types.h>
#include "hccl/base.h"

namespace ops_hccl {
HcclResult HcomCheckOpParam(const char *tag, const u64 count, const HcclDataType dataType, const char *group,
    const void *stream);

HcclResult HcomCheckOpParam(const char *tag, const u64 count, const HcclDataType dataType, const void *stream);

HcclResult HcomCheckOpParam(const char *tag, const u64 count, const HcclDataType dataType);

HcclResult HcomCheckUserRank(const u32 totalRanks, const u32 userRank);
}

#endif
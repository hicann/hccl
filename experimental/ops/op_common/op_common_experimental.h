/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_OP_COMMON_A3
#define OPS_HCCL_OP_COMMON_A3

#include <map>
#include <array>
#include "alg_param.h"
#include "executor_base.h"
#include "alg_type.h"
#include "op_common.h"
#include "scatter.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
extern thread_local std::map<HcclComm, NotifyArray> g_notifiesMap;
}

namespace ops_hccl_experimental {
using ops_hccl::OpParam;

HcclResult ProcessA3(HcclComm comm, OpParam& param, uint64_t beginTime);

HcclResult ExecOpBirs(HcclComm comm, OpParam& param);

bool IsStreamCapture(aclrtStream stream);

bool IsAiCpuMode(HcclDevType deviceType, u32 rankSize);

std::string SetLaunchMode(CommEngine engine);

} // namespace ops_hccl_experimental

#endif

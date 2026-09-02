/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_BARRIER_OP
#define OPS_HCCL_SRC_OPS_BARRIER_OP

#include <string>
#include "hccl.h"
#include "alg_param.h"
#include "alg_type.h"
#include "execute_selector.h"
#include "executor_v2_base.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclBarrier(HcclComm comm, aclrtStream stream);
HcclResult HcclBarrierInner(HcclComm comm, aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
HcclResult BarrierOutPlace(HcclComm comm, aclrtStream stream, const std::string& tag);
HcclResult CheckBarrierInputPara(const HcclComm comm, const aclrtStream stream);
HcclResult BarrierInitAndCheck(HcclComm comm, aclrtStream stream, std::string& opTag);
HcclResult BarrierEntryLog(aclrtStream stream, const std::string& tag, const std::string& opName);
} // namespace ops_hccl

#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_FALLBACK_H
#define CCU_FALLBACK_H

#include "hccl.h"
#include "alg_param.h"
#include "topo_host.h"
#include "executor_v2_base.h"

namespace ops_hccl {
HcclResult CheckCcuResNegotiation(HcclComm comm, const OpParam& param, bool localResAvailable);

HcclResult CheckCcuParamAndFallback(
    HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName);
} // namespace ops_hccl

#endif // CCU_FALLBACK_H

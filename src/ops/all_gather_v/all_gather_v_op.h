/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_ALL_GATHER_V_OP
#define OPS_HCCL_SRC_OPS_ALL_GATHER_V_OP

#include <string>
#include <memory>
#include "hccl.h"

#include "alg_param.h"
#include "executor_v2_base.h"
#include "alg_type.h"
#include "execute_selector.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts,
    const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
HcclResult AllGatherVOutPlace(void *sendBuf, void *recvBuf, uint64_t sendCount, const void *recvCounts,
    const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream, const std::string &tag);

HcclResult CheckAllGatherVInputPara(HcclComm comm, void *sendBuf, void *recvBuf);

HcclResult AllGatherVExecOp(HcclComm comm, OpParam &param);

HcclResult CheckCountAGV(const u64 count);

HcclResult CheckDataTypeAGV(const HcclDataType dataType);

std::string GetSupportDataTypeAGV();

HcclResult CalcBaseTopoInfoAllGatherV(HcclComm comm, OpParam &param, TopoInfo **topoInfo);

}  // namespace ops_hccl
#endif
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_REDUCE_SCATTER_V_OP
#define OPS_HCCL_SRC_OPS_REDUCE_SCATTER_V_OP

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

HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                             HcclReduceOp op, HcclComm comm, aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
HcclResult ReduceScatterVOutPlace(void *sendBuf, const void *sendDispls, const void *sendCounts, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag);

HcclResult ReduceScatterVExecOp(HcclComm comm, OpParam &param);

HcclResult CheckReduceScatterVInputPara(HcclComm comm, void *sendBuf, void *recvBuf, const void *sendCounts, const void *sendDispls, aclrtStream stream);

HcclResult GetAlgResReduceScatterV(HcclComm comm, OpParam &param, std::shared_ptr<InsCollAlgBase> &executor,
    TopoInfo* topoInfo, AlgResourceCtx** resCtx, aclrtNotify* notifies);

HcclResult CheckDataTypeRSV(const HcclDataType dataType, bool needReduce);

std::string GetSupportDataTypeRSV(bool needReduce);

}

#endif
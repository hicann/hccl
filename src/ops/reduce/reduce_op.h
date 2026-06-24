/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_REDUCE_OP
#define OPS_HCCL_SRC_OPS_REDUCE_OP

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

HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream);

HcclResult HcclReduceGraphMode(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, const char *group, aclrtStream stream, const char *tag, void **streams, size_t streamCount,
    void *scratchMemAddr, uint64_t scratchMemSize);

#ifdef __cplusplus
}
#endif

namespace ops_hccl {
HcclResult ReduceOutPlace(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag);

HcclResult ReduceOutPlaceGraphMode(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag, const ResPackGraphMode &resPack);

HcclResult ReduceConstructOpParam(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag, OpParam &param, OpMode opMode);

HcclResult ReduceExecOp(HcclComm comm, OpParam &param);

HcclResult CheckReduceInputPara(const HcclComm comm, const void *sendBuf, const void *recvBuf, const aclrtStream stream);

HcclResult GetAlgResReduce(HcclComm comm, OpParam &param, std::shared_ptr<InsCollAlgBase> &executor, TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceCtx **resCtx, aclrtNotify *notifies);

HcclResult ReduceInitAndCheck(HcclComm comm, void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t root, HcclReduceOp op,
    const aclrtStream stream, std::string &opTag);

HcclResult ReduceOutPlaceCommon(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, HcclComm comm, aclrtStream stream, const std::string &tag, OpMode opMode, const ResPackGraphMode &resPack);
HcclResult ReduceEntryLog(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    uint32_t root, aclrtStream stream, const std::string &tag, const std::string &opName);
}  // namespace ops_hccl

#endif
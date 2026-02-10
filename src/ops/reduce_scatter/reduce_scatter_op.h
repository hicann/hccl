/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_REDUCE_SCATTER_OP
#define OPS_HCCL_SRC_OPS_REDUCE_SCATTER_OP

#include "hccl.h"
#include "alg_param.h"
#include "executor_base.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream);

#ifdef __cplusplus
}
#endif

namespace ops_hccl_reduce_scatter {
HcclResult ReduceScatterOutPlace(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, const std::string &tag);

HcclResult ExecOp(HcclComm comm, OpParam &param);

HcclResult CalcBaseTopoInfo(HcclComm comm, OpParam &param, TopoInfo** topoInfo);

HcclResult GetAlgType(TopoInfo* topoInfo, HcclCMDType opType, AlgType& algType);

HcclResult SelectAlg(HcclComm comm, OpParam &param, TopoInfo* topoInfo, AlgType& algType, std::string &algName);

HcclResult GetAlgRes(HcclComm comm, OpParam &param, std::unique_ptr<ExecutorBase> &executor, TopoInfo *topoInfo,
    AlgType &algType, AlgResourceCtx **resCtx, aclrtNotify* notifies, DPUAlgResourceCtx &dpuResCtx);

HcclResult AllocAlgResource(HcclComm comm, const OpParam& param, AlgResourceRequest &resRequest,
    AlgResourceCtx* resCtxHost, aclrtNotify* notifies);

HcclResult AllocDpuAlgResource(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    DPUAlgResourceCtx &dpuResCtx);

HcclResult CheckCount(const u64 count);

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce);

std::string GetSupportDataType(bool needReduce);

HcclResult CheckReduceScatterInputPara(HcclComm comm, void *sendBuf, void *recvBuf);

int32_t HcclLaunchDpuKernel(uint64_t shmemPtr, int32_t DatSize);

HcclResult CheckHostDPUOnly(HcclComm comm, const TopoInfo &topoInfo, bool &hostDPUOnly);

bool CheckHCCLIndependentOp();
}

#endif
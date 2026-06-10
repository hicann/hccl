/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAllReduceMeshMem2Mem1D : CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct AllReduceMeshMem2Mem1DContext: CcuKernelCtxBase {
    const CcuKernelArgAllReduceMeshMem2Mem1D *arg;
    HcclDataType                      dataType;
    HcclDataType                      outputDataType;
    HcclReduceOp                      reduceOp;
    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable              myScratch;
    ccu::Variable              currentRankSliceInputOffset;
    ccu::Variable              currentRankSliceOutputOffset;
    ccu::Variable              normalSliceSize;
    ccu::Variable              lastSliceSize;
    ccu::Variable              mySliceSize;
    ccu::Variable              sliceOffset;
    ccu::Variable              isInputOutputEqual;
    ccu::Variable              sliceSize;
    std::vector<ccu::Event>    events;

    ccu::LocalAddr              srcMem;
    ccu::LocalAddr              localDstMem;
    ccu::RemoteAddr             remoteDstMem;
    std::vector<ccu::RemoteAddr> reduceScatterSrc;
    std::vector<ccu::LocalAddr> reduceScatterDst;
    GroupOpSizeVars goSize;
};

CcuResult CcuAllReduceMeshMem2Mem1DKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_REDUCE_MESH_1D_H_
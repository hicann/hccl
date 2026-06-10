/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALLGATHER_2DIES_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALLGATHER_2DIES_MESH_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

struct CcuKernelArgAllGather2DiesMeshMem2Mem1D : CcuKernelArgBase {
    uint64_t                                dimSize;
    uint32_t                                rankId;
    std::vector<uint32_t>                   rankIdGroup;
    bool                                    ifHandleSelfRank;
    std::vector<std::vector<uint32_t>>      subCommRanks;
    OpParam                                 opParam;
};

struct AllGather2DiesMeshMem2Mem1DContext : CcuKernelCtxBase {
    const CcuKernelArgAllGather2DiesMeshMem2Mem1D *arg;

    uint64_t rankSize;
    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable offSet;
    ccu::Variable sliceSize;
    std::vector<uint32_t> rankIdGroup;
    bool ifHandleSelfRank;
    GroupOpSizeVars localGoSize;

    ccu::Event event;
    ccu::Event localCopyEvent;
};

CcuResult CcuAllGather2DiesMeshMem2Mem1DKernel(CcuKernelArg arg);

}//namespace ops_hccl
#endif//HCCL_CCU_KERNEL_ALLGATHER_2DIES_MESH_1D_MEM2MEM_H
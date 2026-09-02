/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_MESH1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_MESH1D_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {

struct CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem : CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    uint32_t userRank;
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
};

struct AllGatherOmniPipeMesh1DMem2MemContext : CcuKernelCtxBase {
    CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem* arg;

    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable sliceSize;
    ccu::Variable sliceStride;
    ccu::Variable localCopyFlag;
    ccu::Variable inputOmniPipeSliceStride;
    GroupOpSizeVars groupOpSize;
    // ccu::Event event;
    std::vector<ccu::Event> events;
};

CcuResult CcuAllGatherOmniPipeMesh1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_ALL_GATHER_OMNIPIPE_MESH1D_MEM2MEM_H

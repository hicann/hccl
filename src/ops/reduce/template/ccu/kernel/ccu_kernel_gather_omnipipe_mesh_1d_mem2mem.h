/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H
#define HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H

#include <vector>
#include <ios>
#include <map>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgGatherOmniPipeMesh1DMem2Mem : CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    uint32_t rootId;
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
    uint32_t myrealrank;
};

struct GatherOmniPipeMesh1DMem2MemContext {
    CcuKernelArgGatherOmniPipeMesh1DMem2Mem* arg;
    uint64_t rankSize{0};
    uint32_t rankId{0};
    uint32_t rootId{0};
    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_RESERVED};

    std::vector<ccu::Variable> input;
    ccu::Variable output;
    std::vector<ccu::Variable> token;
    ccu::Variable localCopyFlag;
    ccu::Variable sliceSize;
    ccu::Variable ifNewRoot;
    ccu::Variable isFirstPiece;
    ccu::Variable isLastPiece;

    std::vector<ccu::Variable> sliceSizeOmniSliceStrideVec;
    std::vector<ccu::Variable> inputOmniSliceStrideVec;
    std::vector<ccu::Variable> outputOmniSliceStrideVec;
    
    std::vector<ccu::RemoteAddr> inputMem;
    std::vector<ccu::LocalAddr> outputMem;
    ccu::Event event;
};

CcuResult CcuGatherOmniPipeMesh1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_GATHER_OMNIPIPE_MESH_1D_MEM2MEM_H
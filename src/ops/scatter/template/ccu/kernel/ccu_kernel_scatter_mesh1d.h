/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_SCATTER_MESH_1D__H
#define HCCL_CCU_KERNEL_SCATTER_MESH_1D__H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgScatterMesh1D : CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    uint32_t rootId;
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
};

struct ScatterMesh1DContext {
    const CcuKernelArgScatterMesh1D *arg;

    uint64_t rankSize{0};
    uint32_t rankId{0};
    uint32_t rootId{0};
    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_RESERVED};
    HcclDataType outputDataType{HcclDataType::HCCL_DATA_TYPE_RESERVED};

    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable currentRankSliceInputOffset;
    ccu::Variable outputSliceStride;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable normalSliceSize;
    ccu::Variable lastSliceSize;
    ccu::Variable repeatNum;
    ccu::Variable isInputOutputEqual;
    ccu::Variable flag;

    std::vector<ccu::LocalAddr> inputMem;
    std::vector<ccu::RemoteAddr> outputMem;
    ccu::Event event;
};

CcuResult CcuScatterMesh1DKernel(CcuKernelArg arg);
} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_SCATTER_MESH_1D__H

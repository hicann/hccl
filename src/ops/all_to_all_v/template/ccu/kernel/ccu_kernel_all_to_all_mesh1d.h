/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAlltoAllMesh1D: CcuKernelArgBase{
    uint64_t                                rankSize;
    uint32_t                                rankId;
    bool                                    loadFromMem;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct AlltoAllMesh1DContext: CcuKernelCtxBase {
    const CcuKernelArgAlltoAllMesh1D *arg;
    
    HcclDataType dataType;
    HcclDataType outputDataType;
    ccu::Variable repeatNum;
    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable sliceSize;
    ccu::Variable srcStride;
    ccu::Variable srcOffset;
    ccu::Variable dstOffset;
    GroupOpSizeVars goSize;

    ccu::LocalAddr                   myDst;
    std::vector<ccu::RemoteAddr>     dstAddr;
    std::vector<ccu::LocalAddr>      srcAddr;
    std::vector<ccu::Variable>         dstToken;
    std::vector<ccu::Variable>         srcToken;
    ccu::Event event;
    ccu::Variable flag; // 用以判断是否是第一次重复
    bool loadFromMem = false;
};

CcuResult CcuAlltoAllMesh1DKernel(CcuKernelArg arg);

}// namespace ops_hccl
#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_MESH_1D_H

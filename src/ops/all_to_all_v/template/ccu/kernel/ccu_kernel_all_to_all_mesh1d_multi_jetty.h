/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_TO_ALL_MESH1D_MULTI_JETTY_H
#define HCCL_CCU_KERNEL_ALL_TO_ALL_MESH1D_MULTI_JETTY_H

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAllToAllMesh1DMultiJetty : CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    OpParam                                 opParam;
    std::vector<uint32_t>                   jettyNums;
};

struct AllToAllMesh1DMultiJettyContext : CcuKernelCtxBase {
    const CcuKernelArgAllToAllMesh1DMultiJetty *arg;

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable token;
    std::vector<ccu::Variable> peerInput;
    std::vector<ccu::Variable> peerOutput;
    std::vector<ccu::Variable> peerToken;
    ccu::Variable sliceSize;
    ccu::Variable srcStride;
    ccu::Variable srcOffset;
    ccu::Variable dstOffset;
    GroupOpSizeVars goSize;
    std::vector<ccu::Variable> jettySlice;
    std::vector<ccu::Variable> jettySliceTail;
    std::vector<ccu::Event>    eventList;
};

CcuResult CcuAllToAllMesh1DMultiJettyKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_ALL_TO_ALL_MESH1D_MULTI_JETTY_H

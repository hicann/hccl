/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM
#define HCCL_CCU_KERNEL_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

#ifndef NHR_STEP_INFO_DEFINED
#define NHR_STEP_INFO_DEFINED
using NHRStepInfo = struct NHRStepInfoDef {
    uint32_t step = 0;
    uint32_t myRank = 0;
    uint32_t nSlices;
    uint32_t toRank = 0;
    uint32_t fromRank = 0;
    std::vector<uint32_t> txSliceIdxs;
    std::vector<uint32_t> rxSliceIdxs;

    NHRStepInfoDef() : nSlices(0)
    {
    }
};
#endif

struct CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem : CcuKernelArgBase {
    uint64_t                                rankSize;
    uint32_t                                rankId;
    OpParam                                 opParam;
    uint32_t                                jettyNum;
    std::vector<NHRStepInfo>                stepInfoVector;
    std::map<uint32_t, uint32_t>            rank2ChannelIdx;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct AllGatherNHR1DMultiJettyMem2MemContext : CcuKernelCtxBase {
    const CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem *arg;

    uint64_t                                localSize;
    uint64_t                                myRankIdx;

    ccu::Variable                             input;
    std::vector<ccu::Variable>                output;
    std::vector<ccu::Variable>                token;
    ccu::Variable                             sliceSize;
    ccu::Variable                             sliceSizePerJetty;
    ccu::Variable                             lastSliceSizePerJetty;
    ccu::Variable                             repeatNumInv;
    ccu::Variable                             inputSliceStride;
    ccu::Variable                             outputSliceStride;
    ccu::Variable                             inputRepeatStride;
    ccu::Variable                             outputRepeatStride;
    ccu::Variable                             isInputOutputEqual;
    GroupOpSizeVars                         groupOpSize;
    ccu::Event                                event;
    std::vector<ccu::Variable>                outputSliceOffset;
    ccu::Variable                             constVar1;
    ccu::LocalAddr                            srcMem;
    ccu::RemoteAddr                           dstMem;
    ccu::LocalAddr                            myDstMem;
    ccu::Variable                             repeatTimeflag;
    ccu::Variable                             tmpCopyRepeatNumInv;
};

CcuResult CcuAllGatherNHR1DMultiJettyMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_GATHER_NHR_1D_MULTIJETTY_MEM2MEM

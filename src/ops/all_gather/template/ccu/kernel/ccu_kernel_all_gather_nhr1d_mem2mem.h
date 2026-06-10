/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALL_GATHER_NHR1D_MEM2MEM
#define HCCL_CCU_KERNEL_ALL_GATHER_NHR1D_MEM2MEM

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

#ifndef NHR_STEP_INFO_DEFINED
#define NHR_STEP_INFO_DEFINED
using NHRStepInfo = struct NHRStepInfoDef {
    u32 step = 0;
    u32 myRank = 0;
    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    NHRStepInfoDef() : nSlices(0)
    {
    }
};
#endif

struct CcuKernelArgAllGatherNHR1D : CcuKernelArgBase {
    uint64_t                                dimSize;
    uint64_t                                mySubCommRankId;
    uint64_t                                axisId;
    std::vector<NHRStepInfo>                stepInfoVector;
    std::map<u32, u32>                      rank2ChannelIdx;
    OpParam                                 opParam;
    std::vector<std::vector<uint32_t>>      subCommRanks;
    uint32_t                                axisSize;
};

struct AllGatherNHR1DMem2MemContext : CcuKernelCtxBase {
    const CcuKernelArgAllGatherNHR1D *arg;

    uint64_t localSize;
    uint64_t myRankIdx;
    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable die0Size;
    ccu::Variable die1Size;
    ccu::Variable repeatNum;
    ccu::Variable inputSliceStride;
    ccu::Variable outputSliceStride;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable isInputOutputEqual;
    ccu::Variable die0LastSize;
    ccu::Variable die1LastSize;
    ccu::Event localEvent;
    ccu::Variable repeatTimeflag;
    std::vector<ccu::Variable> outputSliceOffset;
    ccu::Variable myrankInputSliceOffset;
    ccu::LocalAddr srcMem;
    ccu::RemoteAddr dstMem;
    ccu::LocalAddr localDst;
    ccu::Variable constVar1;
};

CcuResult CcuAllGatherNHR1DMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCL_CCU_KERNEL_ALL_GATHER_NHR1D_MEM2MEM
